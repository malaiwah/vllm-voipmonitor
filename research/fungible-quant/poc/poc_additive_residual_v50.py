#!/usr/bin/env python3
"""PoC v50: Measure MSRT at LOW bitrates (3-5 bpw) that were never tested.

The v41 Pareto starts at 5bpw. We need K2+K1 (3bpw), K2+K2 (4bpw),
K3+K1 (4bpw), K2+K1+K1 (4bpw) to complete the Pareto and validate
the H100 cartridge recommendation.
"""
from __future__ import annotations
import argparse, json, math, os, sys, types, importlib.util, gc
from pathlib import Path
import torch

EXL3_PKG = "/opt/fruit-pip/exllamav3"

def _bootstrap():
    pkg = types.ModuleType("exllamav3"); pkg.__path__ = [EXL3_PKG]; sys.modules["exllamav3"] = pkg
    for sub in ["util", "modules", "modules.quant", "modules.quant.exl3_lib"]:
        full = f"exllamav3.{sub}"; m = types.ModuleType(full)
        m.__path__ = [f"{EXL3_PKG}/{sub.replace('.', '/')}"]; sys.modules[full] = m
    class _DPB:
        def __init__(self, *a, **kw): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def update(self, *a): pass
        def new_task(self, *a, **kw): pass
    _s = types.ModuleType("exllamav3.util.progress"); _s.ProgressBar = _DPB; sys.modules["exllamav3.util.progress"] = _s
    _s = types.ModuleType("exllamav3.util.memory"); _s.free_mem = lambda: None; _s.list_gpu_tensors = lambda: []; sys.modules["exllamav3.util.memory"] = _s
    _s = types.ModuleType("exllamav3.util"); _s.__path__ = [f"{EXL3_PKG}/util"]; _s.cuda_sync_active = lambda *a, **kw: torch.cuda.synchronize(); sys.modules["exllamav3.util"] = _s
    _s = types.ModuleType("exllamav3.util.tensor"); _s.save_tensor_image = lambda *a, **kw: None; sys.modules["exllamav3.util.tensor"] = _s
    spec = importlib.util.spec_from_file_location("exllamav3.ext", f"{EXL3_PKG}/ext.py")
    m = importlib.util.module_from_spec(spec); sys.modules["exllamav3.ext"] = m; spec.loader.exec_module(m)
    ext = m.exllamav3_ext
    spec = importlib.util.spec_from_file_location("exllamav3.util.hadamard", f"{EXL3_PKG}/util/hadamard.py")
    m = importlib.util.module_from_spec(spec); sys.modules["exllamav3.util.hadamard"] = m; spec.loader.exec_module(m)
    ghd = m.get_hadamard_dt
    spec = importlib.util.spec_from_file_location("exllamav3.modules.quant.exl3_lib.quantize", f"{EXL3_PKG}/modules/quant/exl3_lib/quantize.py")
    m = importlib.util.module_from_spec(spec); sys.modules["exllamav3.modules.quant.exl3_lib.quantize"] = m; spec.loader.exec_module(m)
    return ext, ghd, m.tensor_core_perm, m.tensor_core_perm_i, m.quantize_tiles, m.codebook_scale

def block_rms(x, dim, keepdim=False):
    return x.square().mean(dim=dim, keepdim=keepdim).sqrt()

def regularize(w, device, ghd, cbs, had_k=128, had_n=128, seed=0):
    k, n = w.shape
    g = torch.Generator(device="cpu").manual_seed(seed)
    su = (torch.randn(k, generator=g).sign() + 1e-5).sign().float().to(device)
    sv = (torch.randn(n, generator=g).sign() + 1e-5).sign().float().to(device)
    out_scales = block_rms(w, dim=0, keepdim=True)
    mean = out_scales.mean().item()
    if mean > 1e-30: out_scales = out_scales / mean
    sv = (sv * out_scales + 1e-10).float()
    w = (w / sv).contiguous()
    had_n_mat = ghd(had_n, device, torch.float, 1.0 / math.sqrt(had_n))
    w = (w.view(k, n // had_n, had_n) @ had_n_mat).view(k, n).contiguous()
    in_scales = block_rms(w, dim=1, keepdim=True).clamp(min=1e-30)
    su = (su.unsqueeze(1) * in_scales / (-cbs) + 1e-10).float()
    w = (w / su).contiguous()
    had_k_mat = ghd(had_k, device, torch.float, 1.0 / math.sqrt(had_k))
    w = (had_k_mat @ w.view(k // had_k, had_k, n)).view(k, n).contiguous()
    return w

def quantize_trellis_raw(data, K, device, tcp, tcpi, qtf):
    k, n = data.shape; tiles_n = n // 16; weight_q = torch.zeros_like(data)
    qa = {"K": K, "mcg": True}
    perm = tcp(device); perm_i = tcpi(device)
    for bi in range(0, k, 16):
        rows = data[bi:bi+16]
        tiles = rows.reshape(16, tiles_n, 16).permute(1, 0, 2).reshape(tiles_n, 256)
        tiles = tiles[:, perm].contiguous()
        quant_w, _ = qtf(tiles, qa)
        quant_w = quant_w[:, perm_i].reshape(tiles_n, 16, 16).permute(1, 0, 2).reshape(16, n)
        weight_q[bi:bi+16] = quant_w
    return weight_q

def rescaled_trellis(base_q, residual, K_res, device, tcp, tcpi, qtf, cbs):
    residual_rms = residual.square().mean().sqrt().item()
    if residual_rms < 1e-12: return base_q
    scale = abs(cbs) / residual_rms
    scaled = residual * scale
    quant = quantize_trellis_raw(scaled, K_res, device, tcp, tcpi, qtf)
    return base_q + quant / scale

def run_experiment(data_dir, device, ghd, tcp, tcpi, qtf, cbs):
    results = {}
    f = data_dir / "layer10_all_gate_proj.pt"
    if not f.exists():
        print("Data not found!", flush=True)
        return results
    experts = torch.load(f, map_location="cpu")
    n = min(10, experts.shape[0])

    # All configs to test, grouped by total bpw
    configs = [
        # (name, total_bpw, stages)
        # stages: list of K values, first is base, rest are rescaled residuals
        ("K2 only", 2.0, [2]),
        ("K3 only", 3.0, [3]),
        ("K2+K1trsc", 3.0, [2, 1]),
        ("K2+K2trsc", 4.0, [2, 2]),
        ("K3+K1trsc", 4.0, [3, 1]),
        ("K2+K1trsc+K1trsc", 4.0, [2, 1, 1]),
        ("K4 only", 4.0, [4]),
        ("K2+K3trsc", 5.0, [2, 3]),
        ("K3+K2trsc", 5.0, [3, 2]),
        ("K2+K1trsc+K2trsc", 5.0, [2, 1, 2]),
        ("K2+K2trsc+K1trsc", 5.0, [2, 2, 1]),
        # Also test K2+K1 without rescaling (to see if rescaling matters at low K)
        ("K2+K1_norescale", 3.0, [2, 1]),
    ]

    print(f"Testing {len(configs)} configs on {n} experts", flush=True)
    print(f"{'Config':<30} {'bpw':>5} {'avg MSE':>12} {'min MSE':>12} {'max MSE':>12}", flush=True)
    print("-" * 75, flush=True)

    for name, bpw, stages in configs:
        mses = []
        for ei in range(n):
            w = experts[ei].to(device)
            w_reg = regularize(w, device, ghd, cbs)
            del w

            if "_norescale" in name:
                # No rescaling: quantize residual directly
                base = quantize_trellis_raw(w_reg, stages[0], device, tcp, tcpi, qtf)
                for sk in stages[1:]:
                    residual = w_reg - base
                    base = base + quantize_trellis_raw(residual, sk, device, tcp, tcpi, qtf)
                mse = (w_reg - base).pow(2).mean().item()
            else:
                # Standard MSRT with rescaling
                base = quantize_trellis_raw(w_reg, stages[0], device, tcp, tcpi, qtf)
                for sk in stages[1:]:
                    residual = w_reg - base
                    base = rescaled_trellis(base, residual, sk, device, tcp, tcpi, qtf, cbs)
                mse = (w_reg - base).pow(2).mean().item()

            mses.append(mse)
            del w_reg, base
            torch.cuda.empty_cache()

        avg = sum(mses) / len(mses)
        results[name] = {"bpw": bpw, "stages": stages, "avg_mse": avg,
                         "min_mse": min(mses), "max_mse": max(mses), "n_experts": n}
        print(f"{name:<30} {bpw:>5.1f} {avg:>12.4e} {min(mses):>12.4e} {max(mses):>12.4e}", flush=True)

    # Also test on layer 40 and a different projection for validation
    for layer in [40]:
        f2 = data_dir / f"layer{layer}_all_gate_proj.pt"
        if f2.exists():
            experts2 = torch.load(f2, map_location="cpu")
            n2 = min(5, experts2.shape[0])
            print(f"\n--- Validation on layer {layer} ({n2} experts) ---", flush=True)
            print(f"{'Config':<30} {'bpw':>5} {'avg MSE':>12}", flush=True)
            print("-" * 50, flush=True)
            for name, bpw, stages in configs:
                if "_norescale" in name:
                    continue  # skip norescale for validation
                mses = []
                for ei in range(n2):
                    w = experts2[ei].to(device)
                    w_reg = regularize(w, device, ghd, cbs)
                    del w
                    base = quantize_trellis_raw(w_reg, stages[0], device, tcp, tcpi, qtf)
                    for sk in stages[1:]:
                        residual = w_reg - base
                        base = rescaled_trellis(base, residual, sk, device, tcp, tcpi, qtf, cbs)
                    mses.append((w_reg - base).pow(2).mean().item())
                    del w_reg, base
                    torch.cuda.empty_cache()
                avg = sum(mses) / len(mses)
                print(f"{name:<30} {bpw:>5.1f} {avg:>12.4e}", flush=True)

    # Summary table grouped by bpw
    print("\n=== Pareto by bpw (layer 10) ===", flush=True)
    by_bpw = {}
    for name, r in results.items():
        bpw = r["bpw"]
        by_bpw.setdefault(bpw, []).append((name, r["avg_mse"]))
    for bpw in sorted(by_bpw.keys()):
        entries = sorted(by_bpw[bpw], key=lambda x: x[1])
        best = entries[0]
        print(f"  {bpw:.1f} bpw: best={best[0]} MSE={best[1]:.4e}", flush=True)
        for name, mse in entries[1:]:
            ratio = mse / best[1]
            print(f"         {name} MSE={mse:.4e} ({ratio:.2f}x worse)", flush=True)

    return results

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--data-dir", default="/tmp/poc_residual/glm52_data")
    ap.add_argument("--out", default="/tmp/poc_residual/results_v50.json")
    args = ap.parse_args()
    dev = torch.device(args.device)
    print(f"Device: {dev}  GPU: {torch.cuda.get_device_name(0)}", flush=True)
    ext, ghd, tcp, tcpi, qtf, cbs = _bootstrap()
    print(f"codebook_scale = {cbs}", flush=True)
    results = run_experiment(Path(args.data_dir), dev, ghd, tcp, tcpi, qtf, cbs)
    Path(args.out).write_text(json.dumps(results, indent=2, default=str))
    print(f"\nResults saved to {args.out}", flush=True)

if __name__ == "__main__":
    main()
