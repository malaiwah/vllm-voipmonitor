#!/usr/bin/env python3
"""PoC v48: EXL3 config exploration — mul1 codebook, up_proj, 70 experts, Hadamard 64.

1. mul1 codebook: Alternative hash multiplier (0x83DCD12D vs mcg 0xCBAC1FED).
   Test if different codebook gives better MSRT results.

2. up_proj: Test MSRT on up_proj (third projection, never tested).

3. All 70 experts: Run MSRT with all 70 experts for robust statistics.

4. Hadamard block size 64: Test if smaller Hadamard (64 instead of 128)
   changes MSRT quality.
"""
from __future__ import annotations
import argparse, json, math, os, sys, types, importlib.util, time
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
    tcp = m.tensor_core_perm; tcpi = m.tensor_core_perm_i; qtf = m.quantize_tiles; cbs = m.codebook_scale
    return ext, ghd, tcp, tcpi, qtf, cbs

def block_rms(x, dim, keepdim=False):
    return x.square().mean(dim=dim, keepdim=keepdim).sqrt()

def regularize(w, device, ghd, cbs, had_k=128, had_n=128):
    k, n = w.shape
    g = torch.Generator(device="cpu").manual_seed(0)
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
    return w, su, sv

def quantize_trellis_raw(data, K, device, tcp, tcpi, qtf, use_mul1=False):
    k, n = data.shape; tiles_n = n // 16; weight_q = torch.zeros_like(data)
    qa = {"K": K, "mcg": True}
    if use_mul1:
        qa = {"K": K, "mul1": True}
    perm = tcp(device); perm_i = tcpi(device)
    for bi in range(0, k, 16):
        rows = data[bi:bi+16]
        tiles = rows.reshape(16, tiles_n, 16).permute(1, 0, 2).reshape(tiles_n, 256)
        tiles = tiles[:, perm].contiguous()
        quant_w, _ = qtf(tiles, qa)
        quant_w = quant_w[:, perm_i].reshape(tiles_n, 16, 16).permute(1, 0, 2).reshape(16, n)
        weight_q[bi:bi+16] = quant_w
    return weight_q

def rescaled_trellis(base_q, residual, K_res, device, tcp, tcpi, qtf, cbs, use_mul1=False):
    residual_rms = residual.square().mean().sqrt().item()
    if residual_rms < 1e-12: return base_q
    scale = abs(cbs) / residual_rms
    scaled = residual * scale
    quant = quantize_trellis_raw(scaled, K_res, device, tcp, tcpi, qtf, use_mul1)
    return base_q + quant / scale

def run_experiment(data_dir, device, ext, ghd, tcp, tcpi, qtf, cbs_scale):
    results = {}
    all_methods = {}

    # === Part 1: mul1 codebook vs mcg ===
    print(f"\n  Part 1: mul1 vs mcg codebook", flush=True)
    f = data_dir / f"layer10_all_gate_proj.pt"
    if not f.exists(): return results
    all_experts = torch.load(f, map_location="cpu")
    n_experts = min(5, all_experts.shape[0])
    k, n = all_experts.shape[1], all_experts.shape[2]

    for ei in range(n_experts):
        w = all_experts[ei].to(device)
        w_reg, _, _ = regularize(w, device, ghd, cbs_scale)
        del w
        qk2 = quantize_trellis_raw(w_reg, 2, device, tcp, tcpi, qtf)
        r2 = w_reg - qk2

        # MSRT 8bpw with mcg (baseline)
        s1 = rescaled_trellis(qk2, r2, 1, device, tcp, tcpi, qtf, cbs_scale)
        r_s1 = w_reg - s1
        s2 = rescaled_trellis(s1, r_s1, 2, device, tcp, tcpi, qtf, cbs_scale)
        r_s2 = w_reg - s2
        recon_mcg = rescaled_trellis(s2, r_s2, 3, device, tcp, tcpi, qtf, cbs_scale)
        mse_mcg = (w_reg - recon_mcg).pow(2).mean().item()
        all_methods.setdefault("MSRT_8bpw_mcg", {"mses": [], "bpw": 8})["mses"].append(mse_mcg)

        # MSRT 8bpw with mul1 (all stages)
        s1m = rescaled_trellis(qk2, r2, 1, device, tcp, tcpi, qtf, cbs_scale, use_mul1=True)
        r_s1m = w_reg - s1m
        s2m = rescaled_trellis(s1m, r_s1m, 2, device, tcp, tcpi, qtf, cbs_scale, use_mul1=True)
        r_s2m = w_reg - s2m
        recon_mul1 = rescaled_trellis(s2m, r_s2m, 3, device, tcp, tcpi, qtf, cbs_scale, use_mul1=True)
        mse_mul1 = (w_reg - recon_mul1).pow(2).mean().item()
        all_methods.setdefault("MSRT_8bpw_mul1", {"mses": [], "bpw": 8})["mses"].append(mse_mul1)

        # MSRT with mul1 for residual stages only (K2 base uses mcg)
        s1h = rescaled_trellis(qk2, r2, 1, device, tcp, tcpi, qtf, cbs_scale, use_mul1=True)
        r_s1h = w_reg - s1h
        s2h = rescaled_trellis(s1h, r_s1h, 2, device, tcp, tcpi, qtf, cbs_scale, use_mul1=True)
        r_s2h = w_reg - s2h
        recon_hybrid = rescaled_trellis(s2h, r_s2h, 3, device, tcp, tcpi, qtf, cbs_scale, use_mul1=True)
        mse_hybrid = (w_reg - recon_hybrid).pow(2).mean().item()
        all_methods.setdefault("MSRT_8bpw_mul1_residual", {"mses": [], "bpw": 8})["mses"].append(mse_hybrid)

        del w_reg, qk2, r2, s1, r_s1, s2, r_s2, recon_mcg, s1m, r_s1m, s2m, r_s2m, recon_mul1
        del s1h, r_s1h, s2h, r_s2h, recon_hybrid
        torch.cuda.empty_cache()

    # === Part 2: up_proj ===
    print(f"\n  Part 2: up_proj", flush=True)
    f_up = data_dir / f"layer10_all_up_proj.pt"
    if f_up.exists():
        up_experts = torch.load(f_up, map_location="cpu")
        n_up = min(5, up_experts.shape[0])
        for ei in range(n_up):
            w = up_experts[ei].to(device)
            w_reg, _, _ = regularize(w, device, ghd, cbs_scale)
            del w
            qk2 = quantize_trellis_raw(w_reg, 2, device, tcp, tcpi, qtf)
            r2 = w_reg - qk2
            s1 = rescaled_trellis(qk2, r2, 1, device, tcp, tcpi, qtf, cbs_scale)
            r_s1 = w_reg - s1
            s2 = rescaled_trellis(s1, r_s1, 2, device, tcp, tcpi, qtf, cbs_scale)
            r_s2 = w_reg - s2
            recon = rescaled_trellis(s2, r_s2, 3, device, tcp, tcpi, qtf, cbs_scale)
            mse = (w_reg - recon).pow(2).mean().item()
            all_methods.setdefault("MSRT_8bpw_up_proj", {"mses": [], "bpw": 8})["mses"].append(mse)
            del w_reg, qk2, r2, s1, r_s1, s2, r_s2, recon
            torch.cuda.empty_cache()

    # === Part 3: All 70 experts ===
    print(f"\n  Part 3: All 70 experts", flush=True)
    f_gate = data_dir / f"layer10_all_gate_proj.pt"
    if f_gate.exists():
        gate_experts = torch.load(f_gate, map_location="cpu")
        n_all = gate_experts.shape[0]
        all_70_mses = []
        for ei in range(n_all):
            w = gate_experts[ei].to(device)
            w_reg, _, _ = regularize(w, device, ghd, cbs_scale)
            del w
            qk2 = quantize_trellis_raw(w_reg, 2, device, tcp, tcpi, qtf)
            r2 = w_reg - qk2
            # MSRT 6bpw: K2+K1+K3trsc
            s1 = rescaled_trellis(qk2, r2, 1, device, tcp, tcpi, qtf, cbs_scale)
            r_s1 = w_reg - s1
            recon = rescaled_trellis(s1, r_s1, 3, device, tcp, tcpi, qtf, cbs_scale)
            mse = (w_reg - recon).pow(2).mean().item()
            all_70_mses.append(mse)
            del w_reg, qk2, r2, s1, r_s1, recon
            torch.cuda.empty_cache()

        import statistics
        mean_mse = statistics.mean(all_70_mses)
        std_mse = statistics.stdev(all_70_mses)
        cv = std_mse / mean_mse * 100
        print(f"    70 experts MSRT 6bpw: mean={mean_mse:.4e} std={std_mse:.4e} CV={cv:.2f}%", flush=True)
        print(f"    Min={min(all_70_mses):.4e} Max={max(all_70_mses):.4e} ratio={max(all_70_mses)/min(all_70_mses):.4f}", flush=True)
        all_methods["MSRT_6bpw_70experts"] = {"mses": all_70_mses, "bpw": 6}

    # === Part 4: Hadamard block 64 ===
    print(f"\n  Part 4: Hadamard block 64", flush=True)
    if f_gate.exists():
        gate_experts = torch.load(f_gate, map_location="cpu")
        n_h64 = min(5, gate_experts.shape[0])
        for ei in range(n_h64):
            w = gate_experts[ei].to(device)
            # Use Hadamard block size 64 instead of 128
            w_reg, _, _ = regularize(w, device, ghd, cbs_scale, had_k=64, had_n=64)
            del w
            qk2 = quantize_trellis_raw(w_reg, 2, device, tcp, tcpi, qtf)
            r2 = w_reg - qk2
            s1 = rescaled_trellis(qk2, r2, 1, device, tcp, tcpi, qtf, cbs_scale)
            r_s1 = w_reg - s1
            recon = rescaled_trellis(s1, r_s1, 3, device, tcp, tcpi, qtf, cbs_scale)
            mse = (w_reg - recon).pow(2).mean().item()
            all_methods.setdefault("MSRT_6bpw_hadamard64", {"mses": [], "bpw": 6})["mses"].append(mse)
            del w_reg, qk2, r2, s1, r_s1, recon
            torch.cuda.empty_cache()

    # Print results
    print(f"\n  {'Method':<35} {'bpw':>5} {'avg MSE':>12}", flush=True)
    print(f"  {'-'*55}", flush=True)
    for name in sorted(all_methods.keys(), key=lambda x: (all_methods[x]["bpw"], x)):
        r = all_methods[name]
        avg = sum(r["mses"]) / len(r["mses"])
        print(f"  {name:<35} {r['bpw']:>5.0f} {avg:>12.4e}  (n={len(r['mses'])})", flush=True)

    # Compare mul1 vs mcg
    if "MSRT_8bpw_mcg" in all_methods and "MSRT_8bpw_mul1" in all_methods:
        mcg_mse = sum(all_methods["MSRT_8bpw_mcg"]["mses"]) / len(all_methods["MSRT_8bpw_mcg"]["mses"])
        mul1_mse = sum(all_methods["MSRT_8bpw_mul1"]["mses"]) / len(all_methods["MSRT_8bpw_mul1"]["mses"])
        ratio = mul1_mse / mcg_mse
        print(f"\n  mul1 vs mcg: {ratio:.4f}x ({'better' if ratio < 1 else 'worse'})", flush=True)

    results["methods"] = {k: {"bpw": v["bpw"], "avg_mse": sum(v["mses"])/len(v["mses"]), "n_experts": len(v["mses"])}
                          for k, v in all_methods.items()}
    return results

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--data-dir", default="/tmp/poc_residual/glm52_data")
    ap.add_argument("--out", default="/tmp/poc_residual/results_v48.json")
    args = ap.parse_args()
    dev = torch.device(args.device)
    print(f"Device: {dev}  GPU: {torch.cuda.get_device_name(0)}", flush=True)
    ext, ghd, tcp, tcpi, qtf, cbs_scale = _bootstrap()
    results = run_experiment(Path(args.data_dir), dev, ext, ghd, tcp, tcpi, qtf, cbs_scale)
    Path(args.out).write_text(json.dumps(results, indent=2, default=str))
    print(f"\nResults saved to {args.out}", flush=True)

if __name__ == "__main__":
    main()
