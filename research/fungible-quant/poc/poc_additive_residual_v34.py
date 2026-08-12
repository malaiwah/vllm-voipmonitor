#!/usr/bin/env python3
"""PoC v34: Trellis-on-residual — use EXL3 trellis for residual quantization.

KEY NEW IDEA: Instead of scalar Lloyd-Max on the residual, use the EXL3
trellis quantizer (TCQ) on the residual. TCQ achieves 40% lower distortion
than scalar quantization on Gaussian sources (QTIP: 0.071 vs 0.118).

Test:
  K4 + K2_on_residual (6 bpw) vs K4 + 2LM (6 bpw)
  K4 + K3_on_residual (7 bpw) vs K4 + 3LM (7 bpw)
  K3 + K2_on_residual (5 bpw) vs K3 + 2LM (5 bpw)

The trellis on residual should beat scalar LM because:
1. TCQ uses 2^L states (larger codebook) vs 2^K levels
2. Viterbi finds optimal path, not greedy nearest-neighbor
3. EXL3 trellis is designed for Gaussian sources (residual is Gaussian)
"""
from __future__ import annotations
import argparse, json, math, os, sys, types, importlib.util, time
from pathlib import Path
import torch

EXL3_PKG = "/opt/fruit-pip/exllamav3"
HAD_K, HAD_N = 128, 128

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

def regularize(w, device, ghd, cbs):
    k, n = w.shape
    g = torch.Generator(device="cpu").manual_seed(0)
    su = (torch.randn(k, generator=g).sign() + 1e-5).sign().float().to(device)
    sv = (torch.randn(n, generator=g).sign() + 1e-5).sign().float().to(device)
    out_scales = block_rms(w, dim=0, keepdim=True)
    mean = out_scales.mean().item()
    if mean > 1e-30: out_scales = out_scales / mean
    sv = (sv * out_scales + 1e-10).float()
    w = (w / sv).contiguous()
    had_n = ghd(HAD_N, device, torch.float, 1.0 / math.sqrt(HAD_N))
    w = (w.view(k, n // HAD_N, HAD_N) @ had_n).view(k, n).contiguous()
    in_scales = block_rms(w, dim=1, keepdim=True).clamp(min=1e-30)
    su = (su.unsqueeze(1) * in_scales / (-cbs) + 1e-10).float()
    w = (w / su).contiguous()
    had_k = ghd(HAD_K, device, torch.float, 1.0 / math.sqrt(HAD_K))
    w = (had_k @ w.view(k // HAD_K, HAD_K, n)).view(k, n).contiguous()
    return w, su, sv

def quantize_trellis(w_reg, K, device, tcp, tcpi, qtf):
    """Standard EXL3 trellis quantization."""
    k, n = w_reg.shape; tiles_n = n // 16; weight_q = torch.zeros_like(w_reg)
    qa = {"K": K, "mcg": True}; perm = tcp(device); perm_i = tcpi(device)
    for bi in range(0, k, 16):
        rows = w_reg[bi:bi+16]
        tiles = rows.reshape(16, tiles_n, 16).permute(1, 0, 2).reshape(tiles_n, 256)
        tiles = tiles[:, perm].contiguous()
        quant_w, _ = qtf(tiles, qa)
        quant_w = quant_w[:, perm_i].reshape(tiles_n, 16, 16).permute(1, 0, 2).reshape(16, n)
        weight_q[bi:bi+16] = quant_w
    return weight_q

def quantize_trellis_with_residual(base_q, residual, K_res, device, tcp, tcpi, qtf):
    """Quantize residual with EXL3 trellis, add to base reconstruction."""
    # The residual is already in regularized space
    # We need to quantize it with the trellis
    # But the trellis expects weights, not residuals
    # The trellis should work on any Gaussian-like data
    k, n = residual.shape; tiles_n = n // 16; weight_q = torch.zeros_like(residual)
    qa = {"K": K_res, "mcg": True}; perm = tcp(device); perm_i = tcpi(device)
    for bi in range(0, k, 16):
        rows = residual[bi:bi+16]
        tiles = rows.reshape(16, tiles_n, 16).permute(1, 0, 2).reshape(tiles_n, 256)
        tiles = tiles[:, perm].contiguous()
        quant_w, _ = qtf(tiles, qa)
        quant_w = quant_w[:, perm_i].reshape(tiles_n, 16, 16).permute(1, 0, 2).reshape(16, n)
        weight_q[bi:bi+16] = quant_w
    return base_q + weight_q

def get_tiles(r, k, n):
    tnk, tnn = k // 16, n // 16
    return r.view(tnk, 16, tnn, 16).permute(0, 2, 1, 3).reshape(tnk * tnn, 256)

def tiles_to_matrix(tiles, k, n):
    tnk, tnn = k // 16, n // 16
    return tiles.reshape(tnk, tnn, 16, 16).permute(0, 2, 1, 3).reshape(k, n)

def cluster_by_sigma(tiles, n_clusters, device):
    n_tiles = tiles.shape[0]
    sigmas = tiles.std(dim=1).clamp(min=1e-12)
    sorted_sigmas, sort_idx = sigmas.sort()
    cluster_size = n_tiles // n_clusters
    cluster_id = torch.zeros(n_tiles, dtype=torch.long, device=device)
    cluster_id[sort_idx] = torch.arange(n_tiles, device=device) // cluster_size
    cluster_id = cluster_id.clamp(max=n_clusters - 1)
    return cluster_id

def lloyd_max_clustered(tiles, cluster_id, n_bits, n_clusters, device):
    n_levels = 2 ** n_bits
    result = torch.empty_like(tiles)
    for cid in range(n_clusters):
        mask = cluster_id == cid
        if mask.sum() == 0: continue
        ct = tiles[mask]
        cf = ct.flatten()
        sigma = cf.std().item()
        if sigma < 1e-12:
            result[mask] = 0.0
            continue
        levels = torch.linspace(-3 * sigma, 3 * sigma, n_levels, device=device)
        for _ in range(12):
            d = (cf.unsqueeze(1) - levels.unsqueeze(0)).abs()
            assign = d.argmin(dim=1)
            new_levels = levels.clone()
            for i in range(n_levels):
                m = assign == i
                if m.sum() > 0: new_levels[i] = cf[m].mean()
            if (new_levels - levels).abs().max() < 1e-10 * sigma: break
            levels = new_levels
        d = (ct.unsqueeze(2) - levels.unsqueeze(0).unsqueeze(0)).abs()
        idx = d.argmin(dim=2)
        result[mask] = levels[idx]
    return result

def run_experiment(data_dir, device, ext, ghd, tcp, tcpi, qtf, cbs_scale):
    results = {}
    n_clusters = 128
    f = data_dir / f"layer10_all_gate_proj.pt"
    if not f.exists(): return results
    all_experts = torch.load(f, map_location="cpu")
    n_experts = min(5, all_experts.shape[0])
    k, n = all_experts.shape[1], all_experts.shape[2]
    print(f"  {n_experts} experts, {k}x{n}", flush=True)

    all_methods = {}

    for ei in range(n_experts):
        print(f"  Expert {ei}...", flush=True)
        w = all_experts[ei].to(device)
        w_reg, _, _ = regularize(w, device, ghd, cbs_scale)
        del w

        qk2 = quantize_trellis(w_reg, 2, device, tcp, tcpi, qtf)
        qk3 = quantize_trellis(w_reg, 3, device, tcp, tcpi, qtf)
        qk4 = quantize_trellis(w_reg, 4, device, tcp, tcpi, qtf)

        r2 = w_reg - qk2
        r3 = w_reg - qk3
        r4 = w_reg - qk4

        methods = {}

        # Base tiers
        methods["K2"] = (qk2, 2.0)
        methods["K3"] = (qk3, 3.0)
        methods["K4"] = (qk4, 4.0)

        # === TRELLIS ON RESIDUAL ===
        # K4 + K2_trellis_on_residual (6 bpw)
        for K_res in [1, 2, 3]:
            recon = quantize_trellis_with_residual(qk4, r4, K_res, device, tcp, tcpi, qtf)
            methods[f"K4+K{K_res}trellis_res"] = (recon, 4.0 + K_res)

        # K3 + K2_trellis_on_residual (5 bpw)
        for K_res in [1, 2, 3, 4]:
            recon = quantize_trellis_with_residual(qk3, r3, K_res, device, tcp, tcpi, qtf)
            methods[f"K3+K{K_res}trellis_res"] = (recon, 3.0 + K_res)

        # K2 + K2_trellis_on_residual (4 bpw)
        for K_res in [2, 3, 4]:
            recon = quantize_trellis_with_residual(qk2, r2, K_res, device, tcp, tcpi, qtf)
            methods[f"K2+K{K_res}trellis_res"] = (recon, 2.0 + K_res)

        # === LLOYD-MAX BASELINE (c128) ===
        for base_q, base_name, residual, base_bpw in [(qk4, "K4", r4, 4.0), (qk3, "K3", r3, 3.0), (qk2, "K2", r2, 2.0)]:
            lm_tiles = get_tiles(residual, k, n)
            lm_cid = cluster_by_sigma(lm_tiles, n_clusters, device)
            for nbits in [1, 2, 3, 4]:
                quant = lloyd_max_clustered(lm_tiles, lm_cid, nbits, n_clusters, device)
                recon = base_q + tiles_to_matrix(quant, k, n)
                methods[f"{base_name}+{nbits}LM_c128"] = (recon, base_bpw + nbits)
                del quant, recon
            del lm_tiles, lm_cid

        # Compute MSEs
        for name, (recon, bpw) in methods.items():
            mse = (w_reg - recon).pow(2).mean().item()
            if name not in all_methods:
                all_methods[name] = {"mses": [], "bpw": bpw}
            all_methods[name]["mses"].append(mse)
            del recon

        del w_reg, qk2, qk3, qk4, r2, r3, r4
        torch.cuda.empty_cache()

    # Print sorted by bpw, grouped
    print(f"\n  {'Method':<30} {'bpw':>5} {'avg MSE':>12}", flush=True)
    print(f"  {'-'*50}", flush=True)
    for name in sorted(all_methods.keys(), key=lambda x: (all_methods[x]["bpw"], x)):
        r = all_methods[name]
        avg = sum(r["mses"]) / len(r["mses"])
        print(f"  {name:<30} {r['bpw']:>5.0f} {avg:>12.4e}", flush=True)

    # Key comparisons: trellis-on-residual vs Lloyd-Max at same bpw
    print(f"\n  Trellis-on-residual vs Lloyd-Max at same bpw:", flush=True)
    print(f"  {'bpw':>5} {'Trellis method':<30} {'Trellis MSE':>12} {'LM method':<25} {'LM MSE':>12} {'Treallis/LM':>12}", flush=True)
    comparisons = [
        (5, "K3+K2trellis_res", "K3+2LM_c128"),
        (6, "K4+K2trellis_res", "K4+2LM_c128"),
        (6, "K3+K3trellis_res", "K3+3LM_c128"),
        (7, "K4+K3trellis_res", "K4+3LM_c128"),
        (7, "K3+K4trellis_res", "K3+4LM_c128"),
        (4, "K2+K2trellis_res", "K2+2LM_c128"),
        (5, "K2+K3trellis_res", "K2+3LM_c128"),
        (6, "K2+K4trellis_res", "K2+4LM_c128"),
    ]
    for bpw, t_name, lm_name in comparisons:
        if t_name in all_methods and lm_name in all_methods:
            t_mse = sum(all_methods[t_name]["mses"]) / len(all_methods[t_name]["mses"])
            lm_mse = sum(all_methods[lm_name]["mses"]) / len(all_methods[lm_name]["mses"])
            ratio = t_mse / lm_mse
            winner = "TRELLIS" if t_mse < lm_mse else "LM"
            print(f"  {bpw:>5} {t_name:<30} {t_mse:>12.4e} {lm_name:<25} {lm_mse:>12.4e} {ratio:>11.3f}x {winner}", flush=True)

    results["methods"] = {k: {"bpw": v["bpw"], "avg_mse": sum(v["mses"])/len(v["mses"])}
                          for k, v in all_methods.items()}
    return results

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--data-dir", default="/tmp/poc_residual/glm52_data")
    ap.add_argument("--out", default="/tmp/poc_residual/results_v34.json")
    args = ap.parse_args()
    dev = torch.device(args.device)
    print(f"Device: {dev}  GPU: {torch.cuda.get_device_name(0)}", flush=True)
    ext, ghd, tcp, tcpi, qtf, cbs_scale = _bootstrap()
    results = run_experiment(Path(args.data_dir), dev, ext, ghd, tcp, tcpi, qtf, cbs_scale)
    Path(args.out).write_text(json.dumps(results, indent=2, default=str))
    print(f"\nResults saved to {args.out}", flush=True)

if __name__ == "__main__":
    main()
