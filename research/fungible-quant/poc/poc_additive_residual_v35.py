#!/usr/bin/env python3
"""PoC v35: Rescaled trellis-on-residual.

v34 showed trellis-on-residual is 2-34× worse because the codebook
is designed for weight σ, not residual σ. 

Fix: rescale residual to match trellis codebook's expected input range.
The regularization normalizes weights to codebook_scale (cbs).
Do the same for the residual: scale to cbs, quantize, scale back.

Also test: per-tile rescaling (each tile's residual scaled independently).
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

def quantize_trellis_raw(data, K, device, tcp, tcpi, qtf):
    """Quantize arbitrary data with EXL3 trellis (no regularization)."""
    k, n = data.shape; tiles_n = n // 16; weight_q = torch.zeros_like(data)
    qa = {"K": K, "mcg": True}; perm = tcp(device); perm_i = tcpi(device)
    for bi in range(0, k, 16):
        rows = data[bi:bi+16]
        tiles = rows.reshape(16, tiles_n, 16).permute(1, 0, 2).reshape(tiles_n, 256)
        tiles = tiles[:, perm].contiguous()
        quant_w, _ = qtf(tiles, qa)
        quant_w = quant_w[:, perm_i].reshape(tiles_n, 16, 16).permute(1, 0, 2).reshape(16, n)
        weight_q[bi:bi+16] = quant_w
    return weight_q

def quantize_trellis(w_reg, K, device, tcp, tcpi, qtf):
    return quantize_trellis_raw(w_reg, K, device, tcp, tcpi, qtf)

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

def rescaled_trellis_on_residual(base_q, residual, K_res, device, tcp, tcpi, qtf, cbs):
    """Rescale residual to match trellis codebook, quantize, scale back.
    
    The regularization normalizes weights so RMS ≈ |cbs|.
    We do the same for the residual: scale so RMS = |cbs|, quantize, scale back.
    """
    # Global rescaling
    residual_rms = residual.square().mean().sqrt().item()
    if residual_rms < 1e-12:
        return base_q
    target_rms = abs(cbs)
    scale = target_rms / residual_rms
    scaled_residual = residual * scale
    quant_scaled = quantize_trellis_raw(scaled_residual, K_res, device, tcp, tcpi, qtf)
    quant_residual = quant_scaled / scale
    return base_q + quant_residual

def per_tile_rescaled_trellis(base_q, residual, K_res, k, n, device, tcp, tcpi, qtf, cbs):
    """Per-tile rescaling: each 16×16 tile scaled independently."""
    tnk, tnn = k // 16, n // 16
    result = torch.zeros_like(residual)
    target_rms = abs(cbs)
    
    for tik in range(tnk):
        for tin in range(tnn):
            rs, re = tik * 16, (tik + 1) * 16
            cs, ce = tin * 16, (tin + 1) * 16
            tile = residual[rs:re, cs:ce]
            tile_rms = tile.square().mean().sqrt().item()
            if tile_rms < 1e-12:
                continue
            scale = target_rms / tile_rms
            scaled = tile * scale
            # Quantize this single tile with trellis
            # Trellis needs 16×n format, but we have 16×16
            # Need to handle this carefully
            tile_16x16 = scaled.reshape(1, 256)
            # Actually trellis expects 16×n, so we need at least 16 rows
            # Just quantize the 16×16 block directly
            qa = {"K": K_res, "mcg": True}
            perm = tcp(device); perm_i = tcpi(device)
            # Reshape to (1, 256) for a single tile
            t = tile_16x16[:, perm].contiguous()
            quant_t, _ = qtf(t, qa)
            quant_t = quant_t[:, perm_i].reshape(16, 16)
            result[rs:re, cs:ce] = quant_t / scale
    
    return base_q + result

def run_experiment(data_dir, device, ext, ghd, tcp, tcpi, qtf, cbs_scale):
    results = {}
    n_clusters = 128
    f = data_dir / f"layer10_all_gate_proj.pt"
    if not f.exists(): return results
    all_experts = torch.load(f, map_location="cpu")
    n_experts = min(5, all_experts.shape[0])
    k, n = all_experts.shape[1], all_experts.shape[2]
    print(f"  {n_experts} experts, {k}x{n}", flush=True)
    print(f"  codebook_scale = {cbs_scale}", flush=True)

    all_methods = {}

    for ei in range(n_experts):
        print(f"  Expert {ei}...", flush=True)
        w = all_experts[ei].to(device)
        w_reg, _, _ = regularize(w, device, ghd, cbs_scale)
        del w

        qk4 = quantize_trellis(w_reg, 4, device, tcp, tcpi, qtf)
        r4 = w_reg - qk4

        # Residual statistics
        if ei == 0:
            print(f"    Weight RMS: {w_reg.square().mean().sqrt().item():.4f}", flush=True)
            print(f"    Residual RMS: {r4.square().mean().sqrt().item():.4f}", flush=True)
            print(f"    Residual/weight ratio: {r4.square().mean().sqrt().item() / w_reg.square().mean().sqrt().item():.4f}", flush=True)

        methods = {}

        # Baselines
        methods["K4"] = (qk4, 4.0)

        # Lloyd-Max baseline (c128)
        tiles = get_tiles(r4, k, n)
        cid = cluster_by_sigma(tiles, n_clusters, device)
        for nbits in [2, 3, 4]:
            quant = lloyd_max_clustered(tiles, cid, nbits, n_clusters, device)
            recon = qk4 + tiles_to_matrix(quant, k, n)
            methods[f"K4+{nbits}LM_c128"] = (recon, 4.0 + nbits)
            del quant

        # Rescaled trellis on residual (global rescaling)
        for K_res in [2, 3, 4]:
            recon = rescaled_trellis_on_residual(qk4, r4, K_res, device, tcp, tcpi, qtf, cbs_scale)
            methods[f"K4+K{K_res}trellis_rescaled"] = (recon, 4.0 + K_res)

        # Per-tile rescaled trellis on residual
        # Only test K=2 for speed (per-tile is slow)
        for K_res in [2]:
            recon = per_tile_rescaled_trellis(qk4, r4, K_res, k, n, device, tcp, tcpi, qtf, cbs_scale)
            methods[f"K4+K{K_res}trellis_pertile_rescaled"] = (recon, 4.0 + K_res)

        # Unscaled trellis (from v34, for comparison)
        for K_res in [2]:
            recon_raw = quantize_trellis_raw(r4, K_res, device, tcp, tcpi, qtf)
            methods[f"K4+K{K_res}trellis_unscaled"] = (qk4 + recon_raw, 4.0 + K_res)

        # Compute MSEs
        for name, (recon, bpw) in methods.items():
            mse = (w_reg - recon).pow(2).mean().item()
            if name not in all_methods:
                all_methods[name] = {"mses": [], "bpw": bpw}
            all_methods[name]["mses"].append(mse)
            del recon

        del w_reg, qk4, r4, tiles, cid
        torch.cuda.empty_cache()

    # Print
    print(f"\n  {'Method':<40} {'bpw':>5} {'avg MSE':>12}", flush=True)
    print(f"  {'-'*60}", flush=True)
    for name in sorted(all_methods.keys(), key=lambda x: (all_methods[x]["bpw"], x)):
        r = all_methods[name]
        avg = sum(r["mses"]) / len(r["mses"])
        print(f"  {name:<40} {r['bpw']:>5.0f} {avg:>12.4e}", flush=True)

    # Key comparison
    print(f"\n  Key comparison at 6 bpw:", flush=True)
    for name in sorted(all_methods.keys(), key=lambda x: sum(all_methods[x]["mses"])/len(all_methods[x]["mses"])):
        r = all_methods[name]
        if abs(r["bpw"] - 6.0) < 0.01:
            avg = sum(r["mses"]) / len(r["mses"])
            print(f"    {name:<40} MSE={avg:.4e}", flush=True)

    print(f"\n  Key comparison at 7 bpw:", flush=True)
    for name in sorted(all_methods.keys(), key=lambda x: sum(all_methods[x]["mses"])/len(all_methods[x]["mses"])):
        r = all_methods[name]
        if abs(r["bpw"] - 7.0) < 0.01:
            avg = sum(r["mses"]) / len(r["mses"])
            print(f"    {name:<40} MSE={avg:.4e}", flush=True)

    results["methods"] = {k: {"bpw": v["bpw"], "avg_mse": sum(v["mses"])/len(v["mses"])}
                          for k, v in all_methods.items()}
    return results

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--data-dir", default="/tmp/poc_residual/glm52_data")
    ap.add_argument("--out", default="/tmp/poc_residual/results_v35.json")
    args = ap.parse_args()
    dev = torch.device(args.device)
    print(f"Device: {dev}  GPU: {torch.cuda.get_device_name(0)}", flush=True)
    ext, ghd, tcp, tcpi, qtf, cbs_scale = _bootstrap()
    results = run_experiment(Path(args.data_dir), dev, ext, ghd, tcp, tcpi, qtf, cbs_scale)
    Path(args.out).write_text(json.dumps(results, indent=2, default=str))
    print(f"\nResults saved to {args.out}", flush=True)

if __name__ == "__main__":
    main()
