#!/usr/bin/env python3
"""PoC v28: Re-verify cross-layer codebook sharing + Haar wavelet test.

1. Re-verify v26's "universal codebook" claim with CORRECT reshape.
   v26 had a reshape bug that made all codebooks give identical (garbage) results.
   
2. Test Haar wavelet decomposition (HBLLM-inspired) as alternative to Hadamard.
   Haar gives multi-resolution decomposition: low-freq (average) + high-freq (detail).
   Could enable true successive refinement.
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

def train_clustered_codebooks(tiles, cluster_id, n_bits, n_clusters, device):
    """Train non-normalized codebooks. Returns list of level tensors."""
    n_levels = 2 ** n_bits
    codebooks = []
    for cid in range(n_clusters):
        mask = cluster_id == cid
        if mask.sum() == 0:
            codebooks.append(torch.zeros(n_levels, device=device))
            continue
        cluster_flat = tiles[mask].flatten()
        sigma = cluster_flat.std().item()
        if sigma < 1e-12:
            codebooks.append(torch.zeros(n_levels, device=device))
            continue
        levels = torch.linspace(-3 * sigma, 3 * sigma, n_levels, device=device)
        for _ in range(12):
            d = (cluster_flat.unsqueeze(1) - levels.unsqueeze(0)).abs()
            assign = d.argmin(dim=1)
            new_levels = levels.clone()
            for i in range(n_levels):
                m = assign == i
                if m.sum() > 0: new_levels[i] = cluster_flat[m].mean()
            if (new_levels - levels).abs().max() < 1e-10 * sigma: break
            levels = new_levels
        codebooks.append(levels)
    return codebooks

def apply_clustered_codebooks(tiles, cluster_id, codebooks, n_clusters):
    """Apply codebooks to tiles. Returns quantized tiles (n_tiles, 256)."""
    result = torch.empty_like(tiles)
    for cid in range(n_clusters):
        mask = cluster_id == cid
        if mask.sum() == 0: continue
        cluster_tiles = tiles[mask]
        levels = codebooks[cid]
        d = (cluster_tiles.unsqueeze(2) - levels.unsqueeze(0).unsqueeze(0)).abs()
        idx = d.argmin(dim=2)
        result[mask] = levels[idx]
    return result

def run_experiment(data_dir, device, ext, ghd, tcp, tcpi, qtf, cbs_scale):
    results = {}
    n_clusters = 64
    n_bits = 2
    layers = [10, 40]
    projections = ["gate", "up", "down"]

    all_data = {}
    for layer in layers:
        for proj in projections:
            f = data_dir / f"layer{layer}_all_{proj}_proj.pt"
            if f.exists():
                all_data[(layer, proj)] = torch.load(f, map_location="cpu")

    if not all_data: return results

    # Part 1: Re-verify cross-layer codebook sharing with CORRECT reshape
    print(f"\n  Part 1: Cross-layer codebook sharing (CORRECT reshape)", flush=True)
    
    # Train codebooks on first expert of each (layer, proj)
    trained = {}
    for (layer, proj), experts in sorted(all_data.items()):
        w = experts[0].to(device)
        w_reg, _, _ = regularize(w, device, ghd, cbs_scale)
        ek, en = w_reg.shape
        del w
        qk4 = quantize_trellis(w_reg, 4, device, tcp, tcpi, qtf)
        r4 = w_reg - qk4
        tiles = get_tiles(r4, ek, en)
        cid = cluster_by_sigma(tiles, n_clusters, device)
        cbs_list = train_clustered_codebooks(tiles, cid, n_bits, n_clusters, device)
        trained[(layer, proj)] = cbs_list
        del w_reg, qk4, r4, tiles
        torch.cuda.empty_cache()

    # Test own vs cross
    print(f"  {'Train':<18} {'Apply to':<18} {'MSE':>12} {'vs own':>8}", flush=True)
    print(f"  {'-'*58}", flush=True)
    own_mses = {}
    cross_results = []

    for (layer_a, proj_a), experts in sorted(all_data.items()):
        w = experts[0].to(device)
        w_reg, _, _ = regularize(w, device, ghd, cbs_scale)
        ek, en = w_reg.shape
        del w
        qk4 = quantize_trellis(w_reg, 4, device, tcp, tcpi, qtf)
        r4 = w_reg - qk4
        tiles = get_tiles(r4, ek, en)
        cid_a = cluster_by_sigma(tiles, n_clusters, device)

        # Own codebooks
        quant_own = apply_clustered_codebooks(tiles, cid_a, trained[(layer_a, proj_a)], n_clusters)
        recon_own = qk4 + tiles_to_matrix(quant_own, ek, en)
        mse_own = (w_reg - recon_own).pow(2).mean().item()
        own_mses[(layer_a, proj_a)] = mse_own

        for (layer_t, proj_t), cbs_t in sorted(trained.items()):
            quant_cross = apply_clustered_codebooks(tiles, cid_a, cbs_t, n_clusters)
            recon_cross = qk4 + tiles_to_matrix(quant_cross, ek, en)
            mse_cross = (w_reg - recon_cross).pow(2).mean().item()
            ratio = mse_cross / mse_own if mse_own > 0 else float('inf')
            label = "self" if layer_t == layer_a and proj_t == proj_a else f"{ratio:.3f}x"
            print(f"  L{layer_t}_{proj_t:<8}     L{layer_a}_{proj_a:<8}     {mse_cross:>12.4e} {label:>8}", flush=True)
            cross_results.append({"train": f"L{layer_t}_{proj_t}", "apply": f"L{layer_a}_{proj_a}", "mse": mse_cross, "ratio": ratio})

        del w_reg, qk4, r4, tiles
        torch.cuda.empty_cache()

    results["cross_layer"] = cross_results
    results["own_mses"] = {f"L{k[0]}_{k[1]}": v for k, v in own_mses.items()}
    return results

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--data-dir", default="/tmp/poc_residual/glm52_data")
    ap.add_argument("--out", default="/tmp/poc_residual/results_v28.json")
    args = ap.parse_args()
    dev = torch.device(args.device)
    print(f"Device: {dev}  GPU: {torch.cuda.get_device_name(0)}", flush=True)
    ext, ghd, tcp, tcpi, qtf, cbs_scale = _bootstrap()
    results = run_experiment(Path(args.data_dir), dev, ext, ghd, tcp, tcpi, qtf, cbs_scale)
    Path(args.out).write_text(json.dumps(results, indent=2, default=str))
    print(f"\nResults saved to {args.out}", flush=True)

if __name__ == "__main__":
    main()
