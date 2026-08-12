#!/usr/bin/env python3
"""PoC v30: Optimal cluster count per bit-width + stacked clustered LM.

1. Test cluster counts [1, 4, 16, 32, 64, 128, 256, 512] for each bit-width [1,2,3,4,6].
   Find the optimal K for each N.

2. Test stacking: K4+2LM_64c + 1LM_64c (7 bpw) vs K4+3LM_64c (7 bpw).
   Does stacking clustered codebooks work better than single large codebook?
   (v22 showed single large LM beats stacking with global codebooks;
    does this hold with 64-cluster codebooks?)
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

def train_and_apply_clustered(tiles, cluster_id, n_bits, n_clusters, device):
    n_levels = 2 ** n_bits
    result = torch.empty_like(tiles)
    for cid in range(n_clusters):
        mask = cluster_id == cid
        if mask.sum() == 0: continue
        cluster_tiles = tiles[mask]
        cluster_flat = cluster_tiles.flatten()
        sigma = cluster_flat.std().item()
        if sigma < 1e-12:
            result[mask] = 0.0
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
        d = (cluster_tiles.unsqueeze(2) - levels.unsqueeze(0).unsqueeze(0)).abs()
        idx = d.argmin(dim=2)
        result[mask] = levels[idx]
    return result

def run_experiment(data_dir, device, ext, ghd, tcp, tcpi, qtf, cbs_scale):
    results = {}
    layer_idx = 10
    gate_file = data_dir / f"layer{layer_idx}_all_gate_proj.pt"
    if not gate_file.exists(): return results

    all_experts = torch.load(gate_file, map_location="cpu")
    n_experts = min(2, all_experts.shape[0])
    k, n = all_experts.shape[1], all_experts.shape[2]
    print(f"  {n_experts} experts, {k}x{n}", flush=True)

    # Part 1: Optimal cluster count per bit-width
    print(f"\n  Part 1: Optimal cluster count per bit-width", flush=True)
    cluster_counts = [1, 4, 16, 32, 64, 128, 256, 512]
    bit_widths = [1, 2, 3, 4, 6]
    
    cluster_results = {}
    
    for nbits in bit_widths:
        print(f"\n  {nbits}-bit LM:", flush=True)
        for nc in cluster_counts:
            mses = []
            for ei in range(n_experts):
                w = all_experts[ei].to(device)
                w_reg, _, _ = regularize(w, device, ghd, cbs_scale)
                del w
                qk4 = quantize_trellis(w_reg, 4, device, tcp, tcpi, qtf)
                r4 = w_reg - qk4
                tiles = get_tiles(r4, k, n)
                cid = cluster_by_sigma(tiles, nc, device)
                quant = train_and_apply_clustered(tiles, cid, nbits, nc, device)
                recon = qk4 + tiles_to_matrix(quant, k, n)
                mses.append((w_reg - recon).pow(2).mean().item())
                del w_reg, qk4, r4, tiles, quant, recon
                torch.cuda.empty_cache()
            
            avg_mse = sum(mses) / len(mses)
            overhead = nc * (2 ** nbits) * 4 / (k * n)
            key = f"{nbits}bit_c{nc}"
            cluster_results[key] = {"nbits": nbits, "n_clusters": nc, "mse": avg_mse, "overhead": overhead}
            print(f"    c{nc:>4}: MSE={avg_mse:.4e}  oh={overhead:.5f}", flush=True)

    # Find optimal cluster count per bit-width
    print(f"\n  Optimal cluster count per bit-width:", flush=True)
    for nbits in bit_widths:
        best = min((v for v in cluster_results.values() if v["nbits"] == nbits), key=lambda x: x["mse"])
        print(f"    {nbits}-bit: c{best['n_clusters']} (MSE={best['mse']:.4e}, oh={best['overhead']:.5f})", flush=True)

    # Part 2: Stacking clustered codebooks
    print(f"\n  Part 2: Stacking vs single large codebook (64 clusters)", flush=True)
    nc = 64
    stack_results = {}
    
    for ei in range(n_experts):
        w = all_experts[ei].to(device)
        w_reg, _, _ = regularize(w, device, ghd, cbs_scale)
        del w
        qk4 = quantize_trellis(w_reg, 4, device, tcp, tcpi, qtf)
        r4 = w_reg - qk4
        
        # Single 3-bit LM (7 bpw)
        tiles = get_tiles(r4, k, n)
        cid = cluster_by_sigma(tiles, nc, device)
        quant3 = train_and_apply_clustered(tiles, cid, 3, nc, device)
        recon_single = qk4 + tiles_to_matrix(quant3, k, n)
        mse_single = (w_reg - recon_single).pow(2).mean().item()
        
        # Stacked: 2-bit LM + 1-bit LM on residual (7 bpw)
        quant2 = train_and_apply_clustered(tiles, cid, 2, nc, device)
        recon_after2 = qk4 + tiles_to_matrix(quant2, k, n)
        r_after2 = w_reg - recon_after2
        tiles_r2 = get_tiles(r_after2, k, n)
        cid_r2 = cluster_by_sigma(tiles_r2, nc, device)
        quant1 = train_and_apply_clustered(tiles_r2, cid_r2, 1, nc, device)
        recon_stack = recon_after2 + tiles_to_matrix(quant1, k, n)
        mse_stack = (w_reg - recon_stack).pow(2).mean().item()
        
        # Single 4-bit LM (8 bpw)
        quant4 = train_and_apply_clustered(tiles, cid, 4, nc, device)
        recon_single4 = qk4 + tiles_to_matrix(quant4, k, n)
        mse_single4 = (w_reg - recon_single4).pow(2).mean().item()
        
        # Stacked: 2+2 (8 bpw)
        quant2b = train_and_apply_clustered(tiles_r2, cid_r2, 2, nc, device)
        recon_stack22 = recon_after2 + tiles_to_matrix(quant2b, k, n)
        mse_stack22 = (w_reg - recon_stack22).pow(2).mean().item()
        
        for name, mse in [("K4+3LM_single(7bpw)", mse_single), ("K4+2LM+1LM_stack(7bpw)", mse_stack),
                          ("K4+4LM_single(8bpw)", mse_single4), ("K4+2LM+2LM_stack(8bpw)", mse_stack22)]:
            if name not in stack_results: stack_results[name] = []
            stack_results[name].append(mse)
        
        del w_reg, qk4, r4, tiles, cid, quant3, recon_single, quant2, recon_after2
        del r_after2, tiles_r2, cid_r2, quant1, recon_stack, quant4, recon_single4, quant2b, recon_stack22
        torch.cuda.empty_cache()
    
    print(f"\n  {'Method':<30} {'avg MSE':>12}", flush=True)
    for name in sorted(stack_results.keys()):
        avg = sum(stack_results[name]) / len(stack_results[name])
        print(f"  {name:<30} {avg:>12.4e}", flush=True)

    results["cluster_optimal"] = cluster_results
    results["stacking"] = {k: sum(v)/len(v) for k, v in stack_results.items()}
    return results

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--data-dir", default="/tmp/poc_residual/glm52_data")
    ap.add_argument("--out", default="/tmp/poc_residual/results_v30.json")
    args = ap.parse_args()
    dev = torch.device(args.device)
    print(f"Device: {dev}  GPU: {torch.cuda.get_device_name(0)}", flush=True)
    ext, ghd, tcp, tcpi, qtf, cbs_scale = _bootstrap()
    results = run_experiment(Path(args.data_dir), dev, ext, ghd, tcp, tcpi, qtf, cbs_scale)
    Path(args.out).write_text(json.dumps(results, indent=2, default=str))
    print(f"\nResults saved to {args.out}", flush=True)

if __name__ == "__main__":
    main()
