#!/usr/bin/env python3
"""PoC v25: Codebook clustering — share codebooks across similar tiles.

Instead of per-tile (49152 codebooks, high overhead) or global (1 codebook,
lower quality), cluster tiles by residual distribution and share codebooks
within clusters.

Clustering features: per-tile sigma (std), abs_mean, skewness, kurtosis.
Test K = 1, 4, 16, 64, 256, 1024, 49152 clusters.

Overhead = K × 2^N × 4 bytes / total_weights
  For 2-bit, 49152 tiles: K=1→0, K=16→0.00002, K=256→0.0003, K=49152→0.0625
  For 4-bit, 49152 tiles: K=1→0, K=16→0.00008, K=256→0.0013, K=49152→0.25
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

def lloyd_max_for_group(r_flat, n_bits, n_iters=15):
    """Lloyd-Max for a flat tensor of residuals."""
    n_levels = 2 ** n_bits
    sigma = r_flat.std().item()
    if sigma < 1e-12: return torch.zeros_like(r_flat)
    levels = torch.linspace(-3 * sigma, 3 * sigma, n_levels, device=r_flat.device)
    for _ in range(n_iters):
        d = (r_flat.unsqueeze(1) - levels.unsqueeze(0)).abs()
        assign = d.argmin(dim=1)
        new_levels = levels.clone()
        for i in range(n_levels):
            mask = assign == i
            if mask.sum() > 0: new_levels[i] = r_flat[mask].mean()
        if (new_levels - levels).abs().max() < 1e-10 * sigma: break
        levels = new_levels
    d = (r_flat.unsqueeze(1) - levels.unsqueeze(0)).abs()
    return levels[d.argmin(dim=1)]

def lloyd_max_clustered(r, n_bits, k, n, n_clusters, device):
    """Cluster tiles by sigma, train one codebook per cluster. Vectorized."""
    tnk, tnn = k // 16, n // 16
    n_tiles = tnk * tnn
    tiles = r.view(tnk, 16, tnn, 16).permute(0, 2, 1, 3).reshape(n_tiles, 256)

    # Feature: per-tile sigma
    sigmas = tiles.std(dim=1).clamp(min=1e-12)  # (n_tiles,)

    if n_clusters <= 1:
        cluster_id = torch.zeros(n_tiles, dtype=torch.long, device=device)
    else:
        # Sort by sigma, assign clusters by sorted position (vectorized)
        sorted_sigmas, sort_idx = sigmas.sort()
        cluster_size = n_tiles // n_clusters
        cluster_id = torch.zeros(n_tiles, dtype=torch.long, device=device)
        cluster_id[sort_idx] = torch.arange(n_tiles, device=device) // cluster_size
        cluster_id = cluster_id.clamp(max=n_clusters - 1)

    # Train one codebook per cluster, then quantize tiles in that cluster
    result_tiles = torch.empty_like(tiles)
    n_levels = 2 ** n_bits
    for cid in range(n_clusters):
        mask = cluster_id == cid
        n_in = mask.sum().item()
        if n_in == 0: continue
        cluster_tiles = tiles[mask]  # (n_in, 256)
        cluster_flat = cluster_tiles.flatten()
        sigma = cluster_flat.std().item()
        if sigma < 1e-12:
            result_tiles[mask] = 0.0
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
        d = (cluster_flat.unsqueeze(1) - levels.unsqueeze(0)).abs()
        result_tiles[mask] = levels[d.argmin(dim=1)].reshape(n_in, 256)

    # Vectorized scatter back: reshape from (n_tiles, 256) to (k, n)
    return result_tiles.reshape(tnk, tnn, 16, 16).permute(0, 2, 1, 3).reshape(k, n)


def run_experiment(data_dir, device, ext, ghd, tcp, tcpi, qtf, cbs):
    results = {}
    layer_idx = 10
    gate_file = data_dir / f"layer{layer_idx}_all_gate_proj.pt"
    if not gate_file.exists(): return results
    all_experts = torch.load(gate_file, map_location="cpu")

    n_experts = min(1, all_experts.shape[0])
    k, n = all_experts.shape[1], all_experts.shape[2]
    n_tiles = (k // 16) * (n // 16)
    total_weights = k * n
    print(f"  {n_experts} experts, {k}x{n}, {n_tiles} tiles", flush=True)

    all_methods = {}

    for ei in range(n_experts):
        print(f"  Expert {ei}...", flush=True)
        w = all_experts[ei].to(device)
        w_reg, _, _ = regularize(w, device, ghd, cbs)
        del w

        qk4 = quantize_trellis(w_reg, 4, device, tcp, tcpi, qtf)
        r4 = w_reg - qk4

        # Test different numbers of clusters for 2-bit and 4-bit LM
        for nbits in [2, 4]:
            # Global (1 cluster)
            lm = lloyd_max_clustered(r4, nbits, k, n, 1, device)
            oh = 0.0
            name = f"K4+{nbits}LM_c1"
            mse = (w_reg - (qk4 + lm)).pow(2).mean().item()
            if name not in all_methods: all_methods[name] = {"mses": [], "bpw": 4 + nbits + oh}
            all_methods[name]["mses"].append(mse)
            del lm

            # Clustered: 4, 16, 64, 256, 1024
            for nc in [4, 16, 64, 256, 1024]:
                oh = nc * (2 ** nbits) * 4 / total_weights
                t0 = time.time()
                lm = lloyd_max_clustered(r4, nbits, k, n, nc, device)
                t1 = time.time()
                name = f"K4+{nbits}LM_c{nc}"
                mse = (w_reg - (qk4 + lm)).pow(2).mean().item()
                if name not in all_methods: all_methods[name] = {"mses": [], "bpw": 4 + nbits + oh}
                all_methods[name]["mses"].append(mse)
                print(f"    {nbits}bit c{nc}: {t1-t0:.1f}s  oh={oh:.5f}  MSE={mse:.4e}", flush=True)
                del lm

            # Per-tile already measured in v23b — skip

        del w_reg, qk4, r4
        torch.cuda.empty_cache()

    # Print
    print(f"\n  {'Method':<30} {'bpw':>7} {'MSE':>12} {'vs global':>10}", flush=True)
    print(f"  {'-'*62}", flush=True)
    for nbits in [2, 4]:
        base = f"K4+{nbits}LM_c1"
        base_mse = sum(all_methods[base]["mses"]) / len(all_methods[base]["mses"])
        for name in sorted(all_methods.keys(), key=lambda x: all_methods[x]["bpw"]):
            if f"{nbits}LM" not in name: continue
            r = all_methods[name]
            avg = sum(r["mses"]) / len(r["mses"])
            ratio = avg / base_mse * 100
            print(f"  {name:<30} {r['bpw']:>7.3f} {avg:>12.4e} {ratio:>9.1f}%", flush=True)
        print()

    results["methods"] = {k: {"bpw": v["bpw"], "avg_mse": sum(v["mses"])/len(v["mses"])}
                          for k, v in all_methods.items()}
    return results

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--data-dir", default="/tmp/poc_residual/glm52_data")
    ap.add_argument("--out", default="/tmp/poc_residual/results_v25.json")
    args = ap.parse_args()
    dev = torch.device(args.device)
    print(f"Device: {dev}  GPU: {torch.cuda.get_device_name(0)}", flush=True)
    ext, ghd, tcp, tcpi, qtf, cbs = _bootstrap()
    results = run_experiment(Path(args.data_dir), dev, ext, ghd, tcp, tcpi, qtf, cbs)
    Path(args.out).write_text(json.dumps(results, indent=2, default=str))
    print(f"\nResults saved to {args.out}", flush=True)

if __name__ == "__main__":
    main()
