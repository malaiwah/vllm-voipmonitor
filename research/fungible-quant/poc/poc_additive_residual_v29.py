#!/usr/bin/env python3
"""PoC v29: Entropy coding of LM indices + product quantization.

1. Entropy coding: Lloyd-Max indices have non-uniform distributions.
   For 2-bit LM on Gaussian: P(level) ≈ [0.33, 0.17, 0.17, 0.33].
   Entropy = 1.87 bits (vs 2 bits raw) → 6.5% savings.
   
2. Product quantization: Split 256-element tile into sub-vectors,
   quantize each with separate codebook. Tests if VQ captures
   intra-tile structure that scalar LM misses.
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
    """Train non-normalized codebooks and apply. Returns quantized tiles + indices."""
    n_levels = 2 ** n_bits
    result = torch.empty_like(tiles)
    all_indices = torch.empty(tiles.shape[0], tiles.shape[1], dtype=torch.long, device=device)
    for cid in range(n_clusters):
        mask = cluster_id == cid
        if mask.sum() == 0: continue
        cluster_tiles = tiles[mask]
        cluster_flat = cluster_tiles.flatten()
        sigma = cluster_flat.std().item()
        if sigma < 1e-12:
            result[mask] = 0.0
            all_indices[mask] = 0
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
        # Final assignment
        d = (cluster_tiles.unsqueeze(2) - levels.unsqueeze(0).unsqueeze(0)).abs()
        idx = d.argmin(dim=2)
        result[mask] = levels[idx]
        all_indices[mask] = idx
    return result, all_indices

def compute_entropy(indices, n_levels):
    """Compute empirical entropy of indices in bits."""
    counts = torch.bincount(indices.flatten(), minlength=n_levels).float()
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return -(probs * probs.log2()).sum().item()

def product_quantize(tiles, n_subvectors, n_bits, device):
    """Product quantization: split 256-dim tile into n_subvectors sub-vectors,
    quantize each with its own codebook."""
    n_tiles = tiles.shape[0]
    sub_dim = 256 // n_subvectors
    n_levels = 2 ** n_bits
    result = torch.empty_like(tiles)
    
    for sv in range(n_subvectors):
        start = sv * sub_dim
        end = start + sub_dim
        sub_vectors = tiles[:, start:end]  # (n_tiles, sub_dim)
        
        # Train codebook on all sub-vectors
        flat = sub_vectors.flatten()
        sigma = flat.std().item()
        if sigma < 1e-12:
            result[:, start:end] = 0.0
            continue
        levels = torch.linspace(-3 * sigma, 3 * sigma, n_levels, device=device)
        for _ in range(12):
            d = (flat.unsqueeze(1) - levels.unsqueeze(0)).abs()
            assign = d.argmin(dim=1)
            new_levels = levels.clone()
            for i in range(n_levels):
                m = assign == i
                if m.sum() > 0: new_levels[i] = flat[m].mean()
            if (new_levels - levels).abs().max() < 1e-10 * sigma: break
            levels = new_levels
        d = (sub_vectors.unsqueeze(2) - levels.unsqueeze(0).unsqueeze(0)).abs()
        idx = d.argmin(dim=2)
        result[:, start:end] = levels[idx]
    
    return result

def run_experiment(data_dir, device, ext, ghd, tcp, tcpi, qtf, cbs_scale):
    results = {}
    n_clusters = 64
    layer_idx = 10
    gate_file = data_dir / f"layer{layer_idx}_all_gate_proj.pt"
    if not gate_file.exists(): return results

    all_experts = torch.load(gate_file, map_location="cpu")
    n_experts = min(3, all_experts.shape[0])
    k, n = all_experts.shape[1], all_experts.shape[2]
    print(f"  {n_experts} experts, {k}x{n}", flush=True)

    entropy_results = {}
    pq_results = {}

    for ei in range(n_experts):
        print(f"  Expert {ei}...", flush=True)
        w = all_experts[ei].to(device)
        w_reg, _, _ = regularize(w, device, ghd, cbs_scale)
        del w

        qk4 = quantize_trellis(w_reg, 4, device, tcp, tcpi, qtf)
        r4 = w_reg - qk4
        tiles = get_tiles(r4, k, n)
        cid = cluster_by_sigma(tiles, n_clusters, device)

        # Part 1: Entropy of LM indices
        for nbits in [1, 2, 3, 4]:
            quant_tiles, indices = train_and_apply_clustered(tiles, cid, nbits, n_clusters, device)
            recon = qk4 + tiles_to_matrix(quant_tiles, k, n)
            mse = (w_reg - recon).pow(2).mean().item()
            n_levels = 2 ** nbits
            ent = compute_entropy(indices, n_levels)
            raw_bpw = 4 + nbits
            entropy_bpw = 4 + ent
            savings = (1 - ent / nbits) * 100
            
            key = f"K4+{nbits}LM"
            if key not in entropy_results:
                entropy_results[key] = {"mses": [], "entropies": [], "raw_bpw": raw_bpw, "entropy_bpw": entropy_bpw}
            entropy_results[key]["mses"].append(mse)
            entropy_results[key]["entropies"].append(ent)
            print(f"    {nbits}bit LM: MSE={mse:.4e}  H={ent:.3f} bits (raw={nbits})  savings={savings:.1f}%", flush=True)
            del quant_tiles, indices, recon

        # Part 2: Product quantization
        # Split 256-dim tile into sub-vectors, quantize each separately
        for n_sv in [2, 4, 8, 16]:
            for nbits in [2, 4]:
                pq_quant = product_quantize(tiles, n_sv, nbits, device)
                recon_pq = qk4 + tiles_to_matrix(pq_quant, k, n)
                mse_pq = (w_reg - recon_pq).pow(2).mean().item()
                # PQ overhead: n_sv codebooks × n_levels × 4 bytes / 256 weights per tile
                overhead = n_sv * (2 ** nbits) * 4 / 256
                bpw_pq = 4 + nbits + overhead
                key = f"K4+PQ{n_sv}x{nbits}bit"
                if key not in pq_results:
                    pq_results[key] = {"mses": [], "bpw": bpw_pq}
                pq_results[key]["mses"].append(mse_pq)
                del pq_quant, recon_pq

        # Compare PQ vs scalar LM at same bitrate
        print(f"\n    Product Q vs Scalar LM:", flush=True)
        for nbits in [2, 4]:
            scalar_key = f"K4+{nbits}LM"
            scalar_mse = sum(entropy_results[scalar_key]["mses"]) / len(entropy_results[scalar_key]["mses"])
            for n_sv in [2, 4, 8, 16]:
                pq_key = f"K4+PQ{n_sv}x{nbits}bit"
                if pq_key in pq_results:
                    pq_mse = sum(pq_results[pq_key]["mses"]) / len(pq_results[pq_key]["mses"])
                    ratio = pq_mse / scalar_mse if scalar_mse > 0 else float('inf')
                    print(f"    PQ{n_sv}x{nbits}bit: MSE={pq_mse:.4e} vs scalar={scalar_mse:.4e} ({ratio:.3f}x)", flush=True)

        del w_reg, qk4, r4, tiles
        torch.cuda.empty_cache()

    # Summary
    print(f"\n  Entropy coding summary:", flush=True)
    print(f"  {'Method':<15} {'raw bpw':>8} {'entropy bpw':>12} {'savings':>8} {'MSE':>12}", flush=True)
    for key in sorted(entropy_results.keys()):
        r = entropy_results[key]
        avg_mse = sum(r["mses"]) / len(r["mses"])
        avg_ent = sum(r["entropies"]) / len(r["entropies"])
        nbits = int(r["raw_bpw"]) - 4
        savings = (1 - avg_ent / nbits) * 100
        print(f"  {key:<15} {r['raw_bpw']:>8.0f} {4+avg_ent:>12.3f} {savings:>7.1f}% {avg_mse:>12.4e}", flush=True)

    results["entropy"] = {k: {"raw_bpw": v["raw_bpw"], "avg_entropy": sum(v["entropies"])/len(v["entropies"]),
                               "avg_mse": sum(v["mses"])/len(v["mses"])}
                          for k, v in entropy_results.items()}
    results["product_quant"] = {k: {"bpw": v["bpw"], "avg_mse": sum(v["mses"])/len(v["mses"])}
                                for k, v in pq_results.items()}
    return results

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--data-dir", default="/tmp/poc_residual/glm52_data")
    ap.add_argument("--out", default="/tmp/poc_residual/results_v29.json")
    args = ap.parse_args()
    dev = torch.device(args.device)
    print(f"Device: {dev}  GPU: {torch.cuda.get_device_name(0)}", flush=True)
    ext, ghd, tcp, tcpi, qtf, cbs_scale = _bootstrap()
    results = run_experiment(Path(args.data_dir), dev, ext, ghd, tcp, tcpi, qtf, cbs_scale)
    Path(args.out).write_text(json.dumps(results, indent=2, default=str))
    print(f"\nResults saved to {args.out}", flush=True)

if __name__ == "__main__":
    main()
