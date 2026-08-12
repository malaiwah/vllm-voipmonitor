#!/usr/bin/env python3
"""PoC v33: Three new ideas — tier-specific clusters, sparse LM, adaptive bit-width.

1. Tier-specific cluster counts: c512 for 6-bit LM, c128 for 2-bit.
   v30 showed different optimal K per bit-width. Test mixed-cluster Pareto.

2. Sparse residual LM: Only quantize top-K% largest |residual| values.
   Leave the rest at 0 (trellis-only). Saves bits on small residuals.
   Sparse overhead: index bitmap (1 bit/weight) + LM indices for sparse subset.

3. Per-tile adaptive LM bit-width: Within K4+LM tier, allow per-tile LM bits.
   Tiles with high residual energy get 3-bit LM, low energy get 1-bit.
   Total budget = average. Tests if per-tile LM allocation beats uniform LM.
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

def tile_mse(w_reg, recon, k, n):
    tnk, tnn = k // 16, n // 16
    diff = (w_reg - recon).pow(2)
    return diff.view(tnk, 16, tnn, 16).mean(dim=(1, 3))

def run_experiment(data_dir, device, ext, ghd, tcp, tcpi, qtf, cbs_scale):
    results = {}
    f = data_dir / f"layer10_all_gate_proj.pt"
    if not f.exists(): return results
    all_experts = torch.load(f, map_location="cpu")
    n_experts = min(5, all_experts.shape[0])
    k, n = all_experts.shape[1], all_experts.shape[2]
    print(f"  {n_experts} experts, {k}x{n}", flush=True)

    all_results = {}

    for ei in range(n_experts):
        print(f"  Expert {ei}...", flush=True)
        w = all_experts[ei].to(device)
        w_reg, _, _ = regularize(w, device, ghd, cbs_scale)
        del w
        qk4 = quantize_trellis(w_reg, 4, device, tcp, tcpi, qtf)
        r4 = w_reg - qk4
        tiles = get_tiles(r4, k, n)

        # === Part 1: Tier-specific cluster counts ===
        # Compare: c128 for all vs c512 for 4-bit/6-bit, c128 for 1-bit/2-bit
        for nbits, nc_list in [(1, [128]), (2, [128]), (3, [128]), (4, [128, 512]), (6, [128, 512])]:
            for nc in nc_list:
                cid = cluster_by_sigma(tiles, nc, device)
                quant = lloyd_max_clustered(tiles, cid, nbits, nc, device)
                recon = qk4 + tiles_to_matrix(quant, k, n)
                mse = (w_reg - recon).pow(2).mean().item()
                key = f"K4+{nbits}LM_c{nc}"
                if key not in all_results: all_results[key] = {"mses": [], "bpw": 4 + nbits}
                all_results[key]["mses"].append(mse)
                del cid, quant, recon

        # === Part 2: Sparse residual LM ===
        # Only quantize top-K% largest |residual| values with 2-bit LM
        # Rest stay at trellis-only (residual = 0)
        for sparsity in [0.1, 0.25, 0.5, 0.75, 1.0]:
            abs_r4 = r4.abs()
            threshold = torch.quantile(abs_r4.flatten(), 1.0 - sparsity)
            sparse_mask = abs_r4 >= threshold
            # Create sparse residual: only non-zero where mask is True
            sparse_r = torch.zeros_like(r4)
            sparse_r[sparse_mask] = r4[sparse_mask]
            # Quantize sparse residual with 2-bit LM (c128)
            sparse_tiles = get_tiles(sparse_r, k, n)
            cid = cluster_by_sigma(sparse_tiles, 128, device)
            quant = lloyd_max_clustered(sparse_tiles, cid, 2, 128, device)
            recon = qk4 + tiles_to_matrix(quant, k, n)
            mse = (w_reg - recon).pow(2).mean().item()
            # Effective bpw: 4 (trellis) + sparsity * 2 (LM) + 1 (sparse bitmap)
            eff_bpw = 4.0 + sparsity * 2.0 + 1.0 / 8.0  # bitmap = 1 bit/weight = 0.125 bpw
            # Actually bitmap is 1 bit per weight, so 1.0 bpw overhead
            eff_bpw_bitmap = 4.0 + sparsity * 2.0 + 1.0
            # With entropy-coded bitmap: H(bitmap) ≈ -p*log2(p) - (1-p)*log2(1-p)
            p = sparsity
            if p > 0 and p < 1:
                bitmap_ent = -p * math.log2(p) - (1-p) * math.log2(1-p)
            else:
                bitmap_ent = 0.0
            eff_bpw_entropy = 4.0 + sparsity * 2.0 + bitmap_ent
            key = f"K4+sparse{int(sparsity*100)}%_2LM"
            if key not in all_results:
                all_results[key] = {"mses": [], "bpw_raw": eff_bpw_bitmap, "bpw_entropy": eff_bpw_entropy}
            all_results[key]["mses"].append(mse)
            del abs_r4, sparse_mask, sparse_r, sparse_tiles, cid, quant, recon

        # === Part 3: Per-tile adaptive LM bit-width ===
        # Within "K4+~2LM" tier: assign 1-bit LM to low-energy tiles, 3-bit to high
        # Total budget = average ~2 bits
        tile_energies = tiles.var(dim=1)  # (n_tiles,)
        sorted_energies, sort_idx = tile_energies.sort()
        n_tiles = tiles.shape[0]

        for split_ratio in [0.3, 0.5, 0.7]:
            n_high = int(n_tiles * split_ratio)
            n_low = n_tiles - n_high
            # High-energy tiles get 3-bit LM, low-energy get 1-bit
            high_tiles_idx = sort_idx[n_low:]
            low_tiles_idx = sort_idx[:n_low]

            result_tiles = torch.empty_like(tiles)
            # Low-energy: 1-bit LM
            if n_low > 0:
                low_tiles = tiles[low_tiles_idx]
                low_cid = cluster_by_sigma(low_tiles, 128, device)
                low_quant = lloyd_max_clustered(low_tiles, low_cid, 1, 128, device)
                result_tiles[low_tiles_idx] = low_quant

            # High-energy: 3-bit LM
            if n_high > 0:
                high_tiles = tiles[high_tiles_idx]
                high_cid = cluster_by_sigma(high_tiles, 128, device)
                high_quant = lloyd_max_clustered(high_tiles, high_cid, 3, 128, device)
                result_tiles[high_tiles_idx] = high_quant

            recon = qk4 + tiles_to_matrix(result_tiles, k, n)
            mse = (w_reg - recon).pow(2).mean().item()
            # Effective bpw: 4 + n_low*1/n_tiles + n_high*3/n_tiles
            avg_lm_bits = (n_low * 1 + n_high * 3) / n_tiles
            eff_bpw = 4.0 + avg_lm_bits
            key = f"K4+adaptive_{int(split_ratio*100)}%_3bit"
            if key not in all_results:
                all_results[key] = {"mses": [], "bpw": eff_bpw}
            all_results[key]["mses"].append(mse)
            del result_tiles

        # Compare adaptive vs uniform 2-bit LM at same bpw
        # Uniform 2-bit LM = 6.0 bpw
        # Adaptive 30% high (3-bit) + 70% low (1-bit) = 4 + 0.3*3 + 0.7*1 = 5.6 bpw
        # Adaptive 50% high + 50% low = 4 + 0.5*3 + 0.5*1 = 6.0 bpw (same as uniform!)
        # Adaptive 70% high + 30% low = 4 + 0.7*3 + 0.3*1 = 6.4 bpw

        del w_reg, qk4, r4, tiles
        torch.cuda.empty_cache()

    # Print results
    print(f"\n  Part 1: Tier-specific clusters", flush=True)
    print(f"  {'Method':<25} {'bpw':>5} {'avg MSE':>12}", flush=True)
    for key in sorted(all_results.keys(), key=lambda x: all_results[x].get("bpw", all_results[x].get("bpw_raw", 0))):
        r = all_results[key]
        avg = sum(r["mses"]) / len(r["mses"])
        bpw = r.get("bpw", r.get("bpw_raw", "?"))
        print(f"  {key:<25} {bpw:>5} {avg:>12.4e}", flush=True)

    # Part 2: Sparse LM analysis
    print(f"\n  Part 2: Sparse residual LM", flush=True)
    print(f"  {'Method':<25} {'bpw_raw':>8} {'bpw_ent':>8} {'avg MSE':>12}", flush=True)
    for key in sorted([k for k in all_results if "sparse" in k], key=lambda x: all_results[x]["bpw_raw"]):
        r = all_results[key]
        avg = sum(r["mses"]) / len(r["mses"])
        print(f"  {key:<25} {r['bpw_raw']:>8.3f} {r['bpw_entropy']:>8.3f} {avg:>12.4e}", flush=True)

    # Compare: K4+2LM (uniform, 6 bpw) vs sparse 100% (same) vs sparse 50% (5+bitmap)
    uniform_key = "K4+2LM_c128"
    if uniform_key in all_results:
        uniform_mse = sum(all_results[uniform_key]["mses"]) / len(all_results[uniform_key]["mses"])
        print(f"\n  Sparse vs uniform at ~6 bpw:", flush=True)
        print(f"  Uniform K4+2LM (6.000): MSE={uniform_mse:.4e}", flush=True)
        for key in [k for k in all_results if "sparse" in k]:
            r = all_results[key]
            avg = sum(r["mses"]) / len(r["mses"])
            print(f"  {key} (raw={r['bpw_raw']:.3f}, ent={r['bpw_entropy']:.3f}): MSE={avg:.4e}", flush=True)

    # Part 3: Adaptive LM
    print(f"\n  Part 3: Per-tile adaptive LM bit-width", flush=True)
    print(f"  {'Method':<30} {'bpw':>5} {'avg MSE':>12}", flush=True)
    for key in sorted([k for k in all_results if "adaptive" in k], key=lambda x: all_results[x]["bpw"]):
        r = all_results[key]
        avg = sum(r["mses"]) / len(r["mses"])
        print(f"  {key:<30} {r['bpw']:>5.1f} {avg:>12.4e}", flush=True)

    results["methods"] = {k: {kk: vv for kk, vv in v.items() if kk != "mses"} | {"avg_mse": sum(v["mses"])/len(v["mses"])}
                          for k, v in all_results.items()}
    return results

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--data-dir", default="/tmp/poc_residual/glm52_data")
    ap.add_argument("--out", default="/tmp/poc_residual/results_v33.json")
    args = ap.parse_args()
    dev = torch.device(args.device)
    print(f"Device: {dev}  GPU: {torch.cuda.get_device_name(0)}", flush=True)
    ext, ghd, tcp, tcpi, qtf, cbs_scale = _bootstrap()
    results = run_experiment(Path(args.data_dir), dev, ext, ghd, tcp, tcpi, qtf, cbs_scale)
    Path(args.out).write_text(json.dumps(results, indent=2, default=str))
    print(f"\nResults saved to {args.out}", flush=True)

if __name__ == "__main__":
    main()
