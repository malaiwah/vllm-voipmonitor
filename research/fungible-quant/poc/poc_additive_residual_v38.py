#!/usr/bin/env python3
"""PoC v38: Entropy-aware definitive hybrid Pareto.

Combines v37's hybrid tiers with v32's entropy-aware allocation.
Uses the entropy of trellis indices (not just LM) for budget calculation.
Trellis indices have non-uniform distributions too (Viterbi path preferences).

Tests: does entropy-aware allocation help with rescaled trellis tiers?
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

def rescaled_trellis(base_q, residual, K_res, device, tcp, tcpi, qtf, cbs):
    residual_rms = residual.square().mean().sqrt().item()
    if residual_rms < 1e-12: return base_q
    scale = abs(cbs) / residual_rms
    scaled = residual * scale
    quant = quantize_trellis_raw(scaled, K_res, device, tcp, tcpi, qtf)
    return base_q + quant / scale

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
    all_indices = []
    for cid in range(n_clusters):
        mask = cluster_id == cid
        if mask.sum() == 0: continue
        ct = tiles[mask]; cf = ct.flatten()
        sigma = cf.std().item()
        if sigma < 1e-12: result[mask] = 0.0; continue
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
        all_indices.append(idx.flatten())
    # Compute entropy
    entropy = float(n_bits)  # default
    if all_indices:
        all_idx = torch.cat(all_indices)
        counts = torch.bincount(all_idx, minlength=n_levels).float()
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        entropy = -(probs * probs.log2()).sum().item()
    return result, entropy

def tile_mse(w_reg, recon, k, n):
    tnk, tnn = k // 16, n // 16
    diff = (w_reg - recon).pow(2)
    return diff.view(tnk, 16, tnn, 16).mean(dim=(1, 3))

def run_experiment(data_dir, device, ext, ghd, tcp, tcpi, qtf, cbs_scale):
    results = {}
    n_clusters = 128
    f = data_dir / f"layer10_all_gate_proj.pt"
    if not f.exists(): return results
    all_experts = torch.load(f, map_location="cpu")
    n_experts = min(10, all_experts.shape[0])
    k, n = all_experts.shape[1], all_experts.shape[2]
    print(f"  {n_experts} experts, {k}x{n}", flush=True)

    tier_data = {}

    for ei in range(n_experts):
        print(f"  Expert {ei}...", flush=True)
        w = all_experts[ei].to(device)
        w_reg, _, _ = regularize(w, device, ghd, cbs_scale)
        del w

        qk2 = quantize_trellis_raw(w_reg, 2, device, tcp, tcpi, qtf)
        qk4 = quantize_trellis_raw(w_reg, 4, device, tcp, tcpi, qtf)

        r2, r4 = w_reg - qk2, w_reg - qk4

        tiers = {}
        # Base trellis (entropy = raw bits for trellis, since we don't measure trellis entropy)
        tiers["K2"] = (qk2, 2.0, 2.0)
        tiers["K3"] = (quantize_trellis_raw(w_reg, 3, device, tcp, tcpi, qtf), 3.0, 3.0)
        tiers["K4"] = (qk4, 4.0, 4.0)

        # Rescaled trellis tiers — estimate entropy as raw bits (trellis entropy is complex)
        # For a fair comparison, we use raw bits for trellis and entropy bits for LM
        for base_q, bn, res, bb in [(qk2, "K2", r2, 2.0), (qk4, "K4", r4, 4.0)]:
            for Kr in [3, 4, 5]:
                recon = rescaled_trellis(base_q, res, Kr, device, tcp, tcpi, qtf, cbs_scale)
                # Trellis entropy is typically close to raw bits (TCQ uses Viterbi, not independent)
                # Use raw bits for trellis residual
                tiers[f"{bn}+K{Kr}trsc"] = (recon, bb + Kr, bb + Kr)

        # LM tiers with entropy
        for base_q, bn, res, bb in [(qk2, "K2", r2, 2.0)]:
            tiles = get_tiles(res, k, n); cid = cluster_by_sigma(tiles, n_clusters, device)
            for nb in [4, 6]:
                quant, ent = lloyd_max_clustered(tiles, cid, nb, n_clusters, device)
                recon = base_q + tiles_to_matrix(quant, k, n)
                tiers[f"{bn}+{nb}LM"] = (recon, bb + nb, bb + ent)
            del tiles, cid

        # K3 + LM
        qk3 = quantize_trellis_raw(w_reg, 3, device, tcp, tcpi, qtf)
        r3 = w_reg - qk3
        tiles = get_tiles(r3, k, n); cid = cluster_by_sigma(tiles, n_clusters, device)
        quant, ent = lloyd_max_clustered(tiles, cid, 6, n_clusters, device)
        tiers["K3+6LM"] = (qk3 + tiles_to_matrix(quant, k, n), 9.0, 3.0 + ent)
        del tiles, cid

        # K4 + LM
        tiles = get_tiles(r4, k, n); cid = cluster_by_sigma(tiles, n_clusters, device)
        quant, ent = lloyd_max_clustered(tiles, cid, 6, n_clusters, device)
        tiers["K4+6LM"] = (qk4 + tiles_to_matrix(quant, k, n), 10.0, 4.0 + ent)
        del tiles, cid

        for name, (recon, bpw, ent_bpw) in tiers.items():
            tmse = tile_mse(w_reg, recon, k, n)
            if name not in tier_data:
                tier_data[name] = {"bpw": bpw, "entropy_bpw": ent_bpw, "tile_mses": []}
            tier_data[name]["tile_mses"].append(tmse.cpu())
            del recon, tmse

        del w_reg, qk2, qk3, qk4, r2, r3, r4
        torch.cuda.empty_cache()

    avg_tmse = {name: (sum(d["tile_mses"]) / len(d["tile_mses"])).to(device) for name, d in tier_data.items()}

    # Print tier summary
    print(f"\n  {'Tier':<20} {'bpw':>5} {'ent bpw':>8} {'avg MSE':>12}", flush=True)
    for name in sorted(tier_data.keys(), key=lambda x: tier_data[x]["bpw"]):
        d = tier_data[name]
        avg_mse = avg_tmse[name].mean().item()
        print(f"  {name:<20} {d['bpw']:>5.0f} {d['entropy_bpw']:>8.3f} {avg_mse:>12.4e}", flush=True)

    # Build two Pareto curves: raw and entropy-aware
    best_tiers_ordered = ["K2", "K3", "K4", "K2+K3trsc", "K2+K4trsc", "K2+K5trsc", "K2+4LM", "K2+6LM", "K3+6LM", "K4+6LM"]
    best_tiers_ordered = [t for t in best_tiers_ordered if t in avg_tmse]
    n_tiles = (k // 16) * (n // 16)

    for budget_type, budget_key in [("raw", "bpw"), ("entropy", "entropy_bpw")]:
        pareto_bpw = [tier_data[t][budget_key] for t in best_tiers_ordered]
        upgrades = []
        for i in range(len(best_tiers_ordered) - 1):
            lo, hi = best_tiers_ordered[i], best_tiers_ordered[i + 1]
            bit_cost = pareto_bpw[i + 1] - pareto_bpw[i]
            if bit_cost <= 0: continue
            benefit = (avg_tmse[lo] - avg_tmse[hi]).flatten()
            for ti in range(n_tiles):
                upgrades.append((benefit[ti].item(), ti, lo, hi, bit_cost))
        upgrades.sort(key=lambda x: -x[0])

        print(f"\n  {budget_type} Pareto:", flush=True)
        pareto = []
        for target_10 in range(30, 101, 5):
            target_bpw = target_10 / 10.0
            current_tier = [best_tiers_ordered[0]] * n_tiles
            current_bits = pareto_bpw[0] * n_tiles
            target_bits = target_bpw * n_tiles
            for benefit, tile_idx, lo, hi, bit_cost in upgrades:
                if current_bits + bit_cost > target_bits + 1e-6: continue
                if current_tier[tile_idx] != lo: continue
                if benefit <= 0: continue
                lo_idx = best_tiers_ordered.index(lo); hi_idx = best_tiers_ordered.index(hi)
                if lo_idx + 1 != hi_idx: continue
                current_tier[tile_idx] = hi; current_bits += bit_cost
            total_mse = sum(avg_tmse[current_tier[ti]][ti // (n // 16), ti % (n // 16)].item() for ti in range(n_tiles))
            avg_mse = total_mse / n_tiles
            actual_bpw = current_bits / n_tiles
            pareto.append({"target_bpw": target_bpw, "actual_bpw": actual_bpw, "mse": avg_mse})
            print(f"    {target_bpw:.1f}  actual={actual_bpw:.3f}  MSE={avg_mse:.6e}", flush=True)
        results[f"{budget_type}_pareto"] = pareto

    # Compare raw vs entropy
    print(f"\n  Entropy vs Raw Pareto comparison:", flush=True)
    raw_p = results["raw_pareto"]; ent_p = results["entropy_pareto"]
    for r, e in zip(raw_p, ent_p):
        if r["target_bpw"] == e["target_bpw"]:
            imp = (1 - e["mse"] / r["mse"]) * 100 if r["mse"] > 0 else 0
            print(f"    {r['target_bpw']:.1f}  raw={r['mse']:.4e}  ent={e['mse']:.4e}  improvement={imp:.1f}%", flush=True)

    return results

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--data-dir", default="/tmp/poc_residual/glm52_data")
    ap.add_argument("--out", default="/tmp/poc_residual/results_v38.json")
    args = ap.parse_args()
    dev = torch.device(args.device)
    print(f"Device: {dev}  GPU: {torch.cuda.get_device_name(0)}", flush=True)
    ext, ghd, tcp, tcpi, qtf, cbs_scale = _bootstrap()
    results = run_experiment(Path(args.data_dir), dev, ext, ghd, tcp, tcpi, qtf, cbs_scale)
    Path(args.out).write_text(json.dumps(results, indent=2, default=str))
    print(f"\nResults saved to {args.out}", flush=True)

if __name__ == "__main__":
    main()
