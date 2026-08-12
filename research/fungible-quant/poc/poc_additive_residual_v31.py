#!/usr/bin/env python3
"""PoC v31: Definitive Pareto frontier with c128 + entropy coding.

Final Pareto: 8-tier system, c128 codebooks, 2-10 bpw in 0.1-bit steps.
Tests on gate_proj AND down_proj, 10 experts.
Includes entropy-coded bpw alongside raw bpw.
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

def train_and_apply_clustered_with_entropy(tiles, cluster_id, n_bits, n_clusters, device):
    """Train, apply, and compute entropy of indices."""
    n_levels = 2 ** n_bits
    result = torch.empty_like(tiles)
    all_indices = []
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
        all_indices.append(idx.flatten())
    
    # Compute entropy
    if all_indices:
        all_idx = torch.cat(all_indices)
        counts = torch.bincount(all_idx, minlength=n_levels).float()
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        entropy = -(probs * probs.log2()).sum().item()
    else:
        entropy = float(n_bits)
    
    return result, entropy

def tile_mse(w_reg, recon, k, n):
    tnk, tnn = k // 16, n // 16
    diff = (w_reg - recon).pow(2)
    return diff.view(tnk, 16, tnn, 16).mean(dim=(1, 3))  # (tnk, tnn)

def run_experiment(data_dir, device, ext, ghd, tcp, tcpi, qtf, cbs_scale):
    results = {}
    n_clusters = 128
    projections = ["gate", "down"]
    n_experts = 10

    for proj in projections:
        f = data_dir / f"layer10_all_{proj}_proj.pt"
        if not f.exists(): continue
        all_experts = torch.load(f, map_location="cpu")
        ne = min(n_experts, all_experts.shape[0])
        k, n = all_experts.shape[1], all_experts.shape[2]
        print(f"\n  {proj}_proj: {ne} experts, {k}x{n}", flush=True)

        # Collect tile MSEs for all tiers across experts
        tier_data = {}  # name -> {"bpw": float, "entropy_bpw": float, "tile_mses": []}

        for ei in range(ne):
            w = all_experts[ei].to(device)
            w_reg, _, _ = regularize(w, device, ghd, cbs_scale)
            del w

            qk2 = quantize_trellis(w_reg, 2, device, tcp, tcpi, qtf)
            qk3 = quantize_trellis(w_reg, 3, device, tcp, tcpi, qtf)
            qk4 = quantize_trellis(w_reg, 4, device, tcp, tcpi, qtf)

            tiers = {}
            tiers["K2"] = (qk2, 2.0, 2.0)
            tiers["K3"] = (qk3, 3.0, 3.0)
            tiers["K4"] = (qk4, 4.0, 4.0)

            # K4 + LM (c128)
            r4 = w_reg - qk4
            tiles_r4 = get_tiles(r4, k, n)
            cid_r4 = cluster_by_sigma(tiles_r4, n_clusters, device)
            for nbits in [1, 2, 3, 4, 6]:
                quant, ent = train_and_apply_clustered_with_entropy(tiles_r4, cid_r4, nbits, n_clusters, device)
                recon = qk4 + tiles_to_matrix(quant, k, n)
                tiers[f"K4+{nbits}LM"] = (recon, 4.0 + nbits, 4.0 + ent)

            # K2 + LM (c128) — for crossover test
            r2 = w_reg - qk2
            tiles_r2 = get_tiles(r2, k, n)
            cid_r2 = cluster_by_sigma(tiles_r2, n_clusters, device)
            for nbits in [4, 6]:
                quant, ent = train_and_apply_clustered_with_entropy(tiles_r2, cid_r2, nbits, n_clusters, device)
                recon = qk2 + tiles_to_matrix(quant, k, n)
                tiers[f"K2+{nbits}LM"] = (recon, 2.0 + nbits, 2.0 + ent)

            # K3 + LM (c128)
            r3 = w_reg - qk3
            tiles_r3 = get_tiles(r3, k, n)
            cid_r3 = cluster_by_sigma(tiles_r3, n_clusters, device)
            for nbits in [4, 6]:
                quant, ent = train_and_apply_clustered_with_entropy(tiles_r3, cid_r3, nbits, n_clusters, device)
                recon = qk3 + tiles_to_matrix(quant, k, n)
                tiers[f"K3+{nbits}LM"] = (recon, 3.0 + nbits, 3.0 + ent)

            # Compute tile MSEs
            for name, (recon, bpw, ent_bpw) in tiers.items():
                tmse = tile_mse(w_reg, recon, k, n)
                if name not in tier_data:
                    tier_data[name] = {"bpw": bpw, "entropy_bpw": ent_bpw, "tile_mses": []}
                tier_data[name]["tile_mses"].append(tmse.cpu())
                del recon, tmse

            del w_reg, qk2, qk3, qk4, r2, r3, r4
            del tiles_r4, cid_r4, tiles_r2, cid_r2, tiles_r3, cid_r3
            torch.cuda.empty_cache()

        # Average tile MSEs
        avg_tmse = {}
        for name, d in tier_data.items():
            avg = sum(d["tile_mses"]) / len(d["tile_mses"])
            avg_tmse[name] = avg.to(device)

        # Print tier summary
        print(f"\n  {'Tier':<15} {'bpw':>5} {'ent bpw':>8} {'avg MSE':>12}", flush=True)
        for name in sorted(avg_tmse.keys(), key=lambda x: tier_data[x]["bpw"]):
            d = tier_data[name]
            avg_mse = avg_tmse[name].mean().item()
            print(f"  {name:<15} {d['bpw']:>5.0f} {d['entropy_bpw']:>8.3f} {avg_mse:>12.4e}", flush=True)

        # Build Pareto using best tiers
        # Select tiers: K2(2), K3(3), K4(4), K4+1LM(5), K4+2LM(6), K4+3LM(7), K2+6LM(8), K4+6LM(10)
        pareto_tiers = ["K2", "K3", "K4", "K4+1LM", "K4+2LM", "K4+3LM", "K2+6LM", "K4+6LM"]
        pareto_tiers = [t for t in pareto_tiers if t in avg_tmse]
        pareto_bpw = [tier_data[t]["bpw"] for t in pareto_tiers]
        pareto_ent_bpw = [tier_data[t]["entropy_bpw"] for t in pareto_tiers]
        n_tiles = (k // 16) * (n // 16)

        # Build upgrades
        upgrades = []
        for i in range(len(pareto_tiers) - 1):
            lo, hi = pareto_tiers[i], pareto_tiers[i + 1]
            bit_cost = pareto_bpw[i + 1] - pareto_bpw[i]
            benefit = (avg_tmse[lo] - avg_tmse[hi]).flatten()
            for ti in range(n_tiles):
                upgrades.append((benefit[ti].item(), ti, lo, hi, bit_cost))
        upgrades.sort(key=lambda x: -x[0])

        # Build Pareto in 0.1-bit steps
        print(f"\n  Pareto frontier ({proj}_proj, c128, {ne} experts):", flush=True)
        pareto = []
        for target_10 in range(20, 101):  # 2.0 to 10.0
            target_bpw = target_10 / 10.0
            current_tier = [pareto_tiers[0]] * n_tiles
            current_bits = pareto_bpw[0] * n_tiles
            target_bits = target_bpw * n_tiles

            for benefit, tile_idx, lo, hi, bit_cost in upgrades:
                if current_bits + bit_cost > target_bits + 1e-6: continue
                if current_tier[tile_idx] != lo: continue
                if benefit <= 0: continue
                lo_idx = pareto_tiers.index(lo)
                hi_idx = pareto_tiers.index(hi)
                if lo_idx + 1 != hi_idx: continue
                current_tier[tile_idx] = hi
                current_bits += bit_cost

            total_mse = 0.0
            for ti in range(n_tiles):
                t = current_tier[ti]
                tik = ti // (n // 16)
                tin = ti % (n // 16)
                total_mse += avg_tmse[t][tik, tin].item()
            avg_mse = total_mse / n_tiles
            actual_bpw = current_bits / n_tiles
            pareto.append({"target_bpw": target_bpw, "actual_bpw": actual_bpw, "mse": avg_mse})
            if target_10 % 5 == 0:
                print(f"    {target_bpw:.1f}  actual={actual_bpw:.3f}  MSE={avg_mse:.6e}", flush=True)

        results[f"{proj}_pareto"] = pareto
        results[f"{proj}_tiers"] = {name: {"bpw": d["bpw"], "entropy_bpw": d["entropy_bpw"],
                                            "avg_mse": avg_tmse[name].mean().item()}
                                     for name, d in tier_data.items() if name in avg_tmse}

    return results

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--data-dir", default="/tmp/poc_residual/glm52_data")
    ap.add_argument("--out", default="/tmp/poc_residual/results_v31.json")
    args = ap.parse_args()
    dev = torch.device(args.device)
    print(f"Device: {dev}  GPU: {torch.cuda.get_device_name(0)}", flush=True)
    ext, ghd, tcp, tcpi, qtf, cbs_scale = _bootstrap()
    results = run_experiment(Path(args.data_dir), dev, ext, ghd, tcp, tcpi, qtf, cbs_scale)
    Path(args.out).write_text(json.dumps(results, indent=2, default=str))
    print(f"\nResults saved to {args.out}", flush=True)

if __name__ == "__main__":
    main()
