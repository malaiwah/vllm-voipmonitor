#!/usr/bin/env python3
"""PoC v32: Entropy-aware Pareto + BPDQ-style bit-plane test.

1. Entropy-aware Pareto: use entropy_bpw instead of raw_bpw for tier budget.
   At 6.0 entropy-bpw, can afford K4+2LM (5.899) + some upgrades.

2. BPDQ-style bit-plane: decompose LM residual into sign + magnitude planes.
   Test if variable grid beats Lloyd-Max.
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
    all_indices = []
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
        all_indices.append(idx.flatten())
    if all_indices:
        all_idx = torch.cat(all_indices)
        counts = torch.bincount(all_idx, minlength=n_levels).float()
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        entropy = -(probs * probs.log2()).sum().item()
    else:
        entropy = float(n_bits)
    return result, entropy

def bitplane_quantize(r, n_bits, device):
    """BPDQ-style bit-plane decomposition.
    
    For 2-bit: sign plane (1 bit) + magnitude plane (1 bit)
    For 4-bit: sign (1) + 3 magnitude levels
    Variable grid: sign × magnitude coefficient
    """
    if n_bits == 2:
        # Sign + 1-bit magnitude
        sign = torch.sign(r)
        mag = r.abs()
        # 1-bit magnitude: 2 levels (small, large)
        median_mag = mag.median()
        mag_q = torch.where(mag > median_mag, mag[mag > median_mag].mean() if (mag > median_mag).sum() > 0 else median_mag,
                           torch.tensor(0.0, device=device))
        # Actually, let's do it properly: 2 magnitude levels
        mag_flat = mag.flatten()
        if mag_flat.numel() == 0 or mag_flat.std() < 1e-12:
            return torch.zeros_like(r), n_bits
        # Split at median
        threshold = mag_flat.median().item()
        high_mask = mag > threshold
        high_mean = mag[high_mask].mean().item() if high_mask.sum() > 0 else threshold
        low_mean = mag[~high_mask].mean().item() if (~high_mask).sum() > 0 else 0
        result = torch.zeros_like(r)
        result[high_mask] = sign[high_mask] * high_mean
        result[~high_mask] = sign[~high_mask] * low_mean
        return result, n_bits
    elif n_bits == 4:
        # Sign (1) + 3-bit magnitude (8 levels)
        sign = torch.sign(r)
        mag = r.abs()
        sigma = mag.std().item()
        if sigma < 1e-12:
            return torch.zeros_like(r), n_bits
        # 8 levels for magnitude via Lloyd-Max
        n_mag_levels = 8
        levels = torch.linspace(0, 3 * sigma, n_mag_levels, device=device)
        for _ in range(12):
            d = (mag.flatten().unsqueeze(1) - levels.unsqueeze(0)).abs()
            assign = d.argmin(dim=1)
            new_levels = levels.clone()
            for i in range(n_mag_levels):
                m = assign == i
                if m.sum() > 0: new_levels[i] = mag.flatten()[m].mean()
            if (new_levels - levels).abs().max() < 1e-10 * sigma: break
            levels = new_levels
        d = (mag.unsqueeze(2) - levels.unsqueeze(0).unsqueeze(0)).abs()
        idx = d.argmin(dim=2)
        result = sign * levels[idx].reshape(r.shape)
        return result, n_bits
    else:
        # Fallback to standard LM
        return torch.zeros_like(r), n_bits

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
    n_experts = min(5, all_experts.shape[0])
    k, n = all_experts.shape[1], all_experts.shape[2]
    print(f"  {n_experts} experts, {k}x{n}", flush=True)

    # Part 1: BPDQ-style bit-plane vs Lloyd-Max
    print(f"\n  Part 1: Bit-plane (BPDQ) vs Lloyd-Max", flush=True)
    bp_results = {}
    lm_results = {}

    for ei in range(n_experts):
        w = all_experts[ei].to(device)
        w_reg, _, _ = regularize(w, device, ghd, cbs_scale)
        del w
        qk4 = quantize_trellis(w_reg, 4, device, tcp, tcpi, qtf)
        r4 = w_reg - qk4
        tiles = get_tiles(r4, k, n)
        cid = cluster_by_sigma(tiles, n_clusters, device)

        for nbits in [2, 4]:
            # Lloyd-Max
            lm_quant, lm_ent = lloyd_max_clustered(tiles, cid, nbits, n_clusters, device)
            lm_recon = qk4 + tiles_to_matrix(lm_quant, k, n)
            lm_mse = (w_reg - lm_recon).pow(2).mean().item()
            key = f"LM_{nbits}bit"
            if key not in lm_results: lm_results[key] = []
            lm_results[key].append(lm_mse)

            # Bit-plane
            bp_quant = torch.empty_like(tiles)
            for cid_i in range(n_clusters):
                mask = cid == cid_i
                if mask.sum() == 0: continue
                bp_result, _ = bitplane_quantize(tiles[mask], nbits, device)
                bp_quant[mask] = bp_result
            bp_recon = qk4 + tiles_to_matrix(bp_quant, k, n)
            bp_mse = (w_reg - bp_recon).pow(2).mean().item()
            key = f"BP_{nbits}bit"
            if key not in bp_results: bp_results[key] = []
            bp_results[key].append(bp_mse)

        del w_reg, qk4, r4, tiles
        torch.cuda.empty_cache()

    print(f"\n  {'Method':<20} {'avg MSE':>12} {'vs LM':>8}", flush=True)
    for nbits in [2, 4]:
        lm_avg = sum(lm_results[f"LM_{nbits}bit"]) / len(lm_results[f"LM_{nbits}bit"])
        bp_avg = sum(bp_results[f"BP_{nbits}bit"]) / len(bp_results[f"BP_{nbits}bit"])
        ratio = bp_avg / lm_avg
        print(f"  LM_{nbits}bit          {lm_avg:>12.4e}  baseline", flush=True)
        print(f"  BP_{nbits}bit          {bp_avg:>12.4e}  {ratio:.3f}x", flush=True)

    # Part 2: Entropy-aware Pareto
    print(f"\n  Part 2: Entropy-aware Pareto", flush=True)
    tier_data = {}
    for ei in range(n_experts):
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

        for base_q, base_name, residual, base_bpw in [(qk4, "K4", w_reg - qk4, 4.0), (qk2, "K2", w_reg - qk2, 2.0)]:
            tiles = get_tiles(residual, k, n)
            cid = cluster_by_sigma(tiles, n_clusters, device)
            for nbits in [1, 2, 3, 4, 6]:
                quant, ent = lloyd_max_clustered(tiles, cid, nbits, n_clusters, device)
                recon = base_q + tiles_to_matrix(quant, k, n)
                tiers[f"{base_name}+{nbits}LM"] = (recon, base_bpw + nbits, base_bpw + ent)

        for name, (recon, bpw, ent_bpw) in tiers.items():
            tmse = tile_mse(w_reg, recon, k, n)
            if name not in tier_data:
                tier_data[name] = {"bpw": bpw, "entropy_bpw": ent_bpw, "tile_mses": []}
            tier_data[name]["tile_mses"].append(tmse.cpu())
            del recon, tmse

        del w_reg, qk2, qk3, qk4
        torch.cuda.empty_cache()

    avg_tmse = {name: (sum(d["tile_mses"]) / len(d["tile_mses"])).to(device) for name, d in tier_data.items()}
    n_tiles = (k // 16) * (n // 16)

    # Build two Pareto curves: raw_bpw and entropy_bpw
    pareto_tiers = ["K2", "K3", "K4", "K4+1LM", "K4+2LM", "K4+3LM", "K2+6LM", "K4+6LM"]
    pareto_tiers = [t for t in pareto_tiers if t in avg_tmse]

    for budget_type, budget_key in [("raw", "bpw"), ("entropy", "entropy_bpw")]:
        pareto_bpw = [tier_data[t][budget_key] for t in pareto_tiers]
        upgrades = []
        for i in range(len(pareto_tiers) - 1):
            lo, hi = pareto_tiers[i], pareto_tiers[i + 1]
            bit_cost = pareto_bpw[i + 1] - pareto_bpw[i]
            benefit = (avg_tmse[lo] - avg_tmse[hi]).flatten()
            for ti in range(n_tiles):
                upgrades.append((benefit[ti].item(), ti, lo, hi, bit_cost))
        upgrades.sort(key=lambda x: -x[0])

        print(f"\n  {budget_type} Pareto:", flush=True)
        pareto = []
        for target_10 in range(30, 101, 5):  # 3.0 to 10.0 in 0.5 steps
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
            total_mse = sum(avg_tmse[current_tier[ti]][ti // (n // 16), ti % (n // 16)].item() for ti in range(n_tiles))
            avg_mse = total_mse / n_tiles
            actual_bpw = current_bits / n_tiles
            pareto.append({"target": target_bpw, "actual": actual_bpw, "mse": avg_mse})
            print(f"    {target_bpw:.1f}  actual={actual_bpw:.3f}  MSE={avg_mse:.6e}", flush=True)

        results[f"{budget_type}_pareto"] = pareto

    # Compare raw vs entropy Pareto
    print(f"\n  Entropy vs Raw Pareto comparison:", flush=True)
    raw_p = results["raw_pareto"]
    ent_p = results["entropy_pareto"]
    for r, e in zip(raw_p, ent_p):
        if r["target"] == e["target"]:
            imp = (1 - e["mse"] / r["mse"]) * 100 if r["mse"] > 0 else 0
            print(f"    {r['target']:.1f}  raw_MSE={r['mse']:.4e}  ent_MSE={e['mse']:.4e}  improvement={imp:.1f}%", flush=True)

    results["bitplane"] = {k: sum(v)/len(v) for k, v in bp_results.items()}
    results["lloyd_max"] = {k: sum(v)/len(v) for k, v in lm_results.items()}
    return results

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--data-dir", default="/tmp/poc_residual/glm52_data")
    ap.add_argument("--out", default="/tmp/poc_residual/results_v32.json")
    args = ap.parse_args()
    dev = torch.device(args.device)
    print(f"Device: {dev}  GPU: {torch.cuda.get_device_name(0)}", flush=True)
    ext, ghd, tcp, tcpi, qtf, cbs_scale = _bootstrap()
    results = run_experiment(Path(args.data_dir), dev, ext, ghd, tcp, tcpi, qtf, cbs_scale)
    Path(args.out).write_text(json.dumps(results, indent=2, default=str))
    print(f"\nResults saved to {args.out}", flush=True)

if __name__ == "__main__":
    main()
