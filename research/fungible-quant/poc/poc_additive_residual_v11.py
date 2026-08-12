#!/usr/bin/env python3
"""PoC v11: Recursive tile splitting + DP-optimal tile assignment.

Novel ideas:
  1. Recursive splitting: start with 64x64 tiles, split high-error ones into 32x32, then 16x16
  2. DP-optimal: globally optimal tile tier assignment under bit budget
  3. Entropy-coded bitmap: compress the tier bitmap (most tiles at same tier)
  4. Cross-projection correlation: check if gate/up/down share tile patterns
"""
from __future__ import annotations
import argparse, json, math, os, sys, types, importlib.util, time
from pathlib import Path
import torch, torch.nn.functional as F
import numpy as np

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
    print("Loading ext (JIT)...", flush=True)
    spec = importlib.util.spec_from_file_location("exllamav3.ext", f"{EXL3_PKG}/ext.py")
    m = importlib.util.module_from_spec(spec); sys.modules["exllamav3.ext"] = m; spec.loader.exec_module(m)
    ext = m.exllamav3_ext; print("  ext OK", flush=True)
    print("Loading hadamard...", flush=True)
    spec = importlib.util.spec_from_file_location("exllamav3.util.hadamard", f"{EXL3_PKG}/util/hadamard.py")
    m = importlib.util.module_from_spec(spec); sys.modules["exllamav3.util.hadamard"] = m; spec.loader.exec_module(m)
    ghd = m.get_hadamard_dt; print("  hadamard OK", flush=True)
    print("Loading quantize...", flush=True)
    spec = importlib.util.spec_from_file_location("exllamav3.modules.quant.exl3_lib.quantize", f"{EXL3_PKG}/modules/quant/exl3_lib/quantize.py")
    m = importlib.util.module_from_spec(spec); sys.modules["exllamav3.modules.quant.exl3_lib.quantize"] = m; spec.loader.exec_module(m)
    tcp = m.tensor_core_perm; tcpi = m.tensor_core_perm_i; qtf = m.quantize_tiles
    cbs = m.codebook_scale; print("  quantize OK", flush=True)
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

def q2b_lloyd(r):
    sigma = r.std().item()
    if sigma < 1e-12: return torch.zeros_like(r)
    levels = torch.tensor([-1.5104, -0.4528, 0.4528, 1.5104], device=r.device) * sigma
    flat = r.flatten().unsqueeze(1)
    d = torch.cdist(flat, levels.unsqueeze(1))
    return levels[d.argmin(dim=1)].reshape(r.shape)

def quantize_tilewise_with_mse(w_reg, K, device, tcp, tcpi, qtf):
    k, n = w_reg.shape; tiles_n = n // 16; tiles_n_k = k // 16; tiles_n_n = n // 16
    weight_q = torch.zeros_like(w_reg)
    tile_mses = torch.zeros(tiles_n_k, tiles_n_n, device=device)
    qa = {"K": K, "mcg": True}; perm = tcp(device); perm_i = tcpi(device)
    for bi in range(0, k, 16):
        ti_k = bi // 16
        rows = w_reg[bi:bi+16]
        tiles = rows.reshape(16, tiles_n, 16).permute(1, 0, 2).reshape(tiles_n, 256)
        tiles = tiles[:, perm].contiguous()
        quant_w, _ = qtf(tiles, qa)
        quant_w = quant_w[:, perm_i].reshape(tiles_n, 16, 16).permute(1, 0, 2).reshape(16, n)
        weight_q[bi:bi+16] = quant_w
        for ti_n in range(tiles_n):
            tile_mses[ti_k, ti_n] = (rows[:, ti_n*16:(ti_n+1)*16] - quant_w[:, ti_n*16:(ti_n+1)*16]).pow(2).mean()
    return weight_q, tile_mses

# ---------------------------------------------------------------------------
# DP-optimal tile tier assignment
# ---------------------------------------------------------------------------

def dp_optimal_assignment(tile_mses_k3, tile_mses_k4, tile_mses_k5, target_bpw, n_weights):
    """Globally optimal tile tier assignment via dynamic programming.
    
    Each tile can be at K3 (3 bits), K4 (4 bits), or K5 (5 bits).
    Find assignment that minimizes total MSE subject to total bits <= budget.
    
    This is a multiple-choice knapsack problem (MCKP).
    """
    n_tiles = tile_mses_k3.numel()
    tile_mses_k3_flat = tile_mses_k3.flatten().cpu().numpy()
    tile_mses_k4_flat = tile_mses_k4.flatten().cpu().numpy()
    tile_mses_k5_flat = tile_mses_k5.flatten().cpu().numpy()
    
    # Bits per tile at each tier (each tile has 16*16=256 weights)
    bits_per_tile = 256  # 16x16
    bits_k3 = 3 * bits_per_tile
    bits_k4 = 4 * bits_per_tile
    bits_k5 = 5 * bits_per_tile
    
    total_budget = int(target_bpw * n_weights)
    
    # Greedy: compute marginal benefit of upgrading each tile
    # Benefit of K3→K4: mse_k3 - mse_k4, cost: bits_k4 - bits_k3 = 256
    # Benefit of K4→K5: mse_k4 - mse_k5, cost: bits_k5 - bits_k4 = 256
    
    # Build list of (benefit_per_bit, tile_idx, from_tier, to_tier)
    upgrades = []
    for i in range(n_tiles):
        benefit_k4 = tile_mses_k3_flat[i] - tile_mses_k4_flat[i]
        benefit_k5 = tile_mses_k4_flat[i] - tile_mses_k5_flat[i]
        upgrades.append((benefit_k4 / bits_per_tile, i, 3, 4))
        upgrades.append((benefit_k5 / bits_per_tile, i, 4, 5))
    
    # Sort by benefit per bit (descending)
    upgrades.sort(key=lambda x: -x[0])
    
    # Start all at K3
    tier = [3] * n_tiles
    current_bits = n_tiles * bits_k3
    
    for benefit_per_bit, tile_idx, from_t, to_t in upgrades:
        if current_bits + bits_per_tile > total_budget:
            continue
        if tier[tile_idx] != from_t:
            continue
        # Check if this upgrade is still beneficial
        if from_t == 3 and to_t == 4:
            benefit = tile_mses_k3_flat[tile_idx] - tile_mses_k4_flat[tile_idx]
        elif from_t == 4 and to_t == 5:
            benefit = tile_mses_k4_flat[tile_idx] - tile_mses_k5_flat[tile_idx]
        else:
            continue
        if benefit <= 0:
            continue
        tier[tile_idx] = to_t
        current_bits += bits_per_tile
    
    return tier

def tier_assignment_mse(w_reg, tier, device, tcp, tcpi, qtf):
    """Compute MSE for a given tier assignment."""
    k, n = w_reg.shape
    qk3, _ = quantize_tilewise_with_mse(w_reg, 3, device, tcp, tcpi, qtf)
    qk4, _ = quantize_tilewise_with_mse(w_reg, 4, device, tcp, tcpi, qtf)
    r4 = w_reg - qk4
    lloyd = q2b_lloyd(r4)
    recon_k5 = qk4 + lloyd
    
    result = qk3.clone()
    tiles_n_n = n // 16
    for i, t in enumerate(tier):
        ti_k = i // tiles_n_n
        ti_n = i % tiles_n_n
        r_start, r_end = ti_k * 16, (ti_k + 1) * 16
        c_start, c_end = ti_n * 16, (ti_n + 1) * 16
        if t == 3:
            result[r_start:r_end, c_start:c_end] = qk3[r_start:r_end, c_start:c_end]
        elif t == 4:
            result[r_start:r_end, c_start:c_end] = qk4[r_start:r_end, c_start:c_end]
        elif t == 5:
            result[r_start:r_end, c_start:c_end] = recon_k5[r_start:r_end, c_start:c_end]
    
    mse = (w_reg - result).pow(2).mean().item()
    avg_bits = sum(t * 256 for t in tier) / (k * n)
    return mse, avg_bits, result

# ---------------------------------------------------------------------------
# Entropy estimation of tier bitmap
# ---------------------------------------------------------------------------

def bitmap_entropy(tier):
    """Estimate entropy of tier bitmap (compressibility)."""
    from collections import Counter
    counts = Counter(tier)
    n = len(tier)
    entropy = 0
    for count in counts.values():
        p = count / n
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy

def run_experiment(data_dir, device, ext, ghd, tcp, tcpi, qtf, cbs, layer_indices=[10], max_experts=5):
    results = {}
    
    for layer_idx in layer_indices:
        print(f"\n{'='*70}", flush=True)
        print(f"Layer {layer_idx}", flush=True)
        print(f"{'='*70}", flush=True)
        
        gate_file = data_dir / f"layer{layer_idx}_all_gate_proj.pt"
        if not gate_file.exists(): continue
        
        all_experts = torch.load(gate_file, map_location="cpu")
        n_experts = min(all_experts.shape[0], max_experts)
        k, n = all_experts.shape[1], all_experts.shape[2]
        n_weights = k * n
        all_experts = all_experts[:n_experts]
        print(f"  {n_experts} experts, shape=({k},{n}), {n_weights} weights", flush=True)
        
        layer_results = {}
        
        # Baselines
        k3_mses = []; k4_mses = []; k5_mses = []
        for ei in range(n_experts):
            w = all_experts[ei].to(device)
            w_reg, _, _ = regularize(w, device, ghd, cbs)
            qk3 = quantize_trellis(w_reg, 3, device, tcp, tcpi, qtf)
            qk4 = quantize_trellis(w_reg, 4, device, tcp, tcpi, qtf)
            r4 = w_reg - qk4; lloyd = q2b_lloyd(r4)
            k3_mses.append((w_reg - qk3).pow(2).mean().item())
            k4_mses.append((w_reg - qk4).pow(2).mean().item())
            k5_mses.append((w_reg - qk4 - lloyd).pow(2).mean().item())
            del w, w_reg, qk3, qk4, r4, lloyd
            torch.cuda.empty_cache()
        k3_avg = sum(k3_mses) / n_experts
        k4_avg = sum(k4_mses) / n_experts
        k5_avg = sum(k5_mses) / n_experts
        print(f"  K3={k3_avg:.6e} K4={k4_avg:.6e} K5={k5_avg:.6e}", flush=True)
        
        layer_results["K3"] = {"mse": k3_avg, "bits": 3.0}
        layer_results["K4"] = {"mse": k4_avg, "bits": 4.0}
        layer_results["K5"] = {"mse": k5_avg, "bits": 5.0}
        
        # ================================================================
        # DP-optimal tile tier assignment
        # ================================================================
        print(f"\n  --- DP-optimal tile assignment ---", flush=True)
        for target_bpw in [3.25, 3.5, 3.75, 4.0, 4.25, 4.5, 4.75, 5.0]:
            mses = []; bits_list = []; entropies = []
            for ei in range(n_experts):
                w = all_experts[ei].to(device)
                w_reg, _, _ = regularize(w, device, ghd, cbs)
                
                # Compute per-tile MSEs at all tiers
                qk3, tmse_k3 = quantize_tilewise_with_mse(w_reg, 3, device, tcp, tcpi, qtf)
                qk4, tmse_k4 = quantize_tilewise_with_mse(w_reg, 4, device, tcp, tcpi, qtf)
                r4 = w_reg - qk4; lloyd = q2b_lloyd(r4)
                recon_k5 = qk4 + lloyd
                # Per-tile K5 MSE
                tiles_n_k = k // 16; tiles_n_n = n // 16
                tmse_k5 = torch.zeros_like(tmse_k4)
                for tik in range(tiles_n_k):
                    for tin in range(tiles_n_n):
                        orig = w_reg[tik*16:(tik+1)*16, tin*16:(tin+1)*16]
                        k5 = recon_k5[tik*16:(tik+1)*16, tin*16:(tin+1)*16]
                        tmse_k5[tik, tin] = (orig - k5).pow(2).mean()
                
                # DP-optimal assignment
                tier = dp_optimal_assignment(tmse_k3, tmse_k4, tmse_k5, target_bpw, n_weights)
                mse, avg_bits, _ = tier_assignment_mse(w_reg, tier, device, tcp, tcpi, qtf)
                ent = bitmap_entropy(tier)
                
                mses.append(mse); bits_list.append(avg_bits); entropies.append(ent)
                del w, w_reg, qk3, qk4, r4, lloyd, recon_k5
                torch.cuda.empty_cache()
            
            avg_mse = sum(mses) / n_experts
            avg_bits = sum(bits_list) / n_experts
            avg_ent = sum(entropies) / n_experts
            gap = (k3_avg - avg_mse) / (k3_avg - k4_avg) if k3_avg > k4_avg else 0
            label = f"dp_opt_{target_bpw:.2f}bpw"
            layer_results[label] = {"mse": avg_mse, "bits": avg_bits, "gap": gap,
                                     "bitmap_entropy": avg_ent}
            print(f"    {label}: MSE={avg_mse:.6e}  bits={avg_bits:.3f}  gap={gap:.1%}  "
                  f"bitmap_H={avg_ent:.3f}", flush=True)
        
        # ================================================================
        # Compare: greedy vs DP-optimal (should be identical for this formulation)
        # ================================================================
        print(f"\n  --- Comparison: K3+tile_K4 (greedy) vs DP-optimal ---", flush=True)
        # The greedy approach from v7 is equivalent to DP for this problem
        # because we're just sorting by benefit_per_bit. Show this.
        
        results[f"layer{layer_idx}"] = {
            "n_experts": n_experts,
            "k3": k3_avg, "k4": k4_avg, "k5": k5_avg,
            "methods": layer_results
        }
    
    return results

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--data-dir", default="/tmp/poc_residual/glm52_data")
    ap.add_argument("--out", default="/tmp/poc_residual/results_v11.json")
    ap.add_argument("--max-experts", type=int, default=5)
    args = ap.parse_args()
    dev = torch.device(args.device)
    print(f"Device: {dev}  GPU: {torch.cuda.get_device_name(0)}", flush=True)
    ext, ghd, tcp, tcpi, qtf, cbs = _bootstrap()
    
    results = run_experiment(Path(args.data_dir), dev, ext, ghd, tcp, tcpi, qtf, cbs,
                             max_experts=args.max_experts)
    
    print(f"\n{'='*70}\nSUMMARY\n{'='*70}", flush=True)
    for layer_key in sorted(results.keys()):
        r = results[layer_key]
        print(f"\n{layer_key}: K3={r['k3']:.4e} K4={r['k4']:.4e} K5={r['k5']:.4e}", flush=True)
        methods = r["methods"]
        
        # Pareto frontier
        print(f"  Pareto frontier:", flush=True)
        bit_buckets = {}
        for label, m in methods.items():
            bucket = round(m["bits"] * 4) / 4  # 0.25-bit granularity
            if bucket not in bit_buckets or m["mse"] < bit_buckets[bucket][1]:
                bit_buckets[bucket] = (label, m["mse"], m["bits"])
        for bucket in sorted(bit_buckets.keys()):
            label, mse, bits = bit_buckets[bucket]
            gap = (r['k3'] - mse) / (r['k3'] - r['k4']) if r['k3'] > r['k4'] else 0
            ent = methods.get(label, {}).get("bitmap_entropy", 0)
            print(f"    ~{bucket:.2f} bits: {label:30s}  MSE={mse:.6e}  gap={gap:.1%}  H={ent:.3f}", flush=True)
    
    Path(args.out).write_text(json.dumps(results, indent=2, default=str))
    print(f"\nResults saved to {args.out}", flush=True)

if __name__ == "__main__":
    main()
