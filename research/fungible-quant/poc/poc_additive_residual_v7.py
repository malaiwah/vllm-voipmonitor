#!/usr/bin/env python3
"""PoC v7: Tile-level mixed precision — K3 base with per-tile K4 upgrades.

Instead of sparse fp16 corrections (22 bits/entry), upgrade individual 16x16 tiles
from K3 to K4. Cost: exactly 1 bit/weight for upgraded tiles, ~0 index overhead
(1-bit bitmap per tile = 1/256 bpw). This is the Q-Palette "half-TCQ" idea applied
at tile granularity.

Key experiments:
  1. K3 base + upgrade top-k% most-damaged tiles to K4
  2. Per-expert tile-upgrade budget allocation (water-filling)
  3. Multi-level: K3 + K3.5 tiles + K4 tiles (3-tier)
  4. Compare vs K4 standalone and K3+2lloyd at matching bit budgets
  5. Continuously variable bpw by controlling upgrade fraction
"""
from __future__ import annotations
import argparse, json, math, os, sys, types, importlib.util, time
from pathlib import Path
import torch, torch.nn.functional as F

EXL3_PKG = "/opt/fruit-pip/exllamav3"
HAD_K, HAD_N = 128, 128
TILE_K, TILE_N = 16, 16  # EXL3 trellis tile size

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

def quantize_trellis_tilewise(w_reg, K, device, tcp, tcpi, qtf):
    """Quantize tile-by-tile and return both the reconstructed weight and per-tile MSE.
    Returns: weight_q (full reconstruction), tile_mses (n_tiles_k, n_tiles_n)
    """
    k, n = w_reg.shape
    tiles_n_k = k // TILE_K
    tiles_n_n = n // TILE_N
    tiles_n = n // TILE_N
    weight_q = torch.zeros_like(w_reg)
    tile_mses = torch.zeros(tiles_n_k, tiles_n_n, device=device)
    
    qa = {"K": K, "mcg": True}
    perm = tcp(device); perm_i = tcpi(device)
    
    for bi in range(0, k, TILE_K):
        ti_k = bi // TILE_K
        rows = w_reg[bi:bi+TILE_K]
        tiles = rows.reshape(TILE_K, tiles_n, TILE_N).permute(1, 0, 2).reshape(tiles_n, 256)
        tiles = tiles[:, perm].contiguous()
        quant_w, _ = qtf(tiles, qa)
        quant_w = quant_w[:, perm_i].reshape(tiles_n, TILE_K, TILE_N).permute(1, 0, 2).reshape(TILE_K, n)
        weight_q[bi:bi+TILE_K] = quant_w
        
        # Per-tile MSE
        for ti_n in range(tiles_n):
            orig_tile = rows[:, ti_n*TILE_N:(ti_n+1)*TILE_N]
            quant_tile = quant_w[:, ti_n*TILE_N:(ti_n+1)*TILE_N]
            tile_mses[ti_k, ti_n] = (orig_tile - quant_tile).pow(2).mean()
    
    return weight_q, tile_mses

def q2b_lloyd(r):
    sigma = r.std().item()
    if sigma < 1e-12: return torch.zeros_like(r)
    levels = torch.tensor([-1.5104, -0.4528, 0.4528, 1.5104], device=r.device) * sigma
    flat = r.flatten().unsqueeze(1)
    d = torch.cdist(flat, levels.unsqueeze(1))
    return levels[d.argmin(dim=1)].reshape(r.shape)

def tile_mixed_precision(w_reg, device, tcp, tcpi, qtf, upgrade_fraction):
    """K3 base + upgrade top-k% most-damaged tiles to K4.
    
    upgrade_fraction: fraction of tiles to upgrade from K3 to K4 (0.0 to 1.0)
    Returns: reconstructed weight, effective bits per weight
    """
    k, n = w_reg.shape
    tiles_n_k = k // TILE_K
    tiles_n_n = n // TILE_N
    n_tiles = tiles_n_k * tiles_n_n
    
    # Quantize all tiles at K3 and K4
    qk3, tile_mse_k3 = quantize_trellis_tilewise(w_reg, 3, device, tcp, tcpi, qtf)
    qk4, tile_mse_k4 = quantize_trellis_tilewise(w_reg, 4, device, tcp, tcpi, qtf)
    
    # Compute per-tile improvement: how much K4 helps over K3
    tile_improvement = tile_mse_k3 - tile_mse_k4  # higher = more benefit from upgrade
    
    # Select top-k% tiles to upgrade
    n_upgrade = int(n_tiles * upgrade_fraction)
    if n_upgrade == 0:
        return qk3, 3.0 + 1.0 / n_tiles  # just K3 + tiny bitmap overhead
    if n_upgrade >= n_tiles:
        return qk4, 4.0 + 1.0 / n_tiles  # just K4
    
    # Get indices of tiles with highest improvement
    flat_improvement = tile_improvement.flatten()
    _, top_indices = flat_improvement.topk(n_upgrade)
    
    # Build mixed-precision reconstruction
    result = qk3.clone()
    for idx in top_indices:
        ti_k = idx.item() // tiles_n_n
        ti_n = idx.item() % tiles_n_n
        # Replace this tile's K3 reconstruction with K4
        r_start, r_end = ti_k * TILE_K, (ti_k + 1) * TILE_K
        c_start, c_end = ti_n * TILE_N, (ti_n + 1) * TILE_N
        result[r_start:r_end, c_start:c_end] = qk4[r_start:r_end, c_start:c_end]
    
    # Effective bits: 3 + upgrade_fraction * 1 + bitmap_overhead
    bitmap_bpw = n_tiles / (k * n)  # 1 bit per tile, amortized over all weights
    eff_bits = 3.0 + upgrade_fraction * 1.0 + bitmap_bpw
    
    return result, eff_bits

def tile_mixed_3tier(w_reg, device, tcp, tcpi, qtf, frac_k4, frac_k5):
    """3-tier: K3 base + some tiles at K4 + some tiles at K5 (3+2lloyd).
    frac_k4: fraction upgraded to K4
    frac_k5: fraction upgraded from K4 to K4+1bit (or K3+2lloyd)
    """
    k, n = w_reg.shape
    tiles_n_k = k // TILE_K
    tiles_n_n = n // TILE_N
    n_tiles = tiles_n_k * tiles_n_n
    
    qk3, tile_mse_k3 = quantize_trellis_tilewise(w_reg, 3, device, tcp, tcpi, qtf)
    qk4, tile_mse_k4 = quantize_trellis_tilewise(w_reg, 4, device, tcp, tcpi, qtf)
    
    # Tile improvement K3->K4
    tile_improvement = tile_mse_k3 - tile_mse_k4
    flat_improvement = tile_improvement.flatten()
    
    # Rank tiles by improvement
    _, sorted_indices = flat_improvement.sort(descending=True)
    
    n_k4 = int(n_tiles * frac_k4)
    n_k5 = int(n_tiles * frac_k5)
    
    # Top n_k5 tiles get K4+2lloyd (5 bits), next n_k4 get K4 (4 bits), rest K3
    result = qk3.clone()
    k5_indices = sorted_indices[:n_k5]
    k4_indices = sorted_indices[n_k5:n_k5 + n_k4]
    
    # Apply K4 to k4 and k5 tiles
    for idx in torch.cat([k4_indices, k5_indices]):
        ti_k = idx.item() // tiles_n_n
        ti_n = idx.item() % tiles_n_n
        r_start, r_end = ti_k * TILE_K, (ti_k + 1) * TILE_K
        c_start, c_end = ti_n * TILE_N, (ti_n + 1) * TILE_N
        result[r_start:r_end, c_start:c_end] = qk4[r_start:r_end, c_start:c_end]
    
    # Apply 2-bit Lloyd-Max on the K4 residual for k5 tiles
    if n_k5 > 0:
        r4 = w_reg - qk4  # K4 residual
        lloyd = q2b_lloyd(r4)
        for idx in k5_indices:
            ti_k = idx.item() // tiles_n_n
            ti_n = idx.item() % tiles_n_n
            r_start, r_end = ti_k * TILE_K, (ti_k + 1) * TILE_K
            c_start, c_end = ti_n * TILE_N, (ti_n + 1) * TILE_N
            result[r_start:r_end, c_start:c_end] = (
                qk4[r_start:r_end, c_start:c_end] + lloyd[r_start:r_end, c_start:c_end]
            )
    
    bitmap_bpw = 2 * n_tiles / (k * n)  # 2-bit bitmap per tile
    eff_bits = 3.0 + (frac_k4 + frac_k5) * 1.0 + frac_k5 * 2.0 + bitmap_bpw
    
    return result, eff_bits

def run_experiment(data_dir, device, ext, ghd, tcp, tcpi, qtf, cbs, layer_indices=[10, 40], max_experts=20):
    results = {}
    
    for layer_idx in layer_indices:
        print(f"\n{'='*70}", flush=True)
        print(f"Layer {layer_idx}", flush=True)
        print(f"{'='*70}", flush=True)
        
        gate_file = data_dir / f"layer{layer_idx}_all_gate_proj.pt"
        if not gate_file.exists():
            print(f"  SKIP: {gate_file} not found", flush=True)
            continue
        
        all_experts = torch.load(gate_file, map_location="cpu")
        n_experts_total = all_experts.shape[0]
        n_experts = min(n_experts_total, max_experts)  # limit for speed
        k, n = all_experts.shape[1], all_experts.shape[2]
        n_weights = k * n
        n_tiles = (k // TILE_K) * (n // TILE_N)
        print(f"  Using {n_experts}/{n_experts_total} experts, shape=({k},{n}), "
              f"{n_weights} weights, {n_tiles} tiles", flush=True)
        
        layer_results = {}
        
        # Baselines: K3 and K4 standalone
        k3_mses = []; k4_mses = []
        print(f"  Computing baselines (K3, K4)...", flush=True)
        for ei in range(n_experts):
            w = all_experts[ei].to(device)
            w_reg, _, _ = regularize(w, device, ghd, cbs)
            qk3 = quantize_trellis_tilewise(w_reg, 3, device, tcp, tcpi, qtf)[0]
            qk4 = quantize_trellis_tilewise(w_reg, 4, device, tcp, tcpi, qtf)[0]
            k3_mses.append((w_reg - qk3).pow(2).mean().item())
            k4_mses.append((w_reg - qk4).pow(2).mean().item())
            del w, w_reg, qk3, qk4
            torch.cuda.empty_cache()
        
        k3_avg = sum(k3_mses) / n_experts
        k4_avg = sum(k4_mses) / n_experts
        print(f"  K3 avg MSE: {k3_avg:.6e}, K4 avg MSE: {k4_avg:.6e}", flush=True)
        
        # K3+2lloyd baseline (5 bits)
        print(f"  Computing K3+2lloyd (5 bits)...", flush=True)
        lloyd_mses = []
        for ei in range(n_experts):
            w = all_experts[ei].to(device)
            w_reg, _, _ = regularize(w, device, ghd, cbs)
            qk3 = quantize_trellis_tilewise(w_reg, 3, device, tcp, tcpi, qtf)[0]
            r3 = w_reg - qk3
            lloyd = q2b_lloyd(r3)
            lloyd_mses.append((w_reg - qk3 - lloyd).pow(2).mean().item())
            del w, w_reg, qk3, r3, lloyd
            torch.cuda.empty_cache()
        lloyd_avg = sum(lloyd_mses) / n_experts
        gap_lloyd = (k3_avg - lloyd_avg) / (k3_avg - k4_avg)
        layer_results["K3+2lloyd_5bit"] = {"mse": lloyd_avg, "bits": 5.0, "gap": gap_lloyd}
        print(f"    K3+2lloyd: MSE={lloyd_avg:.6e}  bits=5.0  gap={gap_lloyd:.1%}", flush=True)
        
        # ================================================================
        # Experiment 1: Tile-level K3→K4 upgrade (continuously variable)
        # ================================================================
        print(f"\n  --- Exp 1: Tile K3→K4 upgrade ---", flush=True)
        for frac in [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0]:
            mses = []; bits_list = []
            for ei in range(n_experts):
                w = all_experts[ei].to(device)
                w_reg, _, _ = regularize(w, device, ghd, cbs)
                recon, eff_bits = tile_mixed_precision(w_reg, device, tcp, tcpi, qtf, frac)
                mses.append((w_reg - recon).pow(2).mean().item())
                bits_list.append(eff_bits)
                del w, w_reg, recon
                torch.cuda.empty_cache()
            avg_mse = sum(mses) / n_experts
            avg_bits = sum(bits_list) / n_experts
            gap = (k3_avg - avg_mse) / (k3_avg - k4_avg)
            label = f"K3+tile_K4_{frac:.2f}"
            layer_results[label] = {"mse": avg_mse, "bits": avg_bits, "gap": gap}
            print(f"    {label}: MSE={avg_mse:.6e}  bits={avg_bits:.3f}  gap={gap:.1%}", flush=True)
        
        # ================================================================
        # Experiment 2: 3-tier (K3 + K4 tiles + K4+2lloyd tiles)
        # ================================================================
        print(f"\n  --- Exp 2: 3-tier K3+K4+K5 tiles ---", flush=True)
        for frac_k4, frac_k5 in [(0.1, 0.1), (0.25, 0.25), (0.5, 0.0), (0.0, 0.5), (0.25, 0.5), (0.5, 0.25)]:
            mses = []; bits_list = []
            for ei in range(n_experts):
                w = all_experts[ei].to(device)
                w_reg, _, _ = regularize(w, device, ghd, cbs)
                recon, eff_bits = tile_mixed_3tier(w_reg, device, tcp, tcpi, qtf, frac_k4, frac_k5)
                mses.append((w_reg - recon).pow(2).mean().item())
                bits_list.append(eff_bits)
                del w, w_reg, recon
                torch.cuda.empty_cache()
            avg_mse = sum(mses) / n_experts
            avg_bits = sum(bits_list) / n_experts
            gap = (k3_avg - avg_mse) / (k3_avg - k4_avg)
            label = f"K3+3tier_K4_{frac_k4:.2f}_K5_{frac_k5:.2f}"
            layer_results[label] = {"mse": avg_mse, "bits": avg_bits, "gap": gap}
            print(f"    {label}: MSE={avg_mse:.6e}  bits={avg_bits:.3f}  gap={gap:.1%}", flush=True)
        
        # ================================================================
        # Experiment 3: Per-expert variable allocation (water-filling on tiles)
        # ================================================================
        print(f"\n  --- Exp 3: Per-expert water-filling on tiles ---", flush=True)
        # Use per-expert K3 MSE as damage signal
        for total_bpw in [3.5, 4.0, 4.5, 5.0]:
            # Total budget = total_bpw * n_experts * n_weights
            # Each expert gets K3 (3 bpw) + upgrade_fraction * 1 bpw
            # So avg upgrade_fraction = (total_bpw - 3) / 1
            avg_upgrade = total_bpw - 3.0
            if avg_upgrade <= 0:
                continue
            
            # Water-filling: give more upgrade to more damaged experts
            # But since damages are nearly identical, this reduces to uniform
            # Let's try a more aggressive allocation: give ALL budget to top-k experts
            # and zero to the rest
            mses = []; bits_list = []
            
            # Sort by damage
            sorted_ei = sorted(range(n_experts), key=lambda i: k3_mses[i], reverse=True)
            
            # Give full upgrade to top experts, none to rest
            n_upgraded = int(n_experts * avg_upgrade)  # fraction of experts fully upgraded
            expert_fracs = [0.0] * n_experts
            for rank, ei in enumerate(sorted_ei):
                if rank < n_upgraded:
                    expert_fracs[ei] = 1.0  # fully K4
                elif rank == n_upgraded and avg_upgrade * n_experts > n_upgraded:
                    expert_fracs[ei] = avg_upgrade * n_experts - n_upgraded  # partial
            
            for ei in range(n_experts):
                w = all_experts[ei].to(device)
                w_reg, _, _ = regularize(w, device, ghd, cbs)
                recon, eff_bits = tile_mixed_precision(w_reg, device, tcp, tcpi, qtf, expert_fracs[ei])
                mses.append((w_reg - recon).pow(2).mean().item())
                bits_list.append(eff_bits)
                del w, w_reg, recon
                torch.cuda.empty_cache()
            
            avg_mse = sum(mses) / n_experts
            avg_bits = sum(bits_list) / n_experts
            gap = (k3_avg - avg_mse) / (k3_avg - k4_avg)
            label = f"K3+expert_waterfill_{total_bpw:.1f}bpw"
            layer_results[label] = {"mse": avg_mse, "bits": avg_bits, "gap": gap,
                                     "n_upgraded": n_upgraded}
            print(f"    {label}: MSE={avg_mse:.6e}  bits={avg_bits:.3f}  gap={gap:.1%}  "
                  f"upgraded={n_upgraded}/{n_experts}", flush=True)
        
        results[f"layer{layer_idx}"] = {
            "k3_avg_mse": k3_avg, "k4_avg_mse": k4_avg,
            "k3_mses": k3_mses, "k4_mses": k4_mses,
            "methods": layer_results, "n_experts": n_experts
        }
    
    return results

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--data-dir", default="/tmp/poc_residual/glm52_data")
    ap.add_argument("--out", default="/tmp/poc_residual/results_v7.json")
    ap.add_argument("--max-experts", type=int, default=20)
    args = ap.parse_args()
    dev = torch.device(args.device)
    print(f"Device: {dev}  GPU: {torch.cuda.get_device_name(0)}", flush=True)
    ext, ghd, tcp, tcpi, qtf, cbs = _bootstrap()
    
    results = run_experiment(Path(args.data_dir), dev, ext, ghd, tcp, tcpi, qtf, cbs,
                             max_experts=args.max_experts)
    
    print(f"\n{'='*70}\nSUMMARY\n{'='*70}", flush=True)
    for layer_key in sorted(results.keys()):
        r = results[layer_key]
        print(f"\n{layer_key}: K3={r['k3_avg_mse']:.4e} K4={r['k4_avg_mse']:.4e}", flush=True)
        methods = r["methods"]
        for label in sorted(methods.keys(), key=lambda x: methods[x].get("gap", 0), reverse=True):
            m = methods[label]
            print(f"    {label:40s}: MSE={m['mse']:.6e}  bits={m['bits']:.3f}  "
                  f"gap={m['gap']:.1%}", flush=True)
        
        # Pareto frontier
        print(f"\n  Pareto frontier:", flush=True)
        bit_buckets = {}
        for label, m in methods.items():
            bucket = round(m["bits"] * 2) / 2
            if bucket not in bit_buckets or m["mse"] < bit_buckets[bucket][1]:
                bit_buckets[bucket] = (label, m["mse"], m["bits"])
        for bucket in sorted(bit_buckets.keys()):
            label, mse, bits = bit_buckets[bucket]
            print(f"    ~{bucket:.1f} bits: {label:40s}  MSE={mse:.6e}", flush=True)
    
    Path(args.out).write_text(json.dumps(results, indent=2, default=str))
    print(f"\nResults saved to {args.out}", flush=True)

if __name__ == "__main__":
    main()
