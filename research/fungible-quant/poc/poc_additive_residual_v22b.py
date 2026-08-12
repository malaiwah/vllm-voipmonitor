#!/usr/bin/env python3
"""PoC v22b: Corrected Pareto with proper bpw + missing tiers.

Critical fix: K5=K4+2LM is 6 bpw, not 5. Previous Pareto frontiers (v15-v21)
had systematic bpw labeling error: each tier upgrade was treated as +1 bpw,
but 2-bit LM adds +2 bpw.

New tiers tested:
  K3 (3 bpw), K3+1LM (4), K3+2LM (5), K3+3LM (6), K3+4LM (7)
  K4 (4 bpw), K4+1LM (5), K4+2LM (6), K4+3LM (7), K4+4LM (8)

Builds corrected tile-level Pareto frontier with actual bpw.
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

def lloyd_max_quantize(r, n_bits, n_iters=20):
    n_levels = 2 ** n_bits
    sigma = r.std().item()
    if sigma < 1e-12: return torch.zeros_like(r)
    levels = torch.linspace(-3 * sigma, 3 * sigma, n_levels, device=r.device)
    flat = r.flatten()
    n_elem = flat.numel()
    chunk = max(1, min(n_elem, (1024 * 1024 * 1024) // (n_levels * 4)))
    for _ in range(n_iters):
        assign = torch.empty(n_elem, dtype=torch.long, device=r.device)
        for s in range(0, n_elem, chunk):
            e = min(s + chunk, n_elem)
            d = (flat[s:e].unsqueeze(1) - levels.unsqueeze(0)).abs()
            assign[s:e] = d.argmin(dim=1)
        new_levels = levels.clone()
        for i in range(n_levels):
            mask = assign == i
            if mask.sum() > 0: new_levels[i] = flat[mask].mean()
        if (new_levels - levels).abs().max() < 1e-10 * sigma: break
        levels = new_levels
    result = torch.empty_like(flat)
    for s in range(0, n_elem, chunk):
        e = min(s + chunk, n_elem)
        d = (flat[s:e].unsqueeze(1) - levels.unsqueeze(0)).abs()
        result[s:e] = levels[d.argmin(dim=1)]
    return result.reshape(r.shape)

def q1b_scalar(r):
    s = r.abs().mean().item()
    return torch.zeros_like(r) if s < 1e-12 else torch.sign(r) * s

def tile_mse(w_reg, recon, k, n):
    """Compute per-tile MSE."""
    tnk, tnn = k // 16, n // 16
    mses = torch.zeros(tnk, tnn, device=w_reg.device)
    for ik in range(tnk):
        for inn in range(tnn):
            rs, re = ik*16, (ik+1)*16
            cs, ce = inn*16, (inn+1)*16
            mses[ik, inn] = (w_reg[rs:re, cs:ce] - recon[rs:re, cs:ce]).pow(2).mean()
    return mses

def run_experiment(data_dir, device, ext, ghd, tcp, tcpi, qtf, cbs):
    results = {}
    layer_idx = 10
    gate_file = data_dir / f"layer{layer_idx}_all_gate_proj.pt"
    if not gate_file.exists(): return results
    
    all_experts = torch.load(gate_file, map_location="cpu")
    n_experts = min(2, all_experts.shape[0])
    k, n = all_experts.shape[1], all_experts.shape[2]
    n_tiles = (k // 16) * (n // 16)
    print(f"  {n_experts} experts, {n_tiles} tiles each", flush=True)
    
    # Collect per-tier data across experts
    tier_data = {}  # tier_name -> {"bpw": float, "mses": [per-expert avg mse], "tile_mses": [per-expert tile mses]}
    
    for ei in range(n_experts):
        w = all_experts[ei].to(device)
        w_reg, _, _ = regularize(w, device, ghd, cbs)
        del w
        
        # Compute all tiers
        qk3 = quantize_trellis(w_reg, 3, device, tcp, tcpi, qtf)
        qk4 = quantize_trellis(w_reg, 4, device, tcp, tcpi, qtf)
        r3 = w_reg - qk3
        r4 = w_reg - qk4
        
        tiers = {
            "K3": (qk3, 3.0),
            "K4": (qk4, 4.0),
        }
        
        # K3 + N-bit LM
        for nbits in [1, 2, 3, 4]:
            lm = lloyd_max_quantize(r3, nbits)
            recon = qk3 + lm
            tiers[f"K3+{nbits}LM"] = (recon, 3.0 + nbits)
            del lm, recon
        
        # K4 + N-bit LM
        for nbits in [1, 2, 3, 4]:
            lm = lloyd_max_quantize(r4, nbits)
            recon = qk4 + lm
            tiers[f"K4+{nbits}LM"] = (recon, 4.0 + nbits)
            del lm, recon
        
        # K4 + 2LM + 1sc (old K6 approach, 7 bpw)
        lm2_r4 = lloyd_max_quantize(r4, 2)
        recon_k5 = qk4 + lm2_r4
        r5 = w_reg - recon_k5
        sc1 = q1b_scalar(r5)
        tiers["K4+2LM+1sc"] = (recon_k5 + sc1, 7.0)
        
        # K3 + 1sc (4 bpw alternative to K4)
        sc_r3 = q1b_scalar(r3)
        tiers["K3+1sc"] = (qk3 + sc_r3, 4.0)
        
        # Compute tile-level MSEs and avg MSE for each tier
        for name, (recon, bpw) in tiers.items():
            tmse = tile_mse(w_reg, recon, k, n)
            avg_mse = tmse.mean().item()
            if name not in tier_data:
                tier_data[name] = {"bpw": bpw, "avg_mses": [], "tile_mses": []}
            tier_data[name]["avg_mses"].append(avg_mse)
            tier_data[name]["tile_mses"].append(tmse.cpu())
            del tmse
        
        # Cleanup
        del w_reg, qk3, qk4, r3, r4, lm2_r4, recon_k5, r5, sc1, sc_r3
        for name, (recon, _) in list(tiers.items()):
            del recon
        tiers.clear()
        torch.cuda.empty_cache()
    
    # Print tier comparison
    print(f"\n  {'Tier':<20} {'bpw':>5} {'avg MSE':>12}", flush=True)
    print(f"  {'-'*40}", flush=True)
    for name in sorted(tier_data.keys(), key=lambda x: tier_data[x]["bpw"]):
        d = tier_data[name]
        avg = sum(d["avg_mses"]) / len(d["avg_mses"])
        print(f"  {name:<20} {d['bpw']:>5.0f} {avg:>12.4e}", flush=True)
    
    # Build corrected Pareto frontier using best tiers
    # Available tiers with their actual bpw costs:
    # K3(3), K3+1sc(4), K4(4), K3+1LM(4), K3+2LM(5), K4+1LM(5),
    # K3+3LM(6), K4+2LM(6), K3+4LM(7), K4+3LM(7), K4+2LM+1sc(7), K4+4LM(8)
    
    # For Pareto, use the best tier at each bpw level
    best_at_bpw = {}
    for name, d in tier_data.items():
        bpw = d["bpw"]
        avg = sum(d["avg_mses"]) / len(d["avg_mses"])
        if bpw not in best_at_bpw or avg < best_at_bpw[bpw][1]:
            best_at_bpw[bpw] = (name, avg)
    
    print(f"\n  Best tier at each bpw:", flush=True)
    for bpw in sorted(best_at_bpw.keys()):
        name, mse = best_at_bpw[bpw]
        print(f"    {bpw:.0f} bpw: {name}  MSE={mse:.4e}", flush=True)
    
    # Build tile-level Pareto with correct bpw
    # Use the best tiers for mixing: K3(3), K4(4), K3+2LM(5), K4+2LM(6), K3+4LM(7)
    pareto_tiers = ["K3", "K4", "K3+2LM", "K4+2LM", "K3+4LM", "K4+4LM"]
    pareto_bpw = [3, 4, 5, 6, 7, 8]
    
    # Check we have tile-level data for these tiers
    available = [t for t in pareto_tiers if t in tier_data]
    if len(available) < 2:
        print("  Not enough tiers for Pareto", flush=True)
        return results
    
    # Average tile MSEs across experts
    avg_tile_mses = {}
    for name in available:
        tmses = tier_data[name]["tile_mses"]
        avg_tmse = sum(tmses) / len(tmses)  # average across experts
        avg_tile_mses[name] = avg_tmse.to(device)
    
    # Build upgrades list: for each pair of adjacent tiers, compute per-tile benefit
    upgrades = []
    for i in range(len(available) - 1):
        lo, hi = available[i], available[i + 1]
        lo_bpw = tier_data[lo]["bpw"]
        hi_bpw = tier_data[hi]["bpw"]
        bit_cost = hi_bpw - lo_bpw  # actual bit cost of upgrade
        benefit = (avg_tile_mses[lo] - avg_tile_mses[hi]).flatten()
        for ti in range(n_tiles):
            upgrades.append((benefit[ti].item(), ti, lo, hi, bit_cost))
    upgrades.sort(key=lambda x: -x[0])
    
    # Greedy Pareto
    print(f"\n  Corrected tile-level Pareto (actual bpw):", flush=True)
    pareto = []
    for target_bpw_10 in range(30, 81):  # 3.0 to 8.0
        target_bpw = target_bpw_10 / 10.0
        tier_assignment = {t: [False] * n_tiles for t in available}
        for t in available:
            tier_assignment[t] = [True] * n_tiles if t == available[0] else [False] * n_tiles
        
        current_bits = pareto_bpw[0] * n_tiles  # start at lowest tier
        target_bits = target_bpw * n_tiles
        
        for benefit, tile_idx, lo, hi, bit_cost in upgrades:
            if current_bits + bit_cost > target_bits + 1e-6: continue
            # Check tile is currently at 'lo' tier and not yet at 'hi'
            if not tier_assignment[lo][tile_idx]: continue
            if tier_assignment[hi][tile_idx]: continue
            if benefit <= 0: continue
            # Also check no intermediate tier has this tile
            lo_idx = available.index(lo)
            hi_idx = available.index(hi)
            if lo_idx + 1 != hi_idx: continue  # only adjacent tiers
            
            tier_assignment[lo][tile_idx] = False
            tier_assignment[hi][tile_idx] = True
            current_bits += bit_cost
        
        # Compute actual MSE
        total_mse = 0.0
        for ti in range(n_tiles):
            for t in available:
                if tier_assignment[t][ti]:
                    tik = ti // (n // 16)
                    tin = ti % (n // 16)
                    total_mse += avg_tile_mses[t][tik, tin].item()
                    break
        avg_mse = total_mse / n_tiles
        actual_bpw = current_bits / n_tiles
        
        pareto.append({"target_bpw": target_bpw, "actual_bpw": actual_bpw, "mse": avg_mse})
        if target_bpw_10 % 5 == 0:
            print(f"    {target_bpw:.1f}  actual={actual_bpw:.3f}  MSE={avg_mse:.6e}", flush=True)
    
    results["tiers"] = {k: {"bpw": v["bpw"], "avg_mse": sum(v["avg_mses"])/len(v["avg_mses"])}
                        for k, v in tier_data.items()}
    results["best_at_bpw"] = {str(k): {"name": v[0], "mse": v[1]} for k, v in best_at_bpw.items()}
    results["pareto"] = pareto
    return results

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--data-dir", default="/tmp/poc_residual/glm52_data")
    ap.add_argument("--out", default="/tmp/poc_residual/results_v22b.json")
    args = ap.parse_args()
    dev = torch.device(args.device)
    print(f"Device: {dev}  GPU: {torch.cuda.get_device_name(0)}", flush=True)
    ext, ghd, tcp, tcpi, qtf, cbs = _bootstrap()
    results = run_experiment(Path(args.data_dir), dev, ext, ghd, tcp, tcpi, qtf, cbs)
    Path(args.out).write_text(json.dumps(results, indent=2, default=str))
    print(f"\nResults saved to {args.out}", flush=True)

if __name__ == "__main__":
    main()
