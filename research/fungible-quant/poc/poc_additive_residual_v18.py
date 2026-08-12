#!/usr/bin/env python3
"""PoC v18: Progressive upgrade bitstream — single bit per tile upgrade.

Instead of a 2-bit bitmap (4 tiers), use a progressive bitstream:
  - Start with all tiles at K3
  - Each bit in the bitstream upgrades one tile to the next tier
  - The bitstream is ORDERED by benefit-per-bit (highest benefit first)
  - To decode at bpw B: read floor((B-3) * n_tiles) bits from the bitstream

This means:
  - 3.0 bpw: 0 bits of bitmap (all K3)
  - 3.5 bpw: 0.5 * n_tiles bits of bitmap (50% upgraded to K4)
  - 4.0 bpw: 1.0 * n_tiles bits (all K4)
  - 4.5 bpw: 1.5 * n_tiles bits (50% further upgraded to K5)
  - etc.

The bitmap is 1 bit per upgrade, and the ordering is fixed at encode time.
At runtime, just read N bits from the stream to determine how many tiles to upgrade.

Storage cost: (max_bpw - 3) * n_tiles bits total, but only the first
(target_bpw - 3) * n_tiles bits are loaded at runtime.

This is the most storage-efficient fungible representation possible.
"""
from __future__ import annotations
import argparse, json, math, os, sys, types, importlib.util, time
from pathlib import Path
import torch, torch.nn.functional as F

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
    levels = torch.tensor([-1.5104, -0.4528, 0.4528, 1.5104]) * sigma
    flat_cpu = r.flatten().cpu()
    d = (flat_cpu.unsqueeze(1) - levels.unsqueeze(0)).abs()
    return levels[d.argmin(dim=1)].to(r.device).reshape(r.shape)

def q1b_scalar(r):
    s = r.abs().mean().item()
    return torch.zeros_like(r) if s < 1e-12 else torch.sign(r) * s

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

def run_experiment(data_dir, device, ext, ghd, tcp, tcpi, qtf, cbs):
    results = {}
    layer_idx = 10
    gate_file = data_dir / f"layer{layer_idx}_all_gate_proj.pt"
    if not gate_file.exists(): return results
    
    all_experts = torch.load(gate_file, map_location="cpu")
    n_experts = min(2, all_experts.shape[0])
    k, n = all_experts.shape[1], all_experts.shape[2]
    n_tiles = (k // 16) * (n // 16)
    n_weights = k * n
    print(f"  {n_experts} experts, {n_tiles} tiles, {n_weights} weights", flush=True)
    
    pareto = []
    
    for ei in range(n_experts):
        w = all_experts[ei].to(device)
        w_reg, _, _ = regularize(w, device, ghd, cbs)
        del w
        
        # Compute all 4 tiers
        qk3, tmse_k3 = quantize_tilewise_with_mse(w_reg, 3, device, tcp, tcpi, qtf)
        qk4, tmse_k4 = quantize_tilewise_with_mse(w_reg, 4, device, tcp, tcpi, qtf)
        r4 = w_reg - qk4; del qk4
        lloyd = q2b_lloyd(r4)
        recon_k5 = (w_reg - r4) + lloyd
        r5 = w_reg - recon_k5
        scalar_1bit = q1b_scalar(r5)
        recon_k6 = recon_k5 + scalar_1bit
        
        tiles_n_k = k // 16; tiles_n_n = n // 16
        tmse_k5 = torch.zeros_like(tmse_k4)
        tmse_k6 = torch.zeros_like(tmse_k4)
        for tik in range(tiles_n_k):
            for tin in range(tiles_n_n):
                rs, re = tik * 16, (tik + 1) * 16
                cs, ce = tin * 16, (tin + 1) * 16
                orig = w_reg[rs:re, cs:ce]
                tmse_k5[tik, tin] = (orig - recon_k5[rs:re, cs:ce]).pow(2).mean()
                tmse_k6[tik, tin] = (orig - recon_k6[rs:re, cs:ce]).pow(2).mean()
        
        # Build progressive upgrade bitstream
        # Each upgrade is 1 bit: which tile to upgrade next
        # But we need to store WHICH tile, not just "upgrade next"
        # The bitstream is: ordered list of tile indices (sorted by benefit)
        # At runtime, read first N entries to know which tiles to upgrade
        
        # Actually, the simplest progressive encoding:
        # 1. Sort all tiles by "upgrade priority" (benefit per bit)
        # 2. Store the sorted order (permutation) — this is the "bitstream"
        # 3. At runtime, first (target_bpw - 3) * n_tiles tiles are upgraded
        
        # The permutation costs log2(n_tiles) bits per tile = ~16 bits per tile
        # This is worse than 2-bit bitmap. So the progressive approach doesn't save bitmap storage.
        
        # BUT: if we store the PERMUTATION once (at encode time), and the 
        # target_bpw is a single number at runtime, then:
        # - Encode: store permutation (16 bits/tile) + all tier reconstructions
        # - Runtime: load first (B-3)*N tiles from permutation, upgrade them
        # - Bitmap cost: 0 (just the permutation, amortized across all bpw targets)
        
        # Compare:
        # - 2-bit bitmap: 2 bits/tile = 0.0078 bpw (per bpw target)
        # - Permutation: 16 bits/tile = 0.0625 bpw (once, shared across all targets)
        # - For a single target: 2-bit bitmap wins
        # - For many targets: permutation wins (amortized)
        
        # Let's measure: what's the actual cost?
        
        # The permutation itself can be stored as:
        # - Full permutation: log2(n_tiles) * n_tiles bits
        # - Sort order: just store the sort key (benefit) per tile
        #   = 32-bit float * n_tiles = 32 * n_tiles bits
        #   But this is just for encoding; at runtime, we pre-sort
        
        # Simplest: store the permutation as 16-bit indices
        perm_cost_bpw = 16 * n_tiles / n_weights  # bits per weight
        bitmap_cost_bpw = 2 * n_tiles / n_weights
        
        print(f"\n  Expert {ei}:", flush=True)
        print(f"    Permutation cost: {perm_cost_bpw:.6f} bpw (stored once, shared)", flush=True)
        print(f"    2-bit bitmap cost: {bitmap_cost_bpw:.6f} bpw (per target)", flush=True)
        print(f"    Break-even: {perm_cost_bpw / bitmap_cost_bpw:.1f} targets", flush=True)
        
        # For a SINGLE target, the 2-bit bitmap is cheaper.
        # For multiple targets (fungible), the permutation is cheaper if >8 targets.
        
        # But actually, the most efficient approach for fungibility is:
        # Store the SORTED BENEFIT per tile (32-bit float = 0.125 bpw)
        # At runtime, threshold the benefit to select which tiles to upgrade
        # This is the "progressive" approach: one stored value per tile,
        # threshold = target_bpw
        
        # Even better: store a QUANTIZED benefit (e.g., 8-bit) per tile
        # = 8/256 = 0.031 bpw overhead (once, shared across all targets)
        
        benefit_cost_8bit = 8 * n_tiles / n_weights
        benefit_cost_4bit = 4 * n_tiles / n_weights
        benefit_cost_2bit = 2 * n_tiles / n_weights
        
        print(f"    8-bit benefit: {benefit_cost_8bit:.6f} bpw (shared)", flush=True)
        print(f"    4-bit benefit: {benefit_cost_4bit:.6f} bpw (shared)", flush=True)
        print(f"    2-bit benefit: {benefit_cost_2bit:.6f} bpw (shared)", flush=True)
        
        # Now let's test: does quantized benefit (4-bit) give good tier selection?
        print(f"\n    Testing quantized benefit tier selection:", flush=True)
        
        # Compute K3→K4 improvement per tile
        imp34 = (tmse_k3 - tmse_k4).flatten()
        imp45 = (tmse_k4 - tmse_k5).flatten()
        imp56 = (tmse_k5 - tmse_k6).flatten()
        
        # Full-precision sort (ground truth)
        all_upgrades = []
        for i in range(n_tiles):
            all_upgrades.append((imp34[i].item(), i, 3, 4))
            all_upgrades.append((imp45[i].item(), i, 4, 5))
            all_upgrades.append((imp56[i].item(), i, 5, 6))
        all_upgrades.sort(key=lambda x: -x[0])
        
        # Quantized benefit (4-bit): quantize to 16 levels
        all_benefits = [u[0] for u in all_upgrades]
        min_b, max_b = min(all_benefits), max(all_benefits)
        n_levels = 16
        quantized_upgrades = []
        for benefit, tile_idx, from_t, to_t in all_upgrades:
            # Quantize benefit to 4-bit
            if max_b > min_b:
                q_benefit = round((benefit - min_b) / (max_b - min_b) * (n_levels - 1)) / (n_levels - 1) * (max_b - min_b) + min_b
            else:
                q_benefit = benefit
            quantized_upgrades.append((q_benefit, tile_idx, from_t, to_t))
        # Re-sort by quantized benefit (ties broken by original order)
        quantized_upgrades.sort(key=lambda x: (-x[0],))
        
        for target_bpw in [3.5, 4.0, 4.5, 5.0, 5.5]:
            # Full-precision tier assignment
            tier_full = [3] * n_tiles
            current_bits = 3.0 * n_tiles
            target_bits = target_bpw * n_tiles
            for benefit, tile_idx, from_t, to_t in all_upgrades:
                if current_bits + 1 > target_bits: continue
                if tier_full[tile_idx] != from_t: continue
                if benefit <= 0: continue
                tier_full[tile_idx] = to_t
                current_bits += 1
            
            # Quantized tier assignment
            tier_quant = [3] * n_tiles
            current_bits_q = 3.0 * n_tiles
            for benefit, tile_idx, from_t, to_t in quantized_upgrades:
                if current_bits_q + 1 > target_bits: continue
                if tier_quant[tile_idx] != from_t: continue
                if benefit <= 0: continue
                tier_quant[tile_idx] = to_t
                current_bits_q += 1
            
            # Compute MSE for both
            result_full = qk3.clone()
            result_quant = qk3.clone()
            qk4_recon = w_reg - r4
            for i in range(n_tiles):
                tik = i // tiles_n_n; tin = i % tiles_n_n
                rs, re = tik * 16, (tik + 1) * 16
                cs, ce = tin * 16, (tin + 1) * 16
                if tier_full[i] == 4: result_full[rs:re, cs:ce] = qk4_recon[rs:re, cs:ce]
                elif tier_full[i] == 5: result_full[rs:re, cs:ce] = recon_k5[rs:re, cs:ce]
                elif tier_full[i] == 6: result_full[rs:re, cs:ce] = recon_k6[rs:re, cs:ce]
                if tier_quant[i] == 4: result_quant[rs:re, cs:ce] = qk4_recon[rs:re, cs:ce]
                elif tier_quant[i] == 5: result_quant[rs:re, cs:ce] = recon_k5[rs:re, cs:ce]
                elif tier_quant[i] == 6: result_quant[rs:re, cs:ce] = recon_k6[rs:re, cs:ce]
            
            mse_full = (w_reg - result_full).pow(2).mean().item()
            mse_quant = (w_reg - result_quant).pow(2).mean().item()
            loss_pct = (mse_quant - mse_full) / mse_full * 100
            
            # Count tier differences
            diffs = sum(1 for i in range(n_tiles) if tier_full[i] != tier_quant[i])
            
            print(f"      {target_bpw:.1f}bpw: full={mse_full:.6e} quant4b={mse_quant:.6e} "
                  f"loss={loss_pct:+.2f}% diffs={diffs}", flush=True)
        
        del w_reg, qk3, r4, lloyd, recon_k5, r5, scalar_1bit, recon_k6
        del tmse_k3, tmse_k4, tmse_k5, tmse_k6
        torch.cuda.empty_cache()
    
    return {"analysis": "completed"}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--data-dir", default="/tmp/poc_residual/glm52_data")
    ap.add_argument("--out", default="/tmp/poc_residual/results_v18.json")
    args = ap.parse_args()
    dev = torch.device(args.device)
    print(f"Device: {dev}  GPU: {torch.cuda.get_device_name(0)}", flush=True)
    ext, ghd, tcp, tcpi, qtf, cbs = _bootstrap()
    results = run_experiment(Path(args.data_dir), dev, ext, ghd, tcp, tcpi, qtf, cbs)
    Path(args.out).write_text(json.dumps(results, indent=2, default=str))
    print(f"\nResults saved to {args.out}", flush=True)

if __name__ == "__main__":
    main()
