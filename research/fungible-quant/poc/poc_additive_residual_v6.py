#!/usr/bin/env python3
"""PoC v6: Continuously variable per-expert sparse quantization for GLM-5.2.

Core idea: K3 base + per-expert sparse correction, where the sparse budget
is allocated by water-filling based on quantization damage and disambiguation.

Key innovations:
  1. K3 base for all experts (uniform quality floor)
  2. Sparse fp16 correction on the K3 residual, allocated per-expert
  3. Water-filling allocation: give more sparse budget to experts with higher damage
  4. Disambiguation weight: experts that differ more from the layer mean get more budget
  5. Hierarchical steps: K3 +% +% +% (each % is a sparse correction on the remaining residual)
  6. ICQuant gap-index coding for efficient sparse storage
  7. Tile-sparse: corrections on 2D tiles (16x16, 32x32) for runtime efficiency

Tests on real GLM-5.2 weights: 70 experts from layer 10 and layer 40.
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
    levels = torch.tensor([-1.5104, -0.4528, 0.4528, 1.5104], device=r.device) * sigma
    flat = r.flatten().unsqueeze(1)
    d = torch.cdist(flat, levels.unsqueeze(1))
    return levels[d.argmin(dim=1)].reshape(r.shape)

# ---------------------------------------------------------------------------
# Sparse correction with ICQuant gap-index coding
# ---------------------------------------------------------------------------

def sparse_correct(residual, top_pct):
    """Apply sparse fp16 correction on top-k% of residual by |magnitude|."""
    flat = residual.flatten()
    sigma = flat.std().item()
    if sigma < 1e-12: return torch.zeros_like(residual)
    k = max(1, int(flat.numel() * top_pct / 100))
    _, topk_idx = flat.abs().topk(k)
    result = torch.zeros_like(flat)
    result[topk_idx] = flat[topk_idx]
    return result.reshape(residual.shape)

def sparse_correct_bits(n_weights, top_pct, idx_bits_per_entry=8):
    """Bits per weight for sparse correction:
    - n_sparse = n_weights * top_pct/100 entries
    - Each entry: idx_bits (index) + 16 (fp16 value)
    - ICQuant gap coding: ~log2(100/top_pct) bits per index instead of log2(n_weights)
    """
    gamma = top_pct / 100
    if gamma < 1e-6: return 0
    # ICQuant gap index cost (Lemma 1 upper bound)
    b = max(4, math.ceil(math.log2(max(1/gamma, 1))) + 1)
    idx_cost_per_entry = b  # gap bits per index
    # Total: gamma * (b + 16) bits per weight
    return gamma * (b + 16)

# ---------------------------------------------------------------------------
# Water-filling allocation
# ---------------------------------------------------------------------------

def water_fill_allocation(damages, total_budget_bits, n_weights_per_expert, min_pct=0.0):
    """Allocate sparse budget across experts by water-filling.
    
    damages: list of per-expert MSE (higher = more damaged)
    total_budget_bits: total extra bits available across all experts
    n_weights_per_expert: number of weights per expert
    Returns: list of top_pct allocations per expert.
    """
    n_experts = len(damages)
    # Normalize damages to [0, 1]
    max_d = max(damages) if max(damages) > 0 else 1
    norm_damages = [d / max_d for d in damages]
    
    # Budget per expert in bits/weight
    budget_bpw = total_budget_bits / (n_experts * n_weights_per_expert)
    
    # Water-filling: give budget proportional to damage, but with a floor
    # allocation_pct = budget_bpw / 16 * damage_weight (since sparse costs ~16 bits per entry)
    # We want: sum(sparse_correct_bits(n, pct_e)) = total_budget_bits
    # sparse_correct_bits ≈ pct/100 * (b + 16) ≈ pct/100 * 22
    # So sum(pct_e/100 * 22 * n_weights) = total_budget
    # => avg_pct = total_budget / (n_experts * n_weights * 22) * 100
    
    avg_pct = budget_bpw / 22 * 100  # average sparsity percentage
    
    # Allocate proportionally to damage
    total_damage = sum(norm_damages) if sum(norm_damages) > 0 else n_experts
    allocations = [max(min_pct, avg_pct * nd / (total_damage / n_experts)) for nd in norm_damages]
    
    # Scale to fit budget exactly
    current_total = sum(sparse_correct_bits(n_weights_per_expert, a) for a in allocations) * n_experts
    if current_total > 0:
        scale = total_budget_bits / (current_total / n_experts * n_experts)
        # Actually compute properly
        actual_total = sum(sparse_correct_bits(n_weights_per_expert, a) * n_weights_per_expert for a in allocations)
        if actual_total > 0:
            scale = total_budget_bits / actual_total
            allocations = [a * scale for a in allocations]
    
    return allocations

def disambiguation_weight(expert_weights, layer_mean_weights):
    """How much an expert differs from the layer mean.
    Higher = more unique = needs more budget for disambiguation.
    """
    diff = (expert_weights - layer_mean_weights).norm() / max(expert_weights.norm(), 1e-8)
    return diff.item()

# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_experiment(data_dir, device, ext, ghd, tcp, tcpi, qtf, cbs, layer_indices=[10, 40]):
    """Run the full experiment across multiple experts and layers."""
    
    results = {}
    
    for layer_idx in layer_indices:
        print(f"\n{'='*70}", flush=True)
        print(f"Layer {layer_idx}", flush=True)
        print(f"{'='*70}", flush=True)
        
        # Load all experts for this layer (gate_proj)
        gate_file = data_dir / f"layer{layer_idx}_all_gate_proj.pt"
        if not gate_file.exists():
            print(f"  SKIP: {gate_file} not found", flush=True)
            continue
        
        all_experts = torch.load(gate_file, map_location="cpu")  # (n_experts, 2048, 6144)
        n_experts, k, n = all_experts.shape
        n_weights = k * n
        print(f"  {n_experts} experts, shape=({k},{n}), {n_weights} weights/expert", flush=True)
        
        # Layer mean (for disambiguation)
        layer_mean = all_experts.mean(dim=0)
        
        # Quantize each expert at K3 and K4
        print(f"  Quantizing {n_experts} experts at K3 and K4...", flush=True)
        k3_mses = []; k4_mses = []; disambig = []
        k3_residuals = []; k4_residuals = []
        
        for ei in range(n_experts):
            w = all_experts[ei].to(device)
            w_reg, su, sv = regularize(w, device, ghd, cbs)
            qk3 = quantize_trellis(w_reg, 3, device, tcp, tcpi, qtf)
            qk4 = quantize_trellis(w_reg, 4, device, tcp, tcpi, qtf)
            r3 = w_reg - qk3  # K3 residual
            r4 = w_reg - qk4  # K4 residual
            k3_mses.append((w_reg - qk3).pow(2).mean().item())
            k4_mses.append((w_reg - qk4).pow(2).mean().item())
            k3_residuals.append(r3.cpu())
            k4_residuals.append(r4.cpu())
            disambig.append(disambiguation_weight(all_experts[ei], layer_mean))
            if ei % 10 == 0:
                print(f"    Expert {ei}: K3 MSE={k3_mses[-1]:.6e}, K4 MSE={k4_mses[-1]:.6e}, "
                      f"disambig={disambig[-1]:.4f}", flush=True)
            del w, w_reg, qk3, qk4, r3, r4
            torch.cuda.empty_cache()
        
        torch.cuda.synchronize()
        
        # Summary
        k3_avg = sum(k3_mses) / n_experts
        k4_avg = sum(k4_mses) / n_experts
        print(f"\n  K3 avg MSE: {k3_avg:.6e}, K4 avg MSE: {k4_avg:.6e}", flush=True)
        print(f"  K3 MSE range: [{min(k3_mses):.6e}, {max(k3_mses):.6e}]", flush=True)
        print(f"  K4 MSE range: [{min(k4_mses):.6e}, {max(k4_mses):.6e}]", flush=True)
        print(f"  Disambiguation range: [{min(disambig):.4f}, {max(disambig):.4f}]", flush=True)
        
        # Sort experts by damage (K3 MSE)
        sorted_indices = sorted(range(n_experts), key=lambda i: k3_mses[i], reverse=True)
        print(f"\n  Most damaged experts (K3): {[f'E{sorted_indices[i]}:{k3_mses[sorted_indices[i]]:.4e}' for i in range(5)]}", flush=True)
        print(f"  Least damaged experts (K3): {[f'E{sorted_indices[-i-1]}:{k3_mses[sorted_indices[-i-1]]:.4e}' for i in range(5)]}", flush=True)
        
        # ================================================================
        # Experiment 1: Fixed sparse% for all experts (baseline)
        # ================================================================
        print(f"\n  --- Experiment 1: Fixed sparse% (uniform) ---", flush=True)
        layer_results = {}
        
        for extra_bpw in [0.1, 0.25, 0.5, 1.0, 2.0]:
            # Convert extra_bpw to sparse%: extra_bpw = pct/100 * 22
            pct = extra_bpw / 22 * 100
            mses_after = []
            actual_bits = []
            for ei in range(n_experts):
                r3 = k3_residuals[ei].to(device)
                correction = sparse_correct(r3, pct)
                # MSE after correction: original K3 error minus what the correction captured
                # correction ≈ the top-k% of the residual, so the remaining error is:
                residual_after = r3 - correction
                # Total reconstruction: K3 + correction, error = w_reg - (K3 + correction) = r3 - correction
                mse_after = residual_after.pow(2).mean().item()
                mses_after.append(mse_after)
                bits = sparse_correct_bits(n_weights, pct)
                actual_bits.append(bits)
                del r3, correction
                torch.cuda.empty_cache()
            
            avg_mse = sum(mses_after) / n_experts
            avg_bits = 3 + sum(actual_bits) / n_experts
            gap = (k3_avg - avg_mse) / (k3_avg - k4_avg) if k3_avg > k4_avg else 0
            label = f"K3+uniform_{extra_bpw}bpw"
            layer_results[label] = {"avg_mse": avg_mse, "avg_bits": avg_bits, "gap_to_K4": gap}
            print(f"    {label}: MSE={avg_mse:.6e}  bits={avg_bits:.3f}  gap={gap:.1%}", flush=True)
        
        # ================================================================
        # Experiment 2: Water-filling allocation (damage-proportional)
        # ================================================================
        print(f"\n  --- Experiment 2: Water-filling (damage-proportional) ---", flush=True)
        
        for total_extra_bpw in [0.25, 0.5, 1.0, 2.0]:
            total_budget = total_extra_bpw * n_experts * n_weights
            allocs = water_fill_allocation(k3_mses, total_budget, n_weights, min_pct=0.01)
            
            mses_after = []
            actual_bits = []
            for ei in range(n_experts):
                r3 = k3_residuals[ei].to(device)
                correction = sparse_correct(r3, allocs[ei])
                residual_after = r3 - correction
                mse_after = residual_after.pow(2).mean().item()
                mses_after.append(mse_after)
                bits = sparse_correct_bits(n_weights, allocs[ei])
                actual_bits.append(bits)
                del r3, correction
                torch.cuda.empty_cache()
            
            avg_mse = sum(mses_after) / n_experts
            avg_bits = 3 + sum(actual_bits) / n_experts
            gap = (k3_avg - avg_mse) / (k3_avg - k4_avg) if k3_avg > k4_avg else 0
            label = f"K3+waterfill_{total_extra_bpw}bpw"
            layer_results[label] = {"avg_mse": avg_mse, "avg_bits": avg_bits, "gap_to_K4": gap,
                                     "alloc_range": [min(allocs), max(allocs)]}
            print(f"    {label}: MSE={avg_mse:.6e}  bits={avg_bits:.3f}  gap={gap:.1%}  "
                  f"alloc=[{min(allocs):.2f}%,{max(allocs):.2f}%]", flush=True)
        
        # ================================================================
        # Experiment 3: Disambiguation-weighted allocation
        # ================================================================
        print(f"\n  --- Experiment 3: Disambiguation-weighted ---", flush=True)
        
        for total_extra_bpw in [0.25, 0.5, 1.0, 2.0]:
            total_budget = total_extra_bpw * n_experts * n_weights
            # Combined: damage * disambiguation
            combined = [k3_mses[i] * disambig[i] for i in range(n_experts)]
            allocs = water_fill_allocation(combined, total_budget, n_weights, min_pct=0.01)
            
            mses_after = []
            actual_bits = []
            for ei in range(n_experts):
                r3 = k3_residuals[ei].to(device)
                correction = sparse_correct(r3, allocs[ei])
                residual_after = r3 - correction
                mse_after = residual_after.pow(2).mean().item()
                mses_after.append(mse_after)
                bits = sparse_correct_bits(n_weights, allocs[ei])
                actual_bits.append(bits)
                del r3, correction
                torch.cuda.empty_cache()
            
            avg_mse = sum(mses_after) / n_experts
            avg_bits = 3 + sum(actual_bits) / n_experts
            gap = (k3_avg - avg_mse) / (k3_avg - k4_avg) if k3_avg > k4_avg else 0
            label = f"K3+disambig_{total_extra_bpw}bpw"
            layer_results[label] = {"avg_mse": avg_mse, "avg_bits": avg_bits, "gap_to_K4": gap,
                                    "alloc_range": [min(allocs), max(allocs)]}
            print(f"    {label}: MSE={avg_mse:.6e}  bits={avg_bits:.3f}  gap={gap:.1%}  "
                  f"alloc=[{min(allocs):.2f}%,{max(allocs):.2f}%]", flush=True)
        
        # ================================================================
        # Experiment 4: Hierarchical K3 +% +% +% (multi-step sparse)
        # ================================================================
        print(f"\n  --- Experiment 4: Hierarchical K3+%+%+% (multi-step) ---", flush=True)
        
        for step_budgets in [(0.25, 0.25, 0.25), (0.5, 0.5, 0.5), (0.25, 0.5, 1.0), (1.0, 1.0, 1.0)]:
            # Each step applies sparse correction on the remaining residual
            mses_after = []
            total_bits = []
            for ei in range(n_experts):
                r = k3_residuals[ei].to(device)
                total_correction = torch.zeros_like(r)
                total_bits_ei = 0
                for step_pct_bpw in step_budgets:
                    pct = step_pct_bpw / 22 * 100
                    correction = sparse_correct(r, pct)
                    total_correction += correction
                    r = r - correction
                    total_bits_ei += sparse_correct_bits(n_weights, pct)
                residual_after = k3_residuals[ei].to(device) - total_correction
                mse_after = residual_after.pow(2).mean().item()
                mses_after.append(mse_after)
                total_bits.append(total_bits_ei)
                del r, total_correction
                torch.cuda.empty_cache()
            
            avg_mse = sum(mses_after) / n_experts
            avg_bits = 3 + sum(total_bits) / n_experts
            gap = (k3_avg - avg_mse) / (k3_avg - k4_avg) if k3_avg > k4_avg else 0
            label = f"K3+steps{'_'.join(str(b) for b in step_budgets)}"
            layer_results[label] = {"avg_mse": avg_mse, "avg_bits": avg_bits, "gap_to_K4": gap}
            print(f"    {label}: MSE={avg_mse:.6e}  bits={avg_bits:.3f}  gap={gap:.1%}", flush=True)
        
        # ================================================================
        # Experiment 5: K3 + Lloyd-Max 2-bit (baseline K5 equiv)
        # ================================================================
        print(f"\n  --- Experiment 5: K3+2lloyd (baseline K5) ---", flush=True)
        mses_after = []
        for ei in range(n_experts):
            r3 = k3_residuals[ei].to(device)
            lloyd = q2b_lloyd(r3)
            residual_after = r3 - lloyd
            mses_after.append(residual_after.pow(2).mean().item())
            del r3, lloyd
            torch.cuda.empty_cache()
        
        avg_mse = sum(mses_after) / n_experts
        label = "K3+2lloyd"
        layer_results[label] = {"avg_mse": avg_mse, "avg_bits": 5.0,
                                 "gap_to_K4": (k3_avg - avg_mse) / (k3_avg - k4_avg) if k3_avg > k4_avg else 0}
        print(f"    {label}: MSE={avg_mse:.6e}  bits=5.000  gap={layer_results[label]['gap_to_K4']:.1%}", flush=True)
        
        # Store layer results
        results[f"layer{layer_idx}"] = {
            "k3_avg_mse": k3_avg, "k4_avg_mse": k4_avg,
            "k3_mses": k3_mses, "k4_mses": k4_mses,
            "disambig": disambig,
            "methods": layer_results,
            "n_experts": n_experts
        }
    
    return results

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--data-dir", default="/tmp/poc_residual/glm52_data")
    ap.add_argument("--out", default="/tmp/poc_residual/results_v6.json")
    args = ap.parse_args()
    dev = torch.device(args.device)
    print(f"Device: {dev}  GPU: {torch.cuda.get_device_name(0)}", flush=True)
    ext, ghd, tcp, tcpi, qtf, cbs = _bootstrap()
    
    results = run_experiment(Path(args.data_dir), dev, ext, ghd, tcp, tcpi, qtf, cbs)
    
    # Print aggregate summary
    print(f"\n{'='*70}\nSUMMARY\n{'='*70}", flush=True)
    for layer_key in sorted(results.keys()):
        r = results[layer_key]
        print(f"\n{layer_key}: K3={r['k3_avg_mse']:.4e} K4={r['k4_avg_mse']:.4e}", flush=True)
        print(f"  Methods (sorted by gap_to_K4):", flush=True)
        methods = r["methods"]
        for label in sorted(methods.keys(), key=lambda x: methods[x]["gap_to_K4"], reverse=True):
            m = methods[label]
            print(f"    {label:35s}: MSE={m['avg_mse']:.6e}  bits={m['avg_bits']:.3f}  "
                  f"gap={m['gap_to_K4']:.1%}", flush=True)
    
    Path(args.out).write_text(json.dumps(results, indent=2, default=str))
    print(f"\nResults saved to {args.out}", flush=True)

if __name__ == "__main__":
    main()
