#!/usr/bin/env python3
"""PoC v10: AlphaQ-style calibration-free per-expert allocation using PL_Alpha_Hill.

AlphaQ principle: experts with heavier-tailed weight spectra (smaller α) are
more important and should get more bits. This is calibration-free — just
needs the weight matrices.

Tests:
  1. Compute PL_Alpha_Hill for each GLM-5.2 expert
  2. Check if there's meaningful variation across 70 experts
  3. Allocate tile-level K3/K4/K5 budget per expert based on α
  4. Compare vs uniform allocation
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
# AlphaQ: PL_Alpha_Hill computation
# ---------------------------------------------------------------------------

def compute_pl_alpha_hill(W, device, n_subsamples=10, subsample_ratio=0.5):
    """Compute PL_Alpha_Hill metric for a weight matrix.
    Uses FARMS (Fixed-Aspect-Ratio Matrix Subsampling) for robustness.
    Smaller α = heavier tail = more important.
    """
    m, n = W.shape
    alphas = []
    for _ in range(n_subsamples):
        # FARMS: subsample with fixed aspect ratio
        sub_m = max(2, int(m * subsample_ratio))
        sub_n = max(2, int(n * subsample_ratio))
        if sub_m >= m or sub_n >= n:
            sub_W = W
        else:
            row_idx = torch.randperm(m)[:sub_m]
            col_idx = torch.randperm(n)[:sub_n]
            sub_W = W[row_idx][:, col_idx]
        
        # Correlation matrix: X = W^T W
        X = (sub_W.T @ sub_W).to(device)
        
        # Eigenvalues
        try:
            eigvals = torch.linalg.eigvalsh(X)
        except:
            continue
        eigvals = eigvals[eigvals > 1e-10].cpu().numpy()
        if len(eigvals) < 10:
            continue
        
        # Sort descending
        eigvals = np.sort(eigvals)[::-1]
        
        # Fix-finger method: find peak of ESD histogram
        n_bins = min(50, len(eigvals) // 4)
        if n_bins < 4:
            continue
        hist, bin_edges = np.histogram(eigvals, bins=n_bins)
        peak_bin = np.argmax(hist)
        lambda_min = bin_edges[peak_bin]
        
        # Tail eigenvalues
        tail_eigvals = eigvals[eigvals >= lambda_min]
        if len(tail_eigvals) < 5:
            continue
        
        # Hill estimator
        k = len(tail_eigvals) - 1
        if k < 2:
            continue
        ratios = tail_eigvals[:-1] / tail_eigvals[-1]  # λ_{n-i+1} / λ_{n-k}
        log_ratios = np.log(ratios[:k])
        if np.mean(log_ratios) < 1e-10:
            continue
        alpha = 1.0 + 1.0 / np.mean(log_ratios)
        if alpha > 0:
            alphas.append(alpha)
    
    if not alphas:
        return float('inf')  # no heavy tail detected
    return np.median(alphas)

def alphaq_allocation(alphas, variances, total_budget_bits, n_weights_per_expert, 
                      available_bits=[3, 4, 5], gamma=None):
    """AlphaQ-style bit allocation.
    
    Minimizes sum of η_{l,b} = (α̃/α_l)^γ * Var(W_l) * 2^{-2b}
    subject to sum(b_l * n_l) <= total_budget_bits
    
    Simplified: greedily assign bits to minimize total scaled noise.
    """
    n_experts = len(alphas)
    alpha_median = np.median(alphas)
    if gamma is None:
        alpha_min = min(alphas)
        alpha_max = max(alphas)
        alpha_var = np.var(alphas)
        if alpha_var > 0 and alpha_min > 0:
            gamma = alpha_min * (alpha_max - alpha_min) / alpha_var
        else:
            gamma = 1.0
    
    # Compute scaled noise for each expert at each bit-width
    # η_{l,b} = (α̃/α_l)^γ * Var(W_l) * 2^{-2b}
    noises = {}
    for l in range(n_experts):
        for b in available_bits:
            if alphas[l] > 0:
                scale = (alpha_median / alphas[l]) ** gamma
            else:
                scale = 1.0
            noises[(l, b)] = scale * variances[l] * (2 ** (-2 * b))
    
    # Greedy allocation: start all at min bits, upgrade one at a time
    bits = [min(available_bits)] * n_experts
    current_budget = sum(bits) * n_weights_per_expert
    
    while current_budget < total_budget_bits:
        # Find expert that benefits most from upgrade
        best_gain = 0
        best_expert = -1
        best_new_bit = 0
        for l in range(n_experts):
            current_b = bits[l]
            # Try upgrading to next available bit
            for new_b in available_bits:
                if new_b <= current_b:
                    continue
                gain = noises[(l, current_b)] - noises[(l, new_b)]
                cost = (new_b - current_b) * n_weights_per_expert
                if current_budget + cost > total_budget_bits:
                    continue
                if gain > best_gain:
                    best_gain = gain
                    best_expert = l
                    best_new_bit = new_b
        
        if best_expert == -1:
            break
        old_b = bits[best_expert]
        bits[best_expert] = best_new_bit
        current_budget += (best_new_bit - old_b) * n_weights_per_expert
    
    return bits

def run_experiment(data_dir, device, ext, ghd, tcp, tcpi, qtf, cbs, layer_indices=[10, 40], max_experts=20):
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
        print(f"  {n_experts} experts, shape=({k},{n})", flush=True)
        
        # ================================================================
        # Step 1: Compute PL_Alpha_Hill for each expert
        # ================================================================
        print(f"\n  Computing PL_Alpha_Hill...", flush=True)
        alphas = []
        variances = []
        for ei in range(n_experts):
            W = all_experts[ei]
            alpha = compute_pl_alpha_hill(W, device)
            var = W.var().item()
            alphas.append(alpha)
            variances.append(var)
        
        alphas = np.array(alphas)
        variances = np.array(variances)
        
        print(f"  PL_Alpha_Hill: min={alphas.min():.4f} max={alphas.max():.4f} "
              f"median={np.median(alphas):.4f} CV={alphas.std()/alphas.mean():.4f}", flush=True)
        print(f"  Variances: min={variances.min():.6e} max={variances.max():.6e} "
              f"CV={variances.std()/variances.mean():.4f}", flush=True)
        
        # Show top/bottom experts by alpha
        sorted_idx = np.argsort(alphas)
        print(f"\n  Heaviest-tailed (most important, smallest α):", flush=True)
        for i in sorted_idx[:5]:
            print(f"    Expert {i}: α={alphas[i]:.4f} Var={variances[i]:.6e}", flush=True)
        print(f"  Lightest-tailed (least important, largest α):", flush=True)
        for i in sorted_idx[-5:]:
            print(f"    Expert {i}: α={alphas[i]:.4f} Var={variances[i]:.6e}", flush=True)
        
        # ================================================================
        # Step 2: Baselines — uniform K3, K4, K5
        # ================================================================
        print(f"\n  Computing baselines...", flush=True)
        k3_mses = []; k4_mses = []; k5_mses = []
        for ei in range(n_experts):
            w = all_experts[ei].to(device)
            w_reg, _, _ = regularize(w, device, ghd, cbs)
            qk3 = quantize_trellis(w_reg, 3, device, tcp, tcpi, qtf)
            qk4 = quantize_trellis(w_reg, 4, device, tcp, tcpi, qtf)
            r4 = w_reg - qk4
            lloyd = q2b_lloyd(r4)
            k3_mses.append((w_reg - qk3).pow(2).mean().item())
            k4_mses.append((w_reg - qk4).pow(2).mean().item())
            k5_mses.append((w_reg - qk4 - lloyd).pow(2).mean().item())
            del w, w_reg, qk3, qk4, r4, lloyd
            torch.cuda.empty_cache()
        
        k3_avg = sum(k3_mses) / n_experts
        k4_avg = sum(k4_mses) / n_experts
        k5_avg = sum(k5_mses) / n_experts
        print(f"  K3={k3_avg:.6e} K4={k4_avg:.6e} K5={k5_avg:.6e}", flush=True)
        
        # ================================================================
        # Step 3: AlphaQ allocation vs uniform allocation
        # ================================================================
        layer_results = {}
        layer_results["K3_uniform"] = {"mse": k3_avg, "bits": 3.0}
        layer_results["K4_uniform"] = {"mse": k4_avg, "bits": 4.0}
        layer_results["K5_uniform"] = {"mse": k5_avg, "bits": 5.0}
        
        for target_bpw in [3.5, 4.0, 4.5]:
            total_budget = int(target_bpw * n_experts * n_weights)
            
            # Uniform: all experts at floor(target_bpw), some at ceil
            uniform_bits = [int(target_bpw)] * n_experts
            # Simple: fraction at K4, rest at K3
            frac_k4 = target_bpw - 3.0
            n_k4 = int(frac_k4 * n_experts)
            uniform_bits = [3] * n_experts
            for i in range(n_k4):
                uniform_bits[i] = 4
            
            # AlphaQ allocation
            alphaq_bits = alphaq_allocation(alphas, variances, total_budget, n_weights,
                                            available_bits=[3, 4, 5])
            
            print(f"\n  Target {target_bpw} bpw:", flush=True)
            print(f"    Uniform: {uniform_bits} (avg={sum(uniform_bits)/n_experts:.2f})", flush=True)
            print(f"    AlphaQ:  {alphaq_bits} (avg={sum(alphaq_bits)/n_experts:.2f})", flush=True)
            
            # Measure uniform
            mses_uniform = []
            for ei in range(n_experts):
                w = all_experts[ei].to(device)
                w_reg, _, _ = regularize(w, device, ghd, cbs)
                K = uniform_bits[ei]
                qk = quantize_trellis(w_reg, K, device, tcp, tcpi, qtf)
                mses_uniform.append((w_reg - qk).pow(2).mean().item())
                del w, w_reg, qk
                torch.cuda.empty_cache()
            avg_mse_uniform = sum(mses_uniform) / n_experts
            avg_bits_uniform = sum(uniform_bits) / n_experts
            label = f"uniform_{target_bpw}bpw"
            layer_results[label] = {"mse": avg_mse_uniform, "bits": avg_bits_uniform}
            print(f"    Uniform MSE: {avg_mse_uniform:.6e}", flush=True)
            
            # Measure AlphaQ
            mses_alphaq = []
            for ei in range(n_experts):
                w = all_experts[ei].to(device)
                w_reg, _, _ = regularize(w, device, ghd, cbs)
                K = alphaq_bits[ei]
                qk = quantize_trellis(w_reg, K, device, tcp, tcpi, qtf)
                mses_alphaq.append((w_reg - qk).pow(2).mean().item())
                del w, w_reg, qk
                torch.cuda.empty_cache()
            avg_mse_alphaq = sum(mses_alphaq) / n_experts
            avg_bits_alphaq = sum(alphaq_bits) / n_experts
            label = f"alphaq_{target_bpw}bpw"
            layer_results[label] = {"mse": avg_mse_alphaq, "bits": avg_bits_alphaq}
            print(f"    AlphaQ MSE:  {avg_mse_alphaq:.6e}", flush=True)
            
            # Improvement
            improvement = (avg_mse_uniform - avg_mse_alphaq) / avg_mse_uniform * 100
            print(f"    AlphaQ improvement: {improvement:+.2f}%", flush=True)
        
        results[f"layer{layer_idx}"] = {
            "n_experts": n_experts,
            "alphas": alphas.tolist(),
            "variances": variances.tolist(),
            "k3_avg": k3_avg, "k4_avg": k4_avg, "k5_avg": k5_avg,
            "methods": layer_results,
            "alpha_stats": {
                "min": float(alphas.min()), "max": float(alphas.max()),
                "median": float(np.median(alphas)), "cv": float(alphas.std()/alphas.mean())
            }
        }
    
    return results

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--data-dir", default="/tmp/poc_residual/glm52_data")
    ap.add_argument("--out", default="/tmp/poc_residual/results_v10.json")
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
        print(f"\n{layer_key}:", flush=True)
        print(f"  Alpha stats: min={r['alpha_stats']['min']:.4f} max={r['alpha_stats']['max']:.4f} "
              f"CV={r['alpha_stats']['cv']:.4f}", flush=True)
        methods = r["methods"]
        for label in sorted(methods.keys(), key=lambda x: methods[x]["mse"]):
            m = methods[label]
            print(f"    {label:30s}: MSE={m['mse']:.6e}  bits={m['bits']:.2f}", flush=True)
    
    Path(args.out).write_text(json.dumps(results, indent=2, default=str))
    print(f"\nResults saved to {args.out}", flush=True)

if __name__ == "__main__":
    main()
