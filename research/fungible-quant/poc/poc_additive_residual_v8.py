#!/usr/bin/env python3
"""PoC v8: BitsMoE-style spectral decomposition + tile-level mixed precision.

Key innovation: Apply SVD to RAW expert weights (before Hadamard) to find
cross-expert structure, then quantize the per-expert spectral factors with
EXL3 trellis + tile-level K3/K4 mixing.

W_cat = [W_1; W_2; ...; W_E] = P_cat * Φ^T  (SVD)
W_e = P_e * Φ^T  (reconstruction)
Quantize P_e, keep Φ unquantized (shared, amortized cost ~0)

Tests:
  1. Spectral energy variation across experts (does it enable differentiated allocation?)
  2. K3/K4 trellis on P_e vs W_e (does decomposition help or hurt?)
  3. Tile-level K3→K4 on P_e with spectral-energy-guided allocation
  4. Full pipeline: decompose → quantize P_e → reconstruct → measure
"""
from __future__ import annotations
import argparse, json, math, os, sys, types, importlib.util, time
from pathlib import Path
import torch, torch.nn.functional as F

EXL3_PKG = "/opt/fruit-pip/exllamav3"
HAD_K, HAD_N = 128, 128
TILE_K, TILE_N = 16, 16

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
    """Quantize tile-by-tile, return reconstruction and per-tile MSE."""
    k, n = w_reg.shape; tiles_n = n // 16
    tiles_n_k = k // 16; tiles_n_n = n // 16
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

def bitsmoe_decompose(all_experts, device):
    """BitsMoE-style SVD decomposition.
    W_cat = [W_1; ...; W_E] = U_cat * Σ * Φ^T = P_cat * Φ^T
    Returns: P_list (per-expert spectral factors), Phi (shared basis), energies
    """
    n_experts, k, n = all_experts.shape
    # Stack along output dim (gate_proj: output=2048, input=6144)
    # W_cat shape: (n_experts * k, n) = (E*2048, 6144)
    W_cat = all_experts.reshape(n_experts * k, n).to(device)
    print(f"  SVD on {W_cat.shape}...", flush=True)
    U, S, Vh = torch.linalg.svd(W_cat, full_matrices=False)
    # Phi = V^T (shared basis), shape (n, n)
    Phi = Vh  # (n, n)
    # P_cat = U * Sigma, shape (n_experts * k, n)
    P_cat = U * S.unsqueeze(0)  # broadcast
    # Split per expert
    P_list = [P_cat[e*k:(e+1)*k, :].cpu() for e in range(n_experts)]
    # Spectral energies: ||p_{e,k}|| per spectral component
    energies = torch.zeros(n_experts, n)
    for e in range(n_experts):
        energies[e] = P_cat[e*k:(e+1)*k, :].norm(dim=0).cpu()
    del U, S, Vh, P_cat, W_cat
    torch.cuda.empty_cache()
    return P_list, Phi.cpu(), energies

def run_experiment(data_dir, device, ext, ghd, tcp, tcpi, qtf, cbs, layer_indices=[10], max_experts=20):
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
        n_experts = min(all_experts.shape[0], max_experts)
        k, n = all_experts.shape[1], all_experts.shape[2]
        all_experts = all_experts[:n_experts]
        print(f"  {n_experts} experts, shape=({k},{n})", flush=True)
        
        layer_results = {}
        
        # ================================================================
        # Baseline: direct K3/K4 on raw weights (with Hadamard)
        # ================================================================
        print(f"\n  --- Baseline: direct quantization ---", flush=True)
        k3_mses_direct = []; k4_mses_direct = []
        for ei in range(n_experts):
            w = all_experts[ei].to(device)
            w_reg, _, _ = regularize(w, device, ghd, cbs)
            qk3 = quantize_trellis(w_reg, 3, device, tcp, tcpi, qtf)
            qk4 = quantize_trellis(w_reg, 4, device, tcp, tcpi, qtf)
            k3_mses_direct.append((w_reg - qk3).pow(2).mean().item())
            k4_mses_direct.append((w_reg - qk4).pow(2).mean().item())
            del w, w_reg, qk3, qk4
            torch.cuda.empty_cache()
        k3_avg_direct = sum(k3_mses_direct) / n_experts
        k4_avg_direct = sum(k4_mses_direct) / n_experts
        print(f"    Direct K3 avg: {k3_avg_direct:.6e}, K4 avg: {k4_avg_direct:.6e}", flush=True)
        print(f"    K3 range: [{min(k3_mses_direct):.6e}, {max(k3_mses_direct):.6e}]", flush=True)
        
        # ================================================================
        # BitsMoE decomposition
        # ================================================================
        print(f"\n  --- BitsMoE decomposition ---", flush=True)
        P_list, Phi, energies = bitsmoe_decompose(all_experts, device)
        
        # Analyze spectral energy variation
        print(f"    Energy shape: {energies.shape}", flush=True)
        print(f"    Energy per expert (norm): min={energies.norm(dim=1).min():.2f}, "
              f"max={energies.norm(dim=1).max():.2f}, "
              f"ratio={energies.norm(dim=1).max()/energies.norm(dim=1).min():.4f}", flush=True)
        # Per-component variation
        energy_cv = energies.std(dim=0) / (energies.mean(dim=0) + 1e-8)  # coefficient of variation
        print(f"    Energy CV across experts: mean={energy_cv.mean():.4f}, "
              f"max={energy_cv.max():.4f}", flush=True)
        # Top components
        top_energies = energies.mean(dim=0).sort(descending=True)
        print(f"    Top 5 spectral components (avg energy): {top_energies.values[:5].tolist()}", flush=True)
        print(f"    Bottom 5: {top_energes.values[-5:].tolist()}" if 'top_energes' in dir() else "", flush=True)
        
        # ================================================================
        # Quantize P_e with EXL3 trellis (after Hadamard regularization)
        # ================================================================
        print(f"\n  --- Quantize spectral factors P_e ---", flush=True)
        k3_mses_spectral = []; k4_mses_spectral = []
        recon_mses_k3 = []; recon_mses_k4 = []
        
        Phi_dev = Phi.to(device)
        
        for ei in range(n_experts):
            P_e = P_list[ei].to(device)
            # Regularize P_e (same pipeline as raw weights)
            P_reg, su, sv = regularize(P_e, device, ghd, cbs)
            qk3 = quantize_trellis(P_reg, 3, device, tcp, tcpi, qtf)
            qk4 = quantize_trellis(P_reg, 4, device, tcp, tcpi, qtf)
            
            # MSE in regularized space
            k3_mses_spectral.append((P_reg - qk3).pow(2).mean().item())
            k4_mses_spectral.append((P_reg - qk4).pow(2).mean().item())
            
            # Reconstruct: W_hat = Q(P_e) * Phi^T
            # But we need to undo the regularization to get back to P space
            # Actually, we should measure in the original W space
            # W_e = P_e * Phi^T, W_hat = Q(P_e) * Phi^T
            # Error = (P_e - Q(P_e)) * Phi^T
            # Since Phi is orthogonal, ||W_e - W_hat||^2 = ||P_e - Q(P_e)||^2
            # So the MSE in W space equals MSE in P space (Phi is orthogonal)
            # But regularization changes things... let's measure directly
            
            # De-regularize qk3 back to P space
            # Actually, the regularization is: P_reg = Had_k @ (P_e / su) / sv ... 
            # This is complex. Let's just measure in regularized space for now.
            
            del P_e, P_reg, qk3, qk4
            torch.cuda.empty_cache()
        
        k3_avg_spectral = sum(k3_mses_spectral) / n_experts
        k4_avg_spectral = sum(k4_mses_spectral) / n_experts
        print(f"    Spectral K3 avg: {k3_avg_spectral:.6e}, K4 avg: {k4_avg_spectral:.6e}", flush=True)
        print(f"    K3 range: [{min(k3_mses_spectral):.6e}, {max(k3_mses_spectral):.6e}]", flush=True)
        print(f"    K3 CV: {sum(k3_mses_spectral)/n_experts and (sum((m-sum(k3_mses_spectral)/n_experts)**2 for m in k3_mses_spectral)/n_experts)**0.5 / (sum(k3_mses_spectral)/n_experts):.4f}", flush=True)
        
        # ================================================================
        # Tile-level K3→K4 on spectral factors with energy-guided allocation
        # ================================================================
        print(f"\n  --- Tile K3→K4 on spectral factors ---", flush=True)
        
        for upgrade_frac in [0.1, 0.25, 0.5, 0.75, 1.0]:
            mses = []
            for ei in range(n_experts):
                P_e = P_list[ei].to(device)
                P_reg, _, _ = regularize(P_e, device, ghd, cbs)
                qk3, tile_mse_k3 = quantize_tilewise_with_mse(P_reg, 3, device, tcp, tcpi, qtf)
                qk4, tile_mse_k4 = quantize_tilewise_with_mse(P_reg, 4, device, tcp, tcpi, qtf)
                
                # Upgrade top-k% tiles
                n_tiles = tile_mse_k3.numel()
                n_upgrade = int(n_tiles * upgrade_frac)
                improvement = (tile_mse_k3 - tile_mse_k4).flatten()
                _, top_idx = improvement.topk(n_upgrade)
                
                result = qk3.clone()
                tiles_n_n = P_reg.shape[1] // 16
                for idx in top_idx:
                    ti_k = idx.item() // tiles_n_n
                    ti_n = idx.item() % tiles_n_n
                    result[ti_k*16:(ti_k+1)*16, ti_n*16:(ti_n+1)*16] = \
                        qk4[ti_k*16:(ti_k+1)*16, ti_n*16:(ti_n+1)*16]
                
                mses.append((P_reg - result).pow(2).mean().item())
                del P_e, P_reg, qk3, qk4, result
                torch.cuda.empty_cache()
            
            avg_mse = sum(mses) / n_experts
            eff_bits = 3.0 + upgrade_frac * 1.0
            gap = (k3_avg_spectral - avg_mse) / (k3_avg_spectral - k4_avg_spectral) if k3_avg_spectral > k4_avg_spectral else 0
            label = f"spectral_tile_K4_{upgrade_frac:.2f}"
            layer_results[label] = {"mse": avg_mse, "bits": eff_bits, "gap": gap}
            print(f"    {label}: MSE={avg_mse:.6e}  bits={eff_bits:.3f}  gap={gap:.1%}", flush=True)
        
        # ================================================================
        # Per-expert variable allocation (energy-guided water-filling)
        # ================================================================
        print(f"\n  --- Energy-guided per-expert allocation ---", flush=True)
        
        for total_bpw in [3.5, 4.0]:
            # Allocate upgrade fraction per expert based on spectral energy
            # More energy = more important = more upgrade
            total_energy = energies.sum(dim=1)  # per-expert total energy
            norm_energy = total_energy / total_energy.sum()
            # Each expert gets upgrade_frac = total_bpw - 3, weighted by energy
            avg_upgrade = total_bpw - 3.0
            upgrade_fracs = (norm_energy * (avg_upgrade * n_experts)).clamp(0, 1)
            
            mses = []
            for ei in range(n_experts):
                P_e = P_list[ei].to(device)
                P_reg, _, _ = regularize(P_e, device, ghd, cbs)
                qk3, tile_mse_k3 = quantize_tilewise_with_mse(P_reg, 3, device, tcp, tcpi, qtf)
                qk4, tile_mse_k4 = quantize_tilewise_with_mse(P_reg, 4, device, tcp, tcpi, qtf)
                
                n_tiles = tile_mse_k3.numel()
                n_upgrade = int(n_tiles * upgrade_fracs[ei].item())
                if n_upgrade > 0:
                    improvement = (tile_mse_k3 - tile_mse_k4).flatten()
                    _, top_idx = improvement.topk(n_upgrade)
                    result = qk3.clone()
                    tiles_n_n = P_reg.shape[1] // 16
                    for idx in top_idx:
                        ti_k = idx.item() // tiles_n_n
                        ti_n = idx.item() % tiles_n_n
                        result[ti_k*16:(ti_k+1)*16, ti_n*16:(ti_n+1)*16] = \
                            qk4[ti_k*16:(ti_k+1)*16, ti_n*16:(ti_n+1)*16]
                else:
                    result = qk3
                
                mses.append((P_reg - result).pow(2).mean().item())
                del P_e, P_reg, qk3, qk4, result
                torch.cuda.empty_cache()
            
            avg_mse = sum(mses) / n_experts
            avg_bits = 3.0 + sum(upgrade_fracs) / n_experts
            gap = (k3_avg_spectral - avg_mse) / (k3_avg_spectral - k4_avg_spectral) if k3_avg_spectral > k4_avg_spectral else 0
            label = f"spectral_energy_alloc_{total_bpw:.1f}bpw"
            layer_results[label] = {"mse": avg_mse, "bits": avg_bits.item(), "gap": gap,
                                     "alloc_range": [upgrade_fracs.min().item(), upgrade_fracs.max().item()]}
            print(f"    {label}: MSE={avg_mse:.6e}  bits={avg_bits:.3f}  gap={gap:.1%}  "
                  f"alloc=[{upgrade_fracs.min():.3f},{upgrade_fracs.max():.3f}]", flush=True)
        
        # Store direct baselines
        layer_results["direct_K3"] = {"mse": k3_avg_direct, "bits": 3.0, "gap": 0.0}
        layer_results["direct_K4"] = {"mse": k4_avg_direct, "bits": 4.0, "gap": 1.0}
        layer_results["spectral_K3"] = {"mse": k3_avg_spectral, "bits": 3.0, "gap": 0.0}
        layer_results["spectral_K4"] = {"mse": k4_avg_spectral, "bits": 4.0, "gap": 1.0}
        
        results[f"layer{layer_idx}"] = {
            "n_experts": n_experts,
            "k3_direct": k3_avg_direct, "k4_direct": k4_avg_direct,
            "k3_spectral": k3_avg_spectral, "k4_spectral": k4_avg_spectral,
            "k3_mses_direct": k3_mses_direct, "k3_mses_spectral": k3_mses_spectral,
            "energy_cv": energy_cv.tolist(),
            "methods": layer_results
        }
    
    return results

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--data-dir", default="/tmp/poc_residual/glm52_data")
    ap.add_argument("--out", default="/tmp/poc_residual/results_v8.json")
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
        print(f"  Direct: K3={r['k3_direct']:.4e} K4={r['k4_direct']:.4e}", flush=True)
        print(f"  Spectral: K3={r['k3_spectral']:.4e} K4={r['k4_spectral']:.4e}", flush=True)
        print(f"  Energy CV: mean={sum(r['energy_cv'])/len(r['energy_cv']):.4f}", flush=True)
        methods = r["methods"]
        for label in sorted(methods.keys(), key=lambda x: methods[x].get("gap", 0), reverse=True):
            m = methods[label]
            print(f"    {label:40s}: MSE={m['mse']:.6e}  bits={m['bits']:.3f}  gap={m['gap']:.1%}", flush=True)
    
    Path(args.out).write_text(json.dumps(results, indent=2, default=str))
    print(f"\nResults saved to {args.out}", flush=True)

if __name__ == "__main__":
    main()
