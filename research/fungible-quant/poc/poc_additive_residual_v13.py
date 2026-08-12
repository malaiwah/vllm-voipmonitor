#!/usr/bin/env python3
"""PoC v13: Cross-layer (vertical) tile-level allocation.

Instead of allocating bits per-expert within a single layer, allocate
across layers. Different layers may have different tile difficulty distributions.

Also tests:
  1. Per-layer tile difficulty variation (does layer 10 differ from layer 40?)
  2. Global budget allocation across layers (give more bits to harder layers)
  3. Cross-layer tier bitmap sharing (do similar layers share patterns?)
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
    levels = torch.tensor([-1.5104, -0.4528, 0.4528, 1.5104]) * sigma
    flat_cpu = r.flatten().cpu()
    d = (flat_cpu.unsqueeze(1) - levels.unsqueeze(0)).abs()
    return levels[d.argmin(dim=1)].to(r.device).reshape(r.shape)

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

def compute_layer_stats(all_experts, device, ghd, cbs, tcp, tcpi, qtf, n_experts=3):
    """Compute per-layer K3/K4/K5 MSE statistics for a few experts."""
    k3_mses = []; k4_mses = []; k5_mses = []
    for ei in range(min(n_experts, all_experts.shape[0])):
        w = all_experts[ei].to(device)
        w_reg, _, _ = regularize(w, device, ghd, cbs)
        del w
        qk3 = quantize_trellis(w_reg, 3, device, tcp, tcpi, qtf)
        k3_mses.append((w_reg - qk3).pow(2).mean().item())
        del qk3; torch.cuda.empty_cache()
        qk4 = quantize_trellis(w_reg, 4, device, tcp, tcpi, qtf)
        k4_mses.append((w_reg - qk4).pow(2).mean().item())
        r4 = w_reg - qk4; del qk4
        lloyd = q2b_lloyd(r4)
        k5_mses.append((r4 - lloyd).pow(2).mean().item())
        del w_reg, r4, lloyd; torch.cuda.empty_cache()
    return {
        "k3_avg": sum(k3_mses) / len(k3_mses),
        "k4_avg": sum(k4_mses) / len(k4_mses),
        "k5_avg": sum(k5_mses) / len(k5_mses),
        "k3_range": [min(k3_mses), max(k3_mses)],
        "k4_range": [min(k4_mses), max(k4_mses)],
    }

def run_experiment(data_dir, device, ext, ghd, tcp, tcpi, qtf, cbs):
    layer_indices = [10, 40]
    results = {}
    
    # Step 1: Compare per-layer statistics
    print(f"\n{'='*70}", flush=True)
    print(f"Cross-layer analysis", flush=True)
    print(f"{'='*70}", flush=True)
    
    layer_stats = {}
    for layer_idx in layer_indices:
        gate_file = data_dir / f"layer{layer_idx}_all_gate_proj.pt"
        if not gate_file.exists(): continue
        all_experts = torch.load(gate_file, map_location="cpu")
        print(f"\n  Layer {layer_idx}: {all_experts.shape[0]} experts", flush=True)
        stats = compute_layer_stats(all_experts, device, ghd, cbs, tcp, tcpi, qtf, n_experts=3)
        layer_stats[layer_idx] = stats
        print(f"    K3={stats['k3_avg']:.6e}  K4={stats['k4_avg']:.6e}  K5={stats['k5_avg']:.6e}", flush=True)
        print(f"    K3 range: [{stats['k3_range'][0]:.6e}, {stats['k3_range'][1]:.6e}]", flush=True)
        del all_experts; torch.cuda.empty_cache()
    
    # Step 2: Cross-layer comparison
    if len(layer_stats) >= 2:
        layers = list(layer_stats.keys())
        print(f"\n  Cross-layer comparison:", flush=True)
        for i in range(len(layers)):
            for j in range(i+1, len(layers)):
                l1, l2 = layers[i], layers[j]
                s1, s2 = layer_stats[l1], layer_stats[l2]
                k3_ratio = s1['k3_avg'] / s2['k3_avg']
                k4_ratio = s1['k4_avg'] / s2['k4_avg']
                print(f"    Layer {l1} vs {l2}: K3 ratio={k3_ratio:.4f}, K4 ratio={k4_ratio:.4f}", flush=True)
        
        # Step 3: Global budget allocation across layers
        print(f"\n  Global budget allocation across layers:", flush=True)
        
        # If one layer has higher MSE, give it more K4 tiles
        for target_bpw in [3.5, 4.0, 4.5]:
            # Simple: allocate proportionally to MSE
            total_mse = sum(layer_stats[l]['k3_avg'] for l in layer_stats)
            weights = {l: layer_stats[l]['k3_avg'] / total_mse for l in layer_stats}
            
            # Weighted allocation: more budget to higher-MSE layer
            for l in layer_stats:
                layer_budget = target_bpw * len(layer_stats) * weights[l]
                print(f"    Target {target_bpw} bpw: Layer {l} gets {layer_budget:.3f} bpw "
                      f"(weight={weights[l]:.4f})", flush=True)
    
    # Step 4: Tile difficulty distribution comparison
    print(f"\n  Tile difficulty distribution comparison:", flush=True)
    for layer_idx in layer_indices:
        gate_file = data_dir / f"layer{layer_idx}_all_gate_proj.pt"
        if not gate_file.exists(): continue
        all_experts = torch.load(gate_file, map_location="cpu")
        w = all_experts[0].to(device)
        w_reg, _, _ = regularize(w, device, ghd, cbs)
        del w
        
        _, tmse_k3 = quantize_tilewise_with_mse(w_reg, 3, device, tcp, tcpi, qtf)
        _, tmse_k4 = quantize_tilewise_with_mse(w_reg, 4, device, tcp, tcpi, qtf)
        improvement = (tmse_k3 - tmse_k4).flatten()
        
        print(f"    Layer {layer_idx}:", flush=True)
        print(f"      Tile K3→K4 improvement: mean={improvement.mean():.6e}, "
              f"std={improvement.std():.6e}, CV={improvement.std()/improvement.mean():.4f}", flush=True)
        print(f"      Top 10% tiles improvement: {improvement.topk(int(len(improvement)*0.1)).values.mean():.6e}", flush=True)
        print(f"      Bottom 10% tiles improvement: {improvement.topk(int(len(improvement)*0.1), largest=False).values.mean():.6e}", flush=True)
        
        del w_reg, tmse_k3, tmse_k4; torch.cuda.empty_cache()
    
    results["layer_stats"] = {str(k): v for k, v in layer_stats.items()}
    return results

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--data-dir", default="/tmp/poc_residual/glm52_data")
    ap.add_argument("--out", default="/tmp/poc_residual/results_v13.json")
    args = ap.parse_args()
    dev = torch.device(args.device)
    print(f"Device: {dev}  GPU: {torch.cuda.get_device_name(0)}", flush=True)
    ext, ghd, tcp, tcpi, qtf, cbs = _bootstrap()
    
    results = run_experiment(Path(args.data_dir), dev, ext, ghd, tcp, tcpi, qtf, cbs)
    
    print(f"\n{'='*70}\nSUMMARY\n{'='*70}", flush=True)
    for layer_key, stats in results["layer_stats"].items():
        print(f"  Layer {layer_key}: K3={stats['k3_avg']:.6e} K4={stats['k4_avg']:.6e} K5={stats['k5_avg']:.6e}", flush=True)
    
    Path(args.out).write_text(json.dumps(results, indent=2, default=str))
    print(f"\nResults saved to {args.out}", flush=True)

if __name__ == "__main__":
    main()
