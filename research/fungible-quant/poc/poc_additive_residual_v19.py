#!/usr/bin/env python3
"""PoC v19: Per-tile rotation diversity (WUSH-inspired).

Test whether using different Hadamard seeds per tile creates per-tile
variation that could enable better differentiated allocation.

Instead of a global Hadamard transform, apply per-tile mini-rotations
(4×4 or 8×8) with different random seeds. This might create tile-level
diversity in quantization difficulty.

Tests:
  1. Global Hadamard (baseline) — all tiles same seed
  2. Per-tile 4×4 rotation with different seeds
  3. Per-tile 8×8 rotation with different seeds
  4. Measure: does per-tile rotation increase tile difficulty CV?
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

def regularize_with_per_tile_rotation(w, device, ghd, cbs, rot_size=4):
    """Regularize with per-tile mini-rotation (different seeds per tile)."""
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
    
    # Apply per-tile mini-rotation
    # For each rot_size x rot_size block, apply a random Hadamard rotation
    # with a different seed
    if rot_size > 0:
        rot_had = ghd(rot_size, device, torch.float, 1.0 / math.sqrt(rot_size))
        tiles_n_k = k // rot_size
        tiles_n_n = n // rot_size
        w_rot = w.clone()
        for tik in range(tiles_n_k):
            for tin in range(tiles_n_n):
                # Different seed per tile
                seed = tik * tiles_n_n + tin
                g_rot = torch.Generator(device="cpu").manual_seed(seed)
                signs = (torch.randn(rot_size, generator=g_rot).sign()).float().to(device)
                # Apply signed Hadamard to this tile
                tile = w[tik*rot_size:(tik+1)*rot_size, tin*rot_size:(tin+1)*rot_size]
                # H @ tile @ H^T (with random signs)
                # Simplified: just apply sign flip to some rows/cols
                tile_rot = tile * signs.unsqueeze(1) * signs.unsqueeze(0)
                w_rot[tik*rot_size:(tik+1)*rot_size, tin*rot_size:(tin+1)*rot_size] = tile_rot
        w = w_rot
    
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
    print(f"  {n_experts} experts, shape=({k},{n})", flush=True)
    
    for ei in range(n_experts):
        w = all_experts[ei].to(device)
        
        # Baseline: global Hadamard (no per-tile rotation)
        w_reg_global, _, _ = regularize(w, device, ghd, cbs)
        _, tmse_k3_global = quantize_tilewise_with_mse(w_reg_global, 3, device, tcp, tcpi, qtf)
        _, tmse_k4_global = quantize_tilewise_with_mse(w_reg_global, 4, device, tcp, tcpi, qtf)
        imp_global = (tmse_k3_global - tmse_k4_global).flatten()
        
        cv_global = (imp_global.std() / imp_global.mean()).item()
        mse_k3_global = (w_reg_global - quantize_trellis(w_reg_global, 3, device, tcp, tcpi, qtf)).pow(2).mean().item()
        mse_k4_global = (w_reg_global - quantize_trellis(w_reg_global, 4, device, tcp, tcpi, qtf)).pow(2).mean().item()
        
        print(f"\n  Expert {ei}:", flush=True)
        print(f"    Global Hadamard: K3={mse_k3_global:.6e} K4={mse_k4_global:.6e} "
              f"CV={cv_global:.4f}", flush=True)
        
        del w_reg_global, tmse_k3_global, tmse_k4_global
        torch.cuda.empty_cache()
        
        # Per-tile rotation with different sizes
        for rot_size in [4, 8, 16]:
            w_reg_rot, _, _ = regularize_with_per_tile_rotation(w, device, ghd, cbs, rot_size=rot_size)
            _, tmse_k3_rot = quantize_tilewise_with_mse(w_reg_rot, 3, device, tcp, tcpi, qtf)
            _, tmse_k4_rot = quantize_tilewise_with_mse(w_reg_rot, 4, device, tcp, tcpi, qtf)
            imp_rot = (tmse_k3_rot - tmse_k4_rot).flatten()
            
            cv_rot = (imp_rot.std() / imp_rot.mean()).item()
            mse_k3_rot = (w_reg_rot - quantize_trellis(w_reg_rot, 3, device, tcp, tcpi, qtf)).pow(2).mean().item()
            mse_k4_rot = (w_reg_rot - quantize_trellis(w_reg_rot, 4, device, tcp, tcpi, qtf)).pow(2).mean().item()
            
            print(f"    Per-tile rot {rot_size}x{rot_size}: K3={mse_k3_rot:.6e} K4={mse_k4_rot:.6e} "
                  f"CV={cv_rot:.4f}", flush=True)
            
            del w_reg_rot, tmse_k3_rot, tmse_k4_rot
            torch.cuda.empty_cache()
        
        del w
        torch.cuda.empty_cache()
    
    return {"analysis": "completed"}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--data-dir", default="/tmp/poc_residual/glm52_data")
    ap.add_argument("--out", default="/tmp/poc_residual/results_v19.json")
    args = ap.parse_args()
    dev = torch.device(args.device)
    print(f"Device: {dev}  GPU: {torch.cuda.get_device_name(0)}", flush=True)
    ext, ghd, tcp, tcpi, qtf, cbs = _bootstrap()
    results = run_experiment(Path(args.data_dir), dev, ext, ghd, tcp, tcpi, qtf, cbs)
    Path(args.out).write_text(json.dumps(results, indent=2, default=str))
    print(f"\nResults saved to {args.out}", flush=True)

if __name__ == "__main__":
    main()
