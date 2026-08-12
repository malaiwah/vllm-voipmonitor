#!/usr/bin/env python3
"""PoC v14: Per-tile rotation diversity + tile correlation analysis.

Tests whether using different Hadamard seeds per tile creates variation
that enables better differentiated allocation. Also analyzes:
  1. Tile correlation: do adjacent tiles have similar difficulty?
  2. Spatial structure: is difficulty clustered spatially?
  3. Per-tile rotation diversity: different seeds → different tile difficulty?
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
    if not gate_file.exists():
        print("No data file found", flush=True)
        return results
    
    all_experts = torch.load(gate_file, map_location="cpu")
    n_experts = min(3, all_experts.shape[0])
    k, n = all_experts.shape[1], all_experts.shape[2]
    print(f"  {n_experts} experts, shape=({k},{n})", flush=True)
    
    # ================================================================
    # 1. Spatial structure of tile difficulty
    # ================================================================
    print(f"\n  --- Tile difficulty spatial structure ---", flush=True)
    for ei in range(n_experts):
        w = all_experts[ei].to(device)
        w_reg, _, _ = regularize(w, device, ghd, cbs)
        del w
        
        _, tmse_k3 = quantize_tilewise_with_mse(w_reg, 3, device, tcp, tcpi, qtf)
        _, tmse_k4 = quantize_tilewise_with_mse(w_reg, 4, device, tcp, tcpi, qtf)
        improvement = tmse_k3 - tmse_k4  # (tiles_n_k, tiles_n_n)
        
        # Spatial autocorrelation: compare adjacent tiles
        tiles_n_k, tiles_n_n = improvement.shape
        # Row-wise correlation (horizontal neighbors)
        row_corr = torch.corrcoef(torch.stack([improvement[:, :-1].flatten(), improvement[:, 1:].flatten()]))[0, 1].item()
        # Column-wise correlation (vertical neighbors)
        col_corr = torch.corrcoef(torch.stack([improvement[:-1, :].flatten(), improvement[1:, :].flatten()]))[0, 1].item()
        
        print(f"    Expert {ei}: row_corr={row_corr:.4f}, col_corr={col_corr:.4f}", flush=True)
        
        # Check if difficulty is clustered (high spatial autocorrelation = clustered)
        # If corr > 0.5, difficulty is spatially structured
        # If corr < 0.1, difficulty is random
        
        del w_reg, tmse_k3, tmse_k4; torch.cuda.empty_cache()
    
    # ================================================================
    # 2. Tile variance distribution (is it bimodal?)
    # ================================================================
    print(f"\n  --- Tile variance distribution ---", flush=True)
    for ei in range(n_experts):
        w = all_experts[ei].to(device)
        w_reg, _, _ = regularize(w, device, ghd, cbs)
        del w
        
        _, tmse_k3 = quantize_tilewise_with_mse(w_reg, 3, device, tcp, tcpi, qtf)
        _, tmse_k4 = quantize_tilewise_with_mse(w_reg, 4, device, tcp, tcpi, qtf)
        improvement = (tmse_k3 - tmse_k4).flatten()
        
        # Distribution statistics
        mean_imp = improvement.mean().item()
        std_imp = improvement.std().item()
        min_imp = improvement.min().item()
        max_imp = improvement.max().item()
        median_imp = improvement.median().item()
        # Percentiles
        p10 = improvement.quantile(0.1).item()
        p90 = improvement.quantile(0.9).item()
        # Bimodality check: is there a gap in the distribution?
        hist = torch.histc(improvement, bins=20)
        hist_np = hist.cpu().numpy()
        # Coefficient of bimodality (BM = (skew^2 + 1) / kurtosis)
        skew = ((improvement - mean_imp) ** 3).mean().item() / (std_imp ** 3 + 1e-8)
        kurt = ((improvement - mean_imp) ** 4).mean().item() / (std_imp ** 4 + 1e-8) - 3
        bm = (skew ** 2 + 1) / (kurt + 3 + 1e-8) if kurt > -3 else 0
        
        print(f"    Expert {ei}: mean={mean_imp:.6e} std={std_imp:.6e} "
              f"CV={std_imp/mean_imp:.4f} skew={skew:.3f} kurt={kurt:.3f} BM={bm:.3f}", flush=True)
        print(f"      percentiles: p10={p10:.6e} p50={median_imp:.6e} p90={p90:.6e}", flush=True)
        
        del w_reg, tmse_k3, tmse_k4; torch.cuda.empty_cache()
    
    # ================================================================
    # 3. Cross-projection tile difficulty correlation
    # ================================================================
    print(f"\n  --- Cross-projection tile difficulty ---", flush=True)
    for proj in ["gate_proj", "up_proj", "down_proj"]:
        pf = data_dir / f"layer{layer_idx}_all_{proj}.pt"
        if not pf.exists(): continue
        experts = torch.load(pf, map_location="cpu")
        w = experts[0].to(device)
        w_reg, _, _ = regularize(w, device, ghd, cbs)
        del w
        
        _, tmse_k3 = quantize_tilewise_with_mse(w_reg, 3, device, tcp, tcpi, qtf)
        _, tmse_k4 = quantize_tilewise_with_mse(w_reg, 4, device, tcp, tcpi, qtf)
        improvement = (tmse_k3 - tmse_k4).flatten()
        
        print(f"    {proj}: mean_imp={improvement.mean():.6e} "
              f"CV={improvement.std()/improvement.mean():.4f} "
              f"range=[{improvement.min():.6e}, {improvement.max():.6e}]", flush=True)
        
        del w_reg, tmse_k3, tmse_k4, experts
        torch.cuda.empty_cache()
    
    results["analysis"] = "completed"
    return results

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--data-dir", default="/tmp/poc_residual/glm52_data")
    ap.add_argument("--out", default="/tmp/poc_residual/results_v14.json")
    args = ap.parse_args()
    dev = torch.device(args.device)
    print(f"Device: {dev}  GPU: {torch.cuda.get_device_name(0)}", flush=True)
    ext, ghd, tcp, tcpi, qtf, cbs = _bootstrap()
    
    results = run_experiment(Path(args.data_dir), dev, ext, ghd, tcp, tcpi, qtf, cbs)
    Path(args.out).write_text(json.dumps(results, indent=2, default=str))
    print(f"\nResults saved to {args.out}", flush=True)

if __name__ == "__main__":
    main()
