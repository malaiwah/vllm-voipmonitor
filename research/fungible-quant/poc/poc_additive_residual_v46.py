#!/usr/bin/env python3
"""PoC v46: Systematic stage allocation search for MSRT.

Exhaustive search over all possible stage allocations at 8bpw (6 residual bits)
and 9bpw (7 residual bits). Also tests all-K1 stages and all-K2 stages.

For 6 residual bits:
  1-stage: K6
  2-stage: (1,5), (2,4), (3,3), (4,2), (5,1)
  3-stage: (1,1,4), (1,2,3), (1,3,2), (1,4,1), (2,1,3), (2,2,2), (2,3,1), (3,1,2), (3,2,1), (4,1,1)
  4-stage: (1,1,1,3), (1,1,2,2), (1,1,3,1), (1,2,1,2), (1,2,2,1), (2,1,1,2), (2,1,2,1), (2,2,1,1), (1,1,1,1,2), etc.
  6-stage: (1,1,1,1,1,1)
"""
from __future__ import annotations
import argparse, json, math, os, sys, types, importlib.util, time
from itertools import combinations_with_replacement, permutations
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

def quantize_trellis_raw(data, K, device, tcp, tcpi, qtf):
    k, n = data.shape; tiles_n = n // 16; weight_q = torch.zeros_like(data)
    qa = {"K": K, "mcg": True}; perm = tcp(device); perm_i = tcpi(device)
    for bi in range(0, k, 16):
        rows = data[bi:bi+16]
        tiles = rows.reshape(16, tiles_n, 16).permute(1, 0, 2).reshape(tiles_n, 256)
        tiles = tiles[:, perm].contiguous()
        quant_w, _ = qtf(tiles, qa)
        quant_w = quant_w[:, perm_i].reshape(tiles_n, 16, 16).permute(1, 0, 2).reshape(16, n)
        weight_q[bi:bi+16] = quant_w
    return weight_q

def rescaled_trellis(base_q, residual, K_res, device, tcp, tcpi, qtf, cbs):
    residual_rms = residual.square().mean().sqrt().item()
    if residual_rms < 1e-12: return base_q
    scale = abs(cbs) / residual_rms
    scaled = residual * scale
    quant = quantize_trellis_raw(scaled, K_res, device, tcp, tcpi, qtf)
    return base_q + quant / scale

def apply_msrt(qk2, r2, stages, device, tcp, tcpi, qtf, cbs_scale, w_reg):
    """Apply MSRT with given stage allocation."""
    recon = qk2
    residual = r2
    for K_res in stages:
        recon = rescaled_trellis(recon, residual, K_res, device, tcp, tcpi, qtf, cbs_scale)
        residual = w_reg - recon
    return recon

def generate_allocations(total_bits, max_stages=6, min_K=1, max_K=6):
    """Generate all ordered stage allocations summing to total_bits."""
    allocations = set()
    def recurse(remaining, current, depth):
        if remaining == 0:
            allocations.add(tuple(current))
            return
        if depth >= max_stages:
            return
        for k in range(min_K, min(max_K, remaining) + 1):
            recurse(remaining - k, current + [k], depth + 1)
    recurse(total_bits, [], 0)
    return sorted(allocations, key=lambda x: (len(x), x))

def run_experiment(data_dir, device, ext, ghd, tcp, tcpi, qtf, cbs_scale):
    results = {}
    f = data_dir / f"layer10_all_gate_proj.pt"
    if not f.exists(): return results
    all_experts = torch.load(f, map_location="cpu")
    n_experts = min(3, all_experts.shape[0])
    k, n = all_experts.shape[1], all_experts.shape[2]
    print(f"  {n_experts} experts, {k}x{n}", flush=True)

    # Test at 8bpw (6 residual bits) and 9bpw (7 residual bits)
    for total_residual, bpw_label in [(6, "8bpw"), (7, "9bpw")]:
        allocations = generate_allocations(total_residual, max_stages=6)
        print(f"\n  {bpw_label} ({total_residual} residual bits): {len(allocations)} allocations", flush=True)

        alloc_results = {}
        for alloc in allocations:
            stages = list(alloc)
            mses = []
            for ei in range(n_experts):
                w = all_experts[ei].to(device)
                w_reg, _, _ = regularize(w, device, ghd, cbs_scale)
                del w
                qk2 = quantize_trellis_raw(w_reg, 2, device, tcp, tcpi, qtf)
                r2 = w_reg - qk2
                recon = apply_msrt(qk2, r2, stages, device, tcp, tcpi, qtf, cbs_scale, w_reg)
                mses.append((w_reg - recon).pow(2).mean().item())
                del w_reg, qk2, r2, recon
                torch.cuda.empty_cache()

            avg_mse = sum(mses) / len(mses)
            alloc_str = "+".join(f"K{s}" for s in stages)
            alloc_results[alloc_str] = avg_mse
            if avg_mse < 1e-3 or len(stages) <= 3:  # Only print good ones or short ones
                print(f"    K2+{alloc_str:<30} MSE={avg_mse:.4e}  ({len(stages)} stages)", flush=True)

        # Sort and print top 10
        print(f"\n  Top 10 allocations for {bpw_label}:", flush=True)
        sorted_allocs = sorted(alloc_results.items(), key=lambda x: x[1])
        for i, (name, mse) in enumerate(sorted_allocs[:10]):
            marker = " ***" if i == 0 else ""
            print(f"    {i+1:>2}. K2+{name:<30} MSE={mse:.4e}{marker}", flush=True)

        # Print worst 3 for comparison
        print(f"  Bottom 3:", flush=True)
        for name, mse in sorted_allocs[-3:]:
            print(f"         K2+{name:<30} MSE={mse:.4e}", flush=True)

        results[bpw_label] = {"best": sorted_allocs[0], "top10": sorted_allocs[:10], "all": alloc_results}

    return results

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--data-dir", default="/tmp/poc_residual/glm52_data")
    ap.add_argument("--out", default="/tmp/poc_residual/results_v46.json")
    args = ap.parse_args()
    dev = torch.device(args.device)
    print(f"Device: {dev}  GPU: {torch.cuda.get_device_name(0)}", flush=True)
    ext, ghd, tcp, tcpi, qtf, cbs_scale = _bootstrap()
    results = run_experiment(Path(args.data_dir), dev, ext, ghd, tcp, tcpi, qtf, cbs_scale)
    Path(args.out).write_text(json.dumps(results, indent=2, default=str))
    print(f"\nResults saved to {args.out}", flush=True)

if __name__ == "__main__":
    main()
