#!/usr/bin/env python3
"""PoC v20: K5 residual decomposition — sign + magnitude vs Lloyd-Max.

K5 = K4 + 2-bit residual. Currently using Lloyd-Max (4 levels).
Test alternatives:
  1. Lloyd-Max (baseline): 4 levels at ±0.45σ, ±1.51σ
  2. Sign + 1-bit magnitude: sign(r) × (|r| > threshold ? s1 : s2)
  3. Uniform 2-bit: 4 equally spaced levels
  4. 1-bit scalar × 2: two rounds of 1-bit sign quantization
  
Also test K6 alternatives:
  1. 1-bit scalar (baseline)
  2. Dithered 1-bit
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

def q2b_sign_magnitude(r):
    """Sign + 1-bit magnitude: sign(r) * (|r| > threshold ? large : small)"""
    sigma = r.std().item()
    if sigma < 1e-12: return torch.zeros_like(r)
    threshold = sigma * 0.6745  # median of |N(0,1)| ≈ 0.6745
    signs = torch.sign(r)
    abs_r = r.abs()
    # Two magnitude levels
    small = abs_r[abs_r <= threshold].mean().item() if (abs_r <= threshold).any() else sigma * 0.3
    large = abs_r[abs_r > threshold].mean().item() if (abs_r > threshold).any() else sigma * 1.2
    result = torch.zeros_like(r)
    result[abs_r <= threshold] = signs[abs_r <= threshold] * small
    result[abs_r > threshold] = signs[abs_r > threshold] * large
    return result

def q2b_uniform(r):
    """Uniform 2-bit: 4 equally spaced levels."""
    sigma = r.std().item()
    if sigma < 1e-12: return torch.zeros_like(r)
    max_val = r.abs().max().item()
    levels = torch.linspace(-max_val, max_val, 4, device=r.device)
    flat_cpu = r.flatten().cpu()
    d = (flat_cpu.unsqueeze(1) - levels.cpu().unsqueeze(0)).abs()
    return levels[d.argmin(dim=1)].to(r.device).reshape(r.shape)

def q2b_double_scalar(r):
    """Two rounds of 1-bit scalar: r1 = sign(r)*s1, r2 = sign(r-r1)*s2"""
    r1 = q1b_scalar(r)
    r_rem = r - r1
    r2 = q1b_scalar(r_rem)
    return r1 + r2

def run_experiment(data_dir, device, ext, ghd, tcp, tcpi, qtf, cbs):
    results = {}
    layer_idx = 10
    gate_file = data_dir / f"layer{layer_idx}_all_gate_proj.pt"
    if not gate_file.exists(): return results
    
    all_experts = torch.load(gate_file, map_location="cpu")
    n_experts = min(3, all_experts.shape[0])
    k, n = all_experts.shape[1], all_experts.shape[2]
    print(f"  {n_experts} experts, shape=({k},{n})", flush=True)
    
    all_mses = {}
    
    for ei in range(n_experts):
        w = all_experts[ei].to(device)
        w_reg, _, _ = regularize(w, device, ghd, cbs)
        del w
        
        # K3 and K4 baselines
        qk3 = quantize_trellis(w_reg, 3, device, tcp, tcpi, qtf)
        qk4 = quantize_trellis(w_reg, 4, device, tcp, tcpi, qtf)
        mse_k3 = (w_reg - qk3).pow(2).mean().item()
        mse_k4 = (w_reg - qk4).pow(2).mean().item()
        del qk3
        torch.cuda.empty_cache()
        
        # K4 residual
        r4 = w_reg - qk4
        
        # K5 variants (K4 + 2-bit residual)
        k5_lloyd = qk4 + q2b_lloyd(r4)
        k5_sign_mag = qk4 + q2b_sign_magnitude(r4)
        k5_uniform = qk4 + q2b_uniform(r4)
        k5_double_scalar = qk4 + q2b_double_scalar(r4)
        
        mse_k5_lloyd = (w_reg - k5_lloyd).pow(2).mean().item()
        mse_k5_sign_mag = (w_reg - k5_sign_mag).pow(2).mean().item()
        mse_k5_uniform = (w_reg - k5_uniform).pow(2).mean().item()
        mse_k5_double_scalar = (w_reg - k5_double_scalar).pow(2).mean().item()
        
        del qk4, r4, k5_lloyd, k5_sign_mag, k5_uniform, k5_double_scalar
        torch.cuda.empty_cache()
        
        # K5 + 1-bit residual (K6 variants)
        qk4_2 = quantize_trellis(w_reg, 4, device, tcp, tcpi, qtf)
        r4_2 = w_reg - qk4_2
        k5_base = qk4_2 + q2b_lloyd(r4_2)
        r5 = w_reg - k5_base
        
        k6_scalar = k5_base + q1b_scalar(r5)
        mse_k6_scalar = (w_reg - k6_scalar).pow(2).mean().item()
        
        del qk4_2, r4_2, k5_base, r5, k6_scalar
        torch.cuda.empty_cache()
        
        # Collect results
        methods = {
            "K3": mse_k3, "K4": mse_k4,
            "K5_lloyd": mse_k5_lloyd, "K5_sign_mag": mse_k5_sign_mag,
            "K5_uniform": mse_k5_uniform, "K5_double_scalar": mse_k5_double_scalar,
            "K6_scalar": mse_k6_scalar,
        }
        
        for label, mse in methods.items():
            if label not in all_mses: all_mses[label] = []
            all_mses[label].append(mse)
        
        print(f"\n  Expert {ei}:", flush=True)
        for label, mse in methods.items():
            print(f"    {label:20s}: MSE={mse:.6e}", flush=True)
        
        del w_reg
        torch.cuda.empty_cache()
    
    # Average
    print(f"\n  Average across {n_experts} experts:", flush=True)
    avg_mses = {label: sum(mses)/len(mses) for label, mses in all_mses.items()}
    for label in sorted(avg_mses.keys(), key=lambda x: avg_mses[x]):
        print(f"    {label:20s}: MSE={avg_mses[label]:.6e}", flush=True)
    
    results["avg_mses"] = avg_mses
    return results

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--data-dir", default="/tmp/poc_residual/glm52_data")
    ap.add_argument("--out", default="/tmp/poc_residual/results_v20.json")
    args = ap.parse_args()
    dev = torch.device(args.device)
    print(f"Device: {dev}  GPU: {torch.cuda.get_device_name(0)}", flush=True)
    ext, ghd, tcp, tcpi, qtf, cbs = _bootstrap()
    results = run_experiment(Path(args.data_dir), dev, ext, ghd, tcp, tcpi, qtf, cbs)
    Path(args.out).write_text(json.dumps(results, indent=2, default=str))
    print(f"\nResults saved to {args.out}", flush=True)

if __name__ == "__main__":
    main()
