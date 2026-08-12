#!/usr/bin/env python3
"""PoC v22: Larger Lloyd-Max codebooks for higher tiers.

Instead of K6 = K5 + 1-bit scalar (weak), try:
  K6_alt = K4 + 4-bit Lloyd-Max on residual
  K7_alt = K4 + 6-bit Lloyd-Max on residual
  K8_alt = K4 + 8-bit Lloyd-Max on residual

Theory: a single N-bit codebook captures the residual distribution
better than stacking independent smaller codebooks.

Also tests: K5 = K4 + 2-bit Lloyd (current best) vs K5_alt = K4 + 2-bit uniform.
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
    """N-bit Lloyd-Max quantizer. Returns quantized values. Chunked for large codebooks."""
    n_levels = 2 ** n_bits
    sigma = r.std().item()
    if sigma < 1e-12:
        return torch.zeros_like(r)

    # Initialize: uniform levels across ±3σ
    levels = torch.linspace(-3 * sigma, 3 * sigma, n_levels, device=r.device)
    flat = r.flatten()
    n_elem = flat.numel()
    chunk = max(1, min(n_elem, (1024 * 1024 * 1024) // (n_levels * 4)))  # ~1GB per chunk

    for _ in range(n_iters):
        assign = torch.empty(n_elem, dtype=torch.long, device=r.device)
        for s in range(0, n_elem, chunk):
            e = min(s + chunk, n_elem)
            d = (flat[s:e].unsqueeze(1) - levels.unsqueeze(0)).abs()
            assign[s:e] = d.argmin(dim=1)
        new_levels = levels.clone()
        for i in range(n_levels):
            mask = assign == i
            if mask.sum() > 0:
                new_levels[i] = flat[mask].mean()
        if (new_levels - levels).abs().max() < 1e-10 * sigma:
            break
        levels = new_levels

    # Final assignment (chunked)
    result = torch.empty_like(flat)
    for s in range(0, n_elem, chunk):
        e = min(s + chunk, n_elem)
        d = (flat[s:e].unsqueeze(1) - levels.unsqueeze(0)).abs()
        result[s:e] = levels[d.argmin(dim=1)]
    return result.reshape(r.shape)

def q1b_scalar(r):
    s = r.abs().mean().item()
    return torch.zeros_like(r) if s < 1e-12 else torch.sign(r) * s

def run_experiment(data_dir, device, ext, ghd, tcp, tcpi, qtf, cbs):
    results = {}
    layer_idx = 10
    gate_file = data_dir / f"layer{layer_idx}_all_gate_proj.pt"
    if not gate_file.exists():
        return results
    
    all_experts = torch.load(gate_file, map_location="cpu")
    n_experts = min(2, all_experts.shape[0])
    k, n = all_experts.shape[1], all_experts.shape[2]
    n_tiles = (k // 16) * (n // 16)
    print(f"  {n_experts} experts, {n_tiles} tiles each", flush=True)
    
    all_methods = {}
    
    for ei in range(n_experts):
        w = all_experts[ei].to(device)
        w_reg, _, _ = regularize(w, device, ghd, cbs)
        del w
        
        # Base tiers
        qk3 = quantize_trellis(w_reg, 3, device, tcp, tcpi, qtf)
        qk4 = quantize_trellis(w_reg, 4, device, tcp, tcpi, qtf)
        
        # K4 residual
        r4 = w_reg - qk4
        
        # Method A: Current stacking (K5=2LM, K6=+1sc, K7=+1sc)
        lm2 = lloyd_max_quantize(r4, 2)
        recon_k5_stack = qk4 + lm2
        r5s = w_reg - recon_k5_stack
        sc1 = q1b_scalar(r5s)
        recon_k6_stack = recon_k5_stack + sc1
        r6s = w_reg - recon_k6_stack
        sc2 = q1b_scalar(r6s)
        recon_k7_stack = recon_k6_stack + sc2
        
        # Method B: Direct Lloyd-Max on K4 residual
        lm4 = lloyd_max_quantize(r4, 4)      # K6_alt = K4 + 4-bit LM
        recon_k6_lm4 = qk4 + lm4
        lm6 = lloyd_max_quantize(r4, 6)      # K7_alt = K4 + 6-bit LM
        recon_k7_lm6 = qk4 + lm6
        lm8 = lloyd_max_quantize(r4, 8)      # K8_alt = K4 + 8-bit LM
        recon_k8_lm8 = qk4 + lm8
        
        # Method C: Progressive Lloyd-Max on residuals
        # K5 = K4 + 2LM, then 4-bit LM on K5 residual
        r5 = w_reg - recon_k5_stack
        lm4_r5 = lloyd_max_quantize(r5, 4)   # K7_prog = K5 + 4-bit LM
        recon_k7_prog = recon_k5_stack + lm4_r5
        r7 = w_reg - recon_k7_prog
        lm2_r7 = lloyd_max_quantize(r7, 2)   # K8_prog = K7 + 2-bit LM
        recon_k8_prog = recon_k7_prog + lm2_r7
        
        # Method D: K3 residual with larger codebooks
        r3 = w_reg - qk3
        lm4_r3 = lloyd_max_quantize(r3, 4)   # K5_from3 = K3 + 4-bit LM (5 bpw)
        recon_k5_from3 = qk3 + lm4_r3
        lm6_r3 = lloyd_max_quantize(r3, 6)   # K6_from3 = K3 + 6-bit LM (6 bpw)
        recon_k6_from3 = qk3 + lm6_r3
        
        # Compute MSEs
        methods = {
            "K3": (qk3, 3.0),
            "K4": (qk4, 4.0),
            # Method A: stacking
            "K5_stack(K4+2LM)": (recon_k5_stack, 5.0),
            "K6_stack(K4+2LM+1sc)": (recon_k6_stack, 6.0),
            "K7_stack(K4+2LM+2sc)": (recon_k7_stack, 7.0),
            # Method B: direct LM on K4 residual
            "K6_LM4(K4+4LM)": (recon_k6_lm4, 6.0),
            "K7_LM6(K4+6LM)": (recon_k7_lm6, 7.0),
            "K8_LM8(K4+8LM)": (recon_k8_lm8, 8.0),
            # Method C: progressive LM
            "K7_prog(K5+4LM)": (recon_k7_prog, 7.0),
            "K8_prog(K5+4LM+2LM)": (recon_k8_prog, 8.0),
            # Method D: from K3
            "K5_from3(K3+4LM)": (recon_k5_from3, 5.0),
            "K6_from3(K3+6LM)": (recon_k6_from3, 6.0),
        }
        
        for name, (recon, bpw) in methods.items():
            mse = (w_reg - recon).pow(2).mean().item()
            if name not in all_methods:
                all_methods[name] = {"mses": [], "bpw": bpw}
            all_methods[name]["mses"].append(mse)
        
        # Cleanup
        del w_reg, qk3, qk4, r4, lm2, recon_k5_stack, r5s, sc1, recon_k6_stack
        del r6s, sc2, recon_k7_stack, lm4, recon_k6_lm4, lm6, recon_k7_lm6
        del lm8, recon_k8_lm8, r5, lm4_r5, recon_k7_prog, r7, lm2_r7, recon_k8_prog
        del r3, lm4_r3, recon_k5_from3, lm6_r3, recon_k6_from3
        torch.cuda.empty_cache()
    
    # Print comparison
    print(f"\n  {'Method':<35} {'bpw':>4} {'MSE':>12} {'vs K4':>10}", flush=True)
    print(f"  {'-'*65}", flush=True)
    k4_mse = sum(all_methods["K4"]["mses"]) / len(all_methods["K4"]["mses"])
    for name in sorted(all_methods.keys(), key=lambda x: all_methods[x]["bpw"]):
        r = all_methods[name]
        avg_mse = sum(r["mses"]) / len(r["mses"])
        vs_k4 = f"{avg_mse/k4_mse*100:.1f}%" if k4_mse > 0 else "N/A"
        print(f"  {name:<35} {r['bpw']:>4.1f} {avg_mse:>12.4e} {vs_k4:>10}", flush=True)
    
    results["methods"] = {k: {"bpw": v["bpw"], "avg_mse": sum(v["mses"])/len(v["mses"])}
                          for k, v in all_methods.items()}
    return results

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--data-dir", default="/tmp/poc_residual/glm52_data")
    ap.add_argument("--out", default="/tmp/poc_residual/results_v22.json")
    args = ap.parse_args()
    dev = torch.device(args.device)
    print(f"Device: {dev}  GPU: {torch.cuda.get_device_name(0)}", flush=True)
    ext, ghd, tcp, tcpi, qtf, cbs = _bootstrap()
    results = run_experiment(Path(args.data_dir), dev, ext, ghd, tcp, tcpi, qtf, cbs)
    Path(args.out).write_text(json.dumps(results, indent=2, default=str))
    print(f"\nResults saved to {args.out}", flush=True)

if __name__ == "__main__":
    main()
