#!/usr/bin/env python3
"""PoC v21: Extend to 7.0 bpw with 5-tier K3/K4/K5/K6/K7.

K7 = K6 + 1-bit scalar on K6 residual.
Tests whether the smooth Pareto curve extends beyond 6.0 bpw.
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
    print(f"  {n_experts} experts, {n_tiles} tiles each", flush=True)
    
    all_results = {}
    for ei in range(n_experts):
        w = all_experts[ei].to(device)
        w_reg, _, _ = regularize(w, device, ghd, cbs)
        del w
        
        # Compute 5 tiers: K3, K4, K5, K6, K7
        qk3, tmse_k3 = quantize_tilewise_with_mse(w_reg, 3, device, tcp, tcpi, qtf)
        qk4, tmse_k4 = quantize_tilewise_with_mse(w_reg, 4, device, tcp, tcpi, qtf)
        
        # K5 = K4 + 2-bit Lloyd
        r4 = w_reg - qk4; del qk4
        lloyd = q2b_lloyd(r4)
        recon_k5 = (w_reg - r4) + lloyd
        
        # K6 = K5 + 1-bit scalar
        r5 = w_reg - recon_k5
        scalar_k6 = q1b_scalar(r5)
        recon_k6 = recon_k5 + scalar_k6
        
        # K7 = K6 + 1-bit scalar
        r6 = w_reg - recon_k6
        scalar_k7 = q1b_scalar(r6)
        recon_k7 = recon_k6 + scalar_k7
        
        # Per-tile MSEs
        tiles_n_k = k // 16; tiles_n_n = n // 16
        tmse_k5 = torch.zeros_like(tmse_k4)
        tmse_k6 = torch.zeros_like(tmse_k4)
        tmse_k7 = torch.zeros_like(tmse_k4)
        for tik in range(tiles_n_k):
            for tin in range(tiles_n_n):
                rs, re = tik * 16, (tik + 1) * 16
                cs, ce = tin * 16, (tin + 1) * 16
                orig = w_reg[rs:re, cs:ce]
                tmse_k5[tik, tin] = (orig - recon_k5[rs:re, cs:ce]).pow(2).mean()
                tmse_k6[tik, tin] = (orig - recon_k6[rs:re, cs:ce]).pow(2).mean()
                tmse_k7[tik, tin] = (orig - recon_k7[rs:re, cs:ce]).pow(2).mean()
        
        # Build upgrades for 5-tier
        upgrades = []
        for i in range(n_tiles):
            upgrades.append(((tmse_k3 - tmse_k4).flatten()[i].item(), i, 3, 4))
            upgrades.append(((tmse_k4 - tmse_k5).flatten()[i].item(), i, 4, 5))
            upgrades.append(((tmse_k5 - tmse_k6).flatten()[i].item(), i, 5, 6))
            upgrades.append(((tmse_k6 - tmse_k7).flatten()[i].item(), i, 6, 7))
        upgrades.sort(key=lambda x: -x[0])
        
        for target_bpw_10 in range(30, 76):  # 3.0 to 7.5
            target_bpw = target_bpw_10 / 10.0
            tier = [3] * n_tiles
            current_bits = 3.0 * n_tiles
            target_bits = target_bpw * n_tiles
            for benefit, tile_idx, from_t, to_t in upgrades:
                if current_bits + 1 > target_bits: continue
                if tier[tile_idx] != from_t: continue
                if benefit <= 0: continue
                tier[tile_idx] = to_t
                current_bits += 1
            
            result = qk3.clone()
            qk4_recon = w_reg - r4
            for i, t in enumerate(tier):
                tik = i // tiles_n_n; tin = i % tiles_n_n
                rs, re = tik * 16, (tik + 1) * 16
                cs, ce = tin * 16, (tin + 1) * 16
                if t == 4: result[rs:re, cs:ce] = qk4_recon[rs:re, cs:ce]
                elif t == 5: result[rs:re, cs:ce] = recon_k5[rs:re, cs:ce]
                elif t == 6: result[rs:re, cs:ce] = recon_k6[rs:re, cs:ce]
                elif t == 7: result[rs:re, cs:ce] = recon_k7[rs:re, cs:ce]
            
            mse = (w_reg - result).pow(2).mean().item()
            avg_bits = current_bits / n_tiles
            key = f"{target_bpw:.1f}"
            if key not in all_results: all_results[key] = {"mses": [], "bits": []}
            all_results[key]["mses"].append(mse)
            all_results[key]["bits"].append(avg_bits)
        
        del w_reg, qk3, r4, lloyd, recon_k5, r5, scalar_k6, recon_k6, r6, scalar_k7, recon_k7
        del tmse_k3, tmse_k4, tmse_k5, tmse_k6, tmse_k7
        torch.cuda.empty_cache()
    
    print(f"\n  Extended Pareto (3.0-7.5 bpw, 0.1-bit steps):", flush=True)
    pareto = []
    for key in sorted(all_results.keys(), key=float):
        r = all_results[key]
        avg_mse = sum(r["mses"]) / len(r["mses"])
        avg_bits = sum(r["bits"]) / len(r["bits"])
        pareto.append({"target_bpw": float(key), "actual_bpw": avg_bits, "mse": avg_mse})
        print(f"    {key}  actual={avg_bits:.3f}  MSE={avg_mse:.6e}", flush=True)
    
    results["pareto"] = pareto
    return results

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--data-dir", default="/tmp/poc_residual/glm52_data")
    ap.add_argument("--out", default="/tmp/poc_residual/results_v21.json")
    args = ap.parse_args()
    dev = torch.device(args.device)
    print(f"Device: {dev}  GPU: {torch.cuda.get_device_name(0)}", flush=True)
    ext, ghd, tcp, tcpi, qtf, cbs = _bootstrap()
    results = run_experiment(Path(args.data_dir), dev, ext, ghd, tcp, tcpi, qtf, cbs)
    Path(args.out).write_text(json.dumps(results, indent=2, default=str))
    print(f"\nResults saved to {args.out}", flush=True)

if __name__ == "__main__":
    main()
