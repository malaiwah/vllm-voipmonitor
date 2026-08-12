#!/usr/bin/env python3
"""PoC v12: Fast proxy-based tier assignment + runtime efficiency analysis.

Instead of computing K3, K4, K5 per tile (expensive at encode time),
use weight statistics as a proxy for tile difficulty:
  - Tile variance (high variance = harder to quantize)
  - Tile kurtosis (heavy tails = harder)
  - Tile max/min ratio (outliers = harder)

Also tests:
  1. Progressive tier encoding (entropy-optimal bitmap)
  2. Tile clustering across experts (shared tier patterns)
  3. Weight magnitude as proxy vs actual MSE
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
    # Compute on CPU to avoid OOM
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

# ---------------------------------------------------------------------------
# Proxy-based tier assignment
# ---------------------------------------------------------------------------

def compute_tile_stats(w_reg, device):
    """Compute per-tile statistics of the regularized weight matrix."""
    k, n = w_reg.shape
    tiles_n_k = k // 16; tiles_n_n = n // 16
    variance = torch.zeros(tiles_n_k, tiles_n_n, device=device)
    kurtosis = torch.zeros(tiles_n_k, tiles_n_n, device=device)
    max_abs = torch.zeros(tiles_n_k, tiles_n_n, device=device)
    
    for tik in range(tiles_n_k):
        for tin in range(tiles_n_n):
            tile = w_reg[tik*16:(tik+1)*16, tin*16:(tin+1)*16]
            variance[tik, tin] = tile.var()
            # Excess kurtosis
            m = tile.mean()
            s = tile.std() + 1e-8
            kurt = ((tile - m) / s).pow(4).mean() - 3
            kurtosis[tik, tin] = kurt
            max_abs[tik, tin] = tile.abs().max()
    
    return variance, kurtosis, max_abs

def proxy_tier_assignment(tile_stats, target_bpw, n_tiles):
    """Assign tiers based on tile statistics (proxy for quantization error).
    Higher variance tiles get higher tiers.
    """
    # Normalize stats to [0, 1]
    stats = tile_stats.flatten()
    stats_norm = (stats - stats.min()) / (stats.max() - stats.min() + 1e-8)
    
    # Determine tier boundaries based on target bpw
    # If target = 3.5: 50% K3, 50% K4
    # If target = 4.0: 100% K4
    # If target = 4.5: 50% K4, 50% K5
    
    if target_bpw <= 4.0:
        frac_k4 = target_bpw - 3.0
        n_k4 = int(frac_k4 * n_tiles)
        # Top variance tiles get K4
        _, top_idx = stats_norm.topk(n_k4)
        tier = [3] * n_tiles
        for idx in top_idx:
            tier[idx.item()] = 4
    else:
        # target > 4.0: all K4, some K5
        frac_k5 = target_bpw - 4.0
        n_k5 = int(frac_k5 * n_tiles)
        tier = [4] * n_tiles
        _, top_idx = stats_norm.topk(n_k5)
        for idx in top_idx:
            tier[idx.item()] = 5
    
    return tier

def tier_assignment_mse(w_reg, tier, qk3, qk4, recon_k5):
    """Compute MSE for a given tier assignment using pre-computed quantizations."""
    k, n = w_reg.shape
    result = qk3.clone()
    tiles_n_n = n // 16
    for i, t in enumerate(tier):
        ti_k = i // tiles_n_n
        ti_n = i % tiles_n_n
        rs, re = ti_k * 16, (ti_k + 1) * 16
        cs, ce = ti_n * 16, (ti_n + 1) * 16
        if t == 3: result[rs:re, cs:ce] = qk3[rs:re, cs:ce]
        elif t == 4: result[rs:re, cs:ce] = qk4[rs:re, cs:ce]
        elif t == 5: result[rs:re, cs:ce] = recon_k5[rs:re, cs:ce]
    mse = (w_reg - result).pow(2).mean().item()
    avg_bits = sum(t * 256 for t in tier) / (k * n)
    return mse, avg_bits

def run_experiment(data_dir, device, ext, ghd, tcp, tcpi, qtf, cbs, layer_indices=[10], max_experts=5):
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
        n_tiles = (k // 16) * (n // 16)
        all_experts = all_experts[:n_experts]
        print(f"  {n_experts} experts, {n_tiles} tiles each", flush=True)
        
        layer_results = {}
        
        # Baselines — compute K3, K4, K5 separately to manage memory
        k3_mses = []; k4_mses = []; k5_mses = []
        for ei in range(n_experts):
            w = all_experts[ei].to(device)
            w_reg, _, _ = regularize(w, device, ghd, cbs)
            del w
            # K3
            qk3 = quantize_trellis(w_reg, 3, device, tcp, tcpi, qtf)
            k3_mses.append((w_reg - qk3).pow(2).mean().item())
            del qk3; torch.cuda.empty_cache()
            # K4
            qk4 = quantize_trellis(w_reg, 4, device, tcp, tcpi, qtf)
            k4_mses.append((w_reg - qk4).pow(2).mean().item())
            r4 = w_reg - qk4; del qk4
            # K5 (K4 + 2-bit Lloyd): error = r4 - lloyd
            lloyd = q2b_lloyd(r4)
            k5_mses.append((r4 - lloyd).pow(2).mean().item())
            del w_reg, r4, lloyd
            torch.cuda.empty_cache()
        k3_avg = sum(k3_mses) / n_experts
        k4_avg = sum(k4_mses) / n_experts
        k5_avg = sum(k5_mses) / n_experts
        print(f"  K3={k3_avg:.6e} K4={k4_avg:.6e} K5={k5_avg:.6e}", flush=True)
        
        layer_results["K3"] = {"mse": k3_avg, "bits": 3.0}
        layer_results["K4"] = {"mse": k4_avg, "bits": 4.0}
        layer_results["K5"] = {"mse": k5_avg, "bits": 5.0}
        
        # ================================================================
        # Proxy-based vs MSE-based tier assignment
        # ================================================================
        print(f"\n  --- Proxy (variance) vs MSE-based tier assignment ---", flush=True)
        
        for target_bpw in [3.5, 4.0, 4.5]:
            mses_proxy = []; mses_mse = []
            bits_proxy = []; bits_mse = []
            
            for ei in range(n_experts):
                w = all_experts[ei].to(device)
                w_reg, _, _ = regularize(w, device, ghd, cbs)
                
                # MSE-based (ground truth)
                qk3, tmse_k3 = quantize_tilewise_with_mse(w_reg, 3, device, tcp, tcpi, qtf)
                qk4, tmse_k4 = quantize_tilewise_with_mse(w_reg, 4, device, tcp, tcpi, qtf)
                r4 = w_reg - qk4; lloyd = q2b_lloyd(r4)
                recon_k5 = qk4 + lloyd
                tiles_n_k = k // 16; tiles_n_n = n // 16
                tmse_k5 = torch.zeros_like(tmse_k4)
                for tik in range(tiles_n_k):
                    for tin in range(tiles_n_n):
                        orig = w_reg[tik*16:(tik+1)*16, tin*16:(tin+1)*16]
                        k5 = recon_k5[tik*16:(tik+1)*16, tin*16:(tin+1)*16]
                        tmse_k5[tik, tin] = (orig - k5).pow(2).mean()
                
                # MSE-based tier assignment (greedy by benefit per bit)
                if target_bpw <= 4.0:
                    frac = target_bpw - 3.0
                    n_upgrade = int(frac * n_tiles)
                    improvement = (tmse_k3 - tmse_k4).flatten()
                    _, top_idx = improvement.topk(n_upgrade)
                    tier_mse = [3] * n_tiles
                    for idx in top_idx: tier_mse[idx.item()] = 4
                else:
                    frac = target_bpw - 4.0
                    n_upgrade = int(frac * n_tiles)
                    improvement = (tmse_k4 - tmse_k5).flatten()
                    _, top_idx = improvement.topk(n_upgrade)
                    tier_mse = [4] * n_tiles
                    for idx in top_idx: tier_mse[idx.item()] = 5
                
                mse_mse, bits_m = tier_assignment_mse(w_reg, tier_mse, qk3, qk4, recon_k5)
                mses_mse.append(mse_mse); bits_mse.append(bits_m)
                
                variance, kurtosis, max_abs = compute_tile_stats(w_reg, device)
                tier_proxy = proxy_tier_assignment(variance, target_bpw, n_tiles)
                mse_proxy, bits_p = tier_assignment_mse(w_reg, tier_proxy, qk3, qk4, recon_k5)
                mses_proxy.append(mse_proxy); bits_proxy.append(bits_p)
                
                del w, w_reg, qk3, qk4, r4, lloyd, recon_k5
                torch.cuda.empty_cache()
            
            avg_mse_mse = sum(mses_mse) / n_experts
            avg_mse_proxy = sum(mses_proxy) / n_experts
            avg_bits = sum(bits_mse) / n_experts
            gap_mse = (k3_avg - avg_mse_mse) / (k3_avg - k4_avg) if k3_avg > k4_avg else 0
            gap_proxy = (k3_avg - avg_mse_proxy) / (k3_avg - k4_avg) if k3_avg > k4_avg else 0
            proxy_loss = (avg_mse_proxy - avg_mse_mse) / avg_mse_mse * 100
            
            label_mse = f"mse_based_{target_bpw:.1f}bpw"
            label_proxy = f"proxy_var_{target_bpw:.1f}bpw"
            layer_results[label_mse] = {"mse": avg_mse_mse, "bits": avg_bits, "gap": gap_mse}
            layer_results[label_proxy] = {"mse": avg_mse_proxy, "bits": avg_bits, "gap": gap_proxy,
                                           "proxy_loss_pct": proxy_loss}
            print(f"    {label_mse}: MSE={avg_mse_mse:.6e}  gap={gap_mse:.1%}", flush=True)
            print(f"    {label_proxy}: MSE={avg_mse_proxy:.6e}  gap={gap_proxy:.1%}  "
                  f"proxy_loss={proxy_loss:+.2f}%", flush=True)
        
        # ================================================================
        # Cross-projection tile correlation
        # ================================================================
        print(f"\n  --- Cross-projection tile variance correlation ---", flush=True)
        for proj in ["gate_proj", "up_proj", "down_proj"]:
            pf = data_dir / f"layer{layer_idx}_all_{proj}.pt"
            if not pf.exists(): continue
            experts = torch.load(pf, map_location="cpu")
            w = experts[0].to(device)
            w_reg, _, _ = regularize(w, device, ghd, cbs)
            var, _, _ = compute_tile_stats(w_reg, device)
            print(f"    {proj}: tile var range=[{var.min():.4e}, {var.max():.4e}], "
                  f"CV={var.std()/var.mean():.4f}", flush=True)
            del w, w_reg, var
            torch.cuda.empty_cache()
        
        results[f"layer{layer_idx}"] = {
            "n_experts": n_experts, "n_tiles": n_tiles,
            "k3": k3_avg, "k4": k4_avg, "k5": k5_avg,
            "methods": layer_results
        }
    
    return results

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--data-dir", default="/tmp/poc_residual/glm52_data")
    ap.add_argument("--out", default="/tmp/poc_residual/results_v12.json")
    ap.add_argument("--max-experts", type=int, default=5)
    args = ap.parse_args()
    dev = torch.device(args.device)
    print(f"Device: {dev}  GPU: {torch.cuda.get_device_name(0)}", flush=True)
    ext, ghd, tcp, tcpi, qtf, cbs = _bootstrap()
    
    results = run_experiment(Path(args.data_dir), dev, ext, ghd, tcp, tcpi, qtf, cbs,
                             max_experts=args.max_experts)
    
    print(f"\n{'='*70}\nSUMMARY\n{'='*70}", flush=True)
    for layer_key in sorted(results.keys()):
        r = results[layer_key]
        print(f"\n{layer_key}: K3={r['k3']:.4e} K4={r['k4']:.4e} K5={r['k5']:.4e}", flush=True)
        methods = r["methods"]
        for label in sorted(methods.keys(), key=lambda x: methods[x].get("mse", 0)):
            m = methods[label]
            extra = f"  proxy_loss={m.get('proxy_loss_pct', 0):+.2f}%" if "proxy" in label else ""
            print(f"    {label:30s}: MSE={m['mse']:.6e}  bits={m['bits']:.2f}  gap={m.get('gap', 0):.1%}{extra}", flush=True)
    
    Path(args.out).write_text(json.dumps(results, indent=2, default=str))
    print(f"\nResults saved to {args.out}", flush=True)

if __name__ == "__main__":
    main()
