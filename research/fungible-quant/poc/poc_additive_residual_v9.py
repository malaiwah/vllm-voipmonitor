#!/usr/bin/env python3
"""PoC v9: K4 base + tile-level K5 upgrade — continuously variable 4-5 bits.

Extends v7's tile-level approach to the 4-5 bit range:
  - K4 base (4 bpw) for all tiles
  - Upgrade top-k% most-damaged tiles to K4+2lloyd (5 bpw)
  - Also test: K3 base + tile K4 + tile K5 (full 3-5 bit range)
  - Also test: different tile sizes (16x16, 32x32, 64x64)
  - Also test: vertical (cross-layer) allocation
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

def tile_mixed_k4_to_k5(w_reg, device, tcp, tcpi, qtf, upgrade_frac):
    """K4 base + upgrade top-k% tiles to K4+2lloyd (5 bits).
    Returns: reconstructed weight, effective bits per weight.
    """
    k, n = w_reg.shape
    tiles_n_k = k // 16; tiles_n_n = n // 16; n_tiles = tiles_n_k * tiles_n_n
    
    qk4, tile_mse_k4 = quantize_tilewise_with_mse(w_reg, 4, device, tcp, tcpi, qtf)
    
    # K4 residual
    r4 = w_reg - qk4
    
    # 2-bit Lloyd-Max on the K4 residual (applied to all, but only kept for upgraded tiles)
    lloyd = q2b_lloyd(r4)
    
    # Tile-level improvement from adding 2-bit Lloyd-Max
    # Improvement = tile_mse_k4 - tile_mse_k4_after_lloyd
    # We need per-tile MSE after lloyd
    tile_mse_k5 = torch.zeros_like(tile_mse_k4)
    recon_k5 = qk4 + lloyd
    for ti_k in range(tiles_n_k):
        for ti_n in range(tiles_n_n):
            tile_orig = w_reg[ti_k*16:(ti_k+1)*16, ti_n*16:(ti_n+1)*16]
            tile_recon = recon_k5[ti_k*16:(ti_k+1)*16, ti_n*16:(ti_n+1)*16]
            tile_mse_k5[ti_k, ti_n] = (tile_orig - tile_recon).pow(2).mean()
    
    improvement = tile_mse_k4 - tile_mse_k5
    
    n_upgrade = int(n_tiles * upgrade_frac)
    if n_upgrade == 0:
        return qk4, 4.0 + 1.0 / n_tiles
    if n_upgrade >= n_tiles:
        return recon_k5, 5.0 + 1.0 / n_tiles
    
    _, top_indices = improvement.flatten().topk(n_upgrade)
    
    result = qk4.clone()
    for idx in top_indices:
        ti_k = idx.item() // tiles_n_n
        ti_n = idx.item() % tiles_n_n
        result[ti_k*16:(ti_k+1)*16, ti_n*16:(ti_n+1)*16] = \
            recon_k5[ti_k*16:(ti_k+1)*16, ti_n*16:(ti_n+1)*16]
    
    bitmap_bpw = n_tiles / (k * n)
    eff_bits = 4.0 + upgrade_frac * 1.0 + bitmap_bpw  # 2-bit lloyd only on upgraded tiles
    # Actually: lloyd costs 2 bits/weight on upgraded tiles, 0 on rest
    # eff_bits = 4 + upgrade_frac * 2 + bitmap_bpw
    eff_bits = 4.0 + upgrade_frac * 2.0 + bitmap_bpw
    
    return result, eff_bits

def tile_mixed_3tier_full(w_reg, device, tcp, tcpi, qtf, frac_k4, frac_k5):
    """3-tier: K3 base + some tiles at K4 + some at K5 (K4+2lloyd).
    frac_k4: fraction of tiles upgraded K3→K4
    frac_k5: fraction of tiles upgraded K4→K5 (from the K4 tiles)
    Total bits = 3 + frac_k4 * 1 + frac_k5 * 2 + bitmap
    """
    k, n = w_reg.shape
    tiles_n_k = k // 16; tiles_n_n = n // 16; n_tiles = tiles_n_k * tiles_n_n
    
    qk3, tile_mse_k3 = quantize_tilewise_with_mse(w_reg, 3, device, tcp, tcpi, qtf)
    qk4, tile_mse_k4 = quantize_tilewise_with_mse(w_reg, 4, device, tcp, tcpi, qtf)
    r4 = w_reg - qk4
    lloyd = q2b_lloyd(r4)
    recon_k5 = qk4 + lloyd
    
    # Per-tile improvement K3→K4 and K4→K5
    improvement_k4 = tile_mse_k3 - tile_mse_k4
    
    # Per-tile MSE at K5
    tile_mse_k5 = torch.zeros_like(tile_mse_k4)
    for ti_k in range(tiles_n_k):
        for ti_n in range(tiles_n_n):
            tile_orig = w_reg[ti_k*16:(ti_k+1)*16, ti_n*16:(ti_n+1)*16]
            tile_k5 = recon_k5[ti_k*16:(ti_k+1)*16, ti_n*16:(ti_n+1)*16]
            tile_mse_k5[ti_k, ti_n] = (tile_orig - tile_k5).pow(2).mean()
    improvement_k5 = tile_mse_k4 - tile_mse_k5
    
    # Rank tiles by K3→K4 improvement
    _, sorted_indices = improvement_k4.flatten().sort(descending=True)
    
    n_k4 = int(n_tiles * frac_k4)
    n_k5 = int(n_tiles * frac_k5)
    
    result = qk3.clone()
    k5_indices = sorted_indices[:n_k5]
    k4_indices = sorted_indices[n_k5:n_k5 + n_k4]
    
    # Apply K4
    for idx in torch.cat([k4_indices, k5_indices]):
        ti_k = idx.item() // tiles_n_n
        ti_n = idx.item() % tiles_n_n
        result[ti_k*16:(ti_k+1)*16, ti_n*16:(ti_n+1)*16] = \
            qk4[ti_k*16:(ti_k+1)*16, ti_n*16:(ti_n+1)*16]
    
    # Apply K5 (K4+2lloyd)
    for idx in k5_indices:
        ti_k = idx.item() // tiles_n_n
        ti_n = idx.item() % tiles_n_n
        result[ti_k*16:(ti_k+1)*16, ti_n*16:(ti_n+1)*16] = \
            recon_k5[ti_k*16:(ti_k+1)*16, ti_n*16:(ti_n+1)*16]
    
    bitmap_bpw = 2 * n_tiles / (k * n)
    eff_bits = 3.0 + frac_k4 * 1.0 + frac_k5 * 2.0 + bitmap_bpw
    
    return result, eff_bits

def run_experiment(data_dir, device, ext, ghd, tcp, tcpi, qtf, cbs, layer_indices=[10, 40], max_experts=10):
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
        all_experts = all_experts[:n_experts]
        print(f"  {n_experts} experts, shape=({k},{n})", flush=True)
        
        layer_results = {}
        
        # Baselines
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
        
        layer_results["K3"] = {"mse": k3_avg, "bits": 3.0, "gap": 0.0}
        layer_results["K4"] = {"mse": k4_avg, "bits": 4.0, "gap": 1.0}
        layer_results["K5(K3+2lloyd)"] = {"mse": k5_avg, "bits": 5.0, "gap": 1.99}
        
        # ================================================================
        # Exp 1: K4 base + tile K5 upgrade (4-5 bit range)
        # ================================================================
        print(f"\n  --- Exp 1: K4 + tile K5 upgrade ---", flush=True)
        for frac in [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0]:
            mses = []; bits_list = []
            for ei in range(n_experts):
                w = all_experts[ei].to(device)
                w_reg, _, _ = regularize(w, device, ghd, cbs)
                recon, eff_bits = tile_mixed_k4_to_k5(w_reg, device, tcp, tcpi, qtf, frac)
                mses.append((w_reg - recon).pow(2).mean().item())
                bits_list.append(eff_bits)
                del w, w_reg, recon
                torch.cuda.empty_cache()
            avg_mse = sum(mses) / n_experts
            avg_bits = sum(bits_list) / n_experts
            gap = (k4_avg - avg_mse) / (k4_avg - k5_avg) if k4_avg > k5_avg else 0
            label = f"K4+tile_K5_{frac:.2f}"
            layer_results[label] = {"mse": avg_mse, "bits": avg_bits, "gap": gap}
            print(f"    {label}: MSE={avg_mse:.6e}  bits={avg_bits:.3f}  gap_K4toK5={gap:.1%}", flush=True)
        
        # ================================================================
        # Exp 2: Full 3-tier K3 + tile K4 + tile K5 (3-5 bit range)
        # ================================================================
        print(f"\n  --- Exp 2: Full 3-tier K3+K4+K5 ---", flush=True)
        configs = [
            (0.25, 0.0), (0.50, 0.0), (0.75, 0.0), (1.0, 0.0),
            (1.0, 0.25), (1.0, 0.50), (1.0, 0.75),
            (0.50, 0.25), (0.50, 0.50),
            (0.75, 0.25), (0.75, 0.50),
        ]
        for frac_k4, frac_k5 in configs:
            mses = []; bits_list = []
            for ei in range(n_experts):
                w = all_experts[ei].to(device)
                w_reg, _, _ = regularize(w, device, ghd, cbs)
                recon, eff_bits = tile_mixed_3tier_full(w_reg, device, tcp, tcpi, qtf, frac_k4, frac_k5)
                mses.append((w_reg - recon).pow(2).mean().item())
                bits_list.append(eff_bits)
                del w, w_reg, recon
                torch.cuda.empty_cache()
            avg_mse = sum(mses) / n_experts
            avg_bits = sum(bits_list) / n_experts
            gap = (k3_avg - avg_mse) / (k3_avg - k4_avg) if k3_avg > k4_avg else 0
            label = f"3tier_K4_{frac_k4:.2f}_K5_{frac_k5:.2f}"
            layer_results[label] = {"mse": avg_mse, "bits": avg_bits, "gap": gap}
            print(f"    {label}: MSE={avg_mse:.6e}  bits={avg_bits:.3f}  gap_K3toK4={gap:.1%}", flush=True)
        
        results[f"layer{layer_idx}"] = {
            "k3": k3_avg, "k4": k4_avg, "k5": k5_avg,
            "methods": layer_results, "n_experts": n_experts
        }
    
    return results

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--data-dir", default="/tmp/poc_residual/glm52_data")
    ap.add_argument("--out", default="/tmp/poc_residual/results_v9.json")
    ap.add_argument("--max-experts", type=int, default=10)
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
        
        # Pareto frontier
        print(f"  Pareto frontier:", flush=True)
        bit_buckets = {}
        for label, m in methods.items():
            bucket = round(m["bits"] * 2) / 2
            if bucket not in bit_buckets or m["mse"] < bit_buckets[bucket][1]:
                bit_buckets[bucket] = (label, m["mse"], m["bits"])
        for bucket in sorted(bit_buckets.keys()):
            label, mse, bits = bit_buckets[bucket]
            gap = (r['k3'] - mse) / (r['k3'] - r['k4']) if r['k3'] > r['k4'] else 0
            print(f"    ~{bucket:.1f} bits: {label:35s}  MSE={mse:.6e}  gap={gap:.1%}", flush=True)
    
    Path(args.out).write_text(json.dumps(results, indent=2, default=str))
    print(f"\nResults saved to {args.out}", flush=True)

if __name__ == "__main__":
    main()
