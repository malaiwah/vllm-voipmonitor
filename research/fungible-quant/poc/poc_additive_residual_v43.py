#!/usr/bin/env python3
"""PoC v43: Definitive tile-level MSRT Pareto — mixing MSRT tiers per-tile.

Builds the final Pareto frontier by mixing MSRT tiers at the tile level.
Uses 10 experts, 0.5-bit steps from 2.0 to 10.0 bpw.

MSRT tiers (from v41-v42):
  K2 (2), K3 (3), K4 (4), K2+K3trsc (5), K2+K1+K3trsc (6),
  K2+K1+K4trsc (7), K2+K1+K2+K3trsc (8), K2+K1+K1+K2+K3trsc (9),
  K2+K1+K1+K1+K2+K3trsc (10)
"""
from __future__ import annotations
import argparse, json, math, os, sys, types, importlib.util, time
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

def tile_mse(w_reg, recon, k, n):
    tnk, tnn = k // 16, n // 16
    diff = (w_reg - recon).pow(2)
    return diff.view(tnk, 16, tnn, 16).mean(dim=(1, 3))

def build_msrt_tiers(w_reg, device, tcp, tcpi, qtf, cbs_scale, k, n):
    """Build all MSRT tier reconstructions for one expert."""
    qk2 = quantize_trellis_raw(w_reg, 2, device, tcp, tcpi, qtf)
    qk3 = quantize_trellis_raw(w_reg, 3, device, tcp, tcpi, qtf)
    qk4 = quantize_trellis_raw(w_reg, 4, device, tcp, tcpi, qtf)
    r2 = w_reg - qk2

    tiers = {}
    tiers["K2"] = (qk2, 2.0)
    tiers["K3"] = (qk3, 3.0)
    tiers["K4"] = (qk4, 4.0)

    # 5bpw: K2+K3trsc
    tiers["5bpw"] = (rescaled_trellis(qk2, r2, 3, device, tcp, tcpi, qtf, cbs_scale), 5.0)

    # 6bpw: K2+K1trsc+K3trsc
    s1 = rescaled_trellis(qk2, r2, 1, device, tcp, tcpi, qtf, cbs_scale)
    r1 = w_reg - s1
    tiers["6bpw"] = (rescaled_trellis(s1, r1, 3, device, tcp, tcpi, qtf, cbs_scale), 6.0)

    # 7bpw: K2+K1trsc+K4trsc
    s2 = rescaled_trellis(qk2, r2, 1, device, tcp, tcpi, qtf, cbs_scale)
    r2_s = w_reg - s2
    tiers["7bpw"] = (rescaled_trellis(s2, r2_s, 4, device, tcp, tcpi, qtf, cbs_scale), 7.0)

    # 8bpw: K2+K1+K2+K3trsc
    s3a = rescaled_trellis(qk2, r2, 1, device, tcp, tcpi, qtf, cbs_scale)
    r3a = w_reg - s3a
    s3b = rescaled_trellis(s3a, r3a, 2, device, tcp, tcpi, qtf, cbs_scale)
    r3b = w_reg - s3b
    tiers["8bpw"] = (rescaled_trellis(s3b, r3b, 3, device, tcp, tcpi, qtf, cbs_scale), 8.0)

    # 9bpw: K2+K1+K1+K2+K3trsc
    s4a = rescaled_trellis(qk2, r2, 1, device, tcp, tcpi, qtf, cbs_scale)
    r4a = w_reg - s4a
    s4b = rescaled_trellis(s4a, r4a, 1, device, tcp, tcpi, qtf, cbs_scale)
    r4b = w_reg - s4b
    s4c = rescaled_trellis(s4b, r4b, 2, device, tcp, tcpi, qtf, cbs_scale)
    r4c = w_reg - s4c
    tiers["9bpw"] = (rescaled_trellis(s4c, r4c, 3, device, tcp, tcpi, qtf, cbs_scale), 9.0)

    # 10bpw: K2+K1+K1+K1+K2+K3trsc
    s5a = rescaled_trellis(qk2, r2, 1, device, tcp, tcpi, qtf, cbs_scale)
    r5a = w_reg - s5a
    s5b = rescaled_trellis(s5a, r5a, 1, device, tcp, tcpi, qtf, cbs_scale)
    r5b = w_reg - s5b
    s5c = rescaled_trellis(s5b, r5b, 1, device, tcp, tcpi, qtf, cbs_scale)
    r5c = w_reg - s5c
    s5d = rescaled_trellis(s5c, r5c, 2, device, tcp, tcpi, qtf, cbs_scale)
    r5d = w_reg - s5d
    tiers["10bpw"] = (rescaled_trellis(s5d, r5d, 3, device, tcp, tcpi, qtf, cbs_scale), 10.0)

    # Cleanup intermediate tensors
    del r2
    return tiers

def run_experiment(data_dir, device, ext, ghd, tcp, tcpi, qtf, cbs_scale):
    results = {}
    f = data_dir / f"layer10_all_gate_proj.pt"
    if not f.exists(): return results
    all_experts = torch.load(f, map_location="cpu")
    n_experts = min(10, all_experts.shape[0])
    k, n = all_experts.shape[1], all_experts.shape[2]
    print(f"  {n_experts} experts, {k}x{n}", flush=True)

    # Collect tile MSEs for all MSRT tiers across experts
    tier_data = {}
    tier_order = ["K2", "K3", "K4", "5bpw", "6bpw", "7bpw", "8bpw", "9bpw", "10bpw"]

    for ei in range(n_experts):
        print(f"  Expert {ei}...", flush=True)
        w = all_experts[ei].to(device)
        w_reg, _, _ = regularize(w, device, ghd, cbs_scale)
        del w

        tiers = build_msrt_tiers(w_reg, device, tcp, tcpi, qtf, cbs_scale, k, n)

        for name in tier_order:
            if name not in tiers: continue
            recon, bpw = tiers[name]
            tmse = tile_mse(w_reg, recon, k, n)
            if name not in tier_data:
                tier_data[name] = {"bpw": bpw, "tile_mses": []}
            tier_data[name]["tile_mses"].append(tmse.cpu())
            del recon, tmse

        # Cleanup
        for name in list(tiers.keys()):
            del tiers[name]
        del w_reg, tiers
        torch.cuda.empty_cache()

    avg_tmse = {name: (sum(d["tile_mses"]) / len(d["tile_mses"])).to(device) for name, d in tier_data.items()}
    n_tiles = (k // 16) * (n // 16)

    # Print tier summary
    print(f"\n  {'Tier':<15} {'bpw':>5} {'avg MSE':>12}", flush=True)
    for name in tier_order:
        if name not in avg_tmse: continue
        d = tier_data[name]
        avg_mse = avg_tmse[name].mean().item()
        print(f"  {name:<15} {d['bpw']:>5.0f} {avg_mse:>12.4e}", flush=True)

    # Build upgrades: for adjacent tier pairs, compute per-tile benefit
    available = [t for t in tier_order if t in avg_tmse]
    upgrades = []
    for i in range(len(available) - 1):
        lo, hi = available[i], available[i + 1]
        bit_cost = tier_data[hi]["bpw"] - tier_data[lo]["bpw"]
        benefit = (avg_tmse[lo] - avg_tmse[hi]).flatten()
        for ti in range(n_tiles):
            upgrades.append((benefit[ti].item(), ti, lo, hi, bit_cost))
    upgrades.sort(key=lambda x: -x[0])

    # Build Pareto
    print(f"\n  Tile-level MSRT Pareto (10 experts, 0.5-bit steps):", flush=True)
    pareto = []
    for target_10 in range(20, 101, 5):  # 2.0 to 10.0 in 0.5 steps
        target_bpw = target_10 / 10.0
        current_tier = [available[0]] * n_tiles
        current_bits = tier_data[available[0]]["bpw"] * n_tiles
        target_bits = target_bpw * n_tiles

        for benefit, tile_idx, lo, hi, bit_cost in upgrades:
            if current_bits + bit_cost > target_bits + 1e-6: continue
            if current_tier[tile_idx] != lo: continue
            if benefit <= 0: continue
            lo_idx = available.index(lo); hi_idx = available.index(hi)
            if lo_idx + 1 != hi_idx: continue
            current_tier[tile_idx] = hi
            current_bits += bit_cost

        total_mse = sum(avg_tmse[current_tier[ti]][ti // (n // 16), ti % (n // 16)].item() for ti in range(n_tiles))
        avg_mse = total_mse / n_tiles
        actual_bpw = current_bits / n_tiles
        pareto.append({"target_bpw": target_bpw, "actual_bpw": actual_bpw, "mse": avg_mse})
        print(f"    {target_bpw:.1f}  actual={actual_bpw:.3f}  MSE={avg_mse:.6e}", flush=True)

    # Compare with uniform MSRT (all tiles at same tier)
    print(f"\n  Uniform vs tile-level MSRT:", flush=True)
    print(f"  {'bpw':>5} {'uniform MSE':>12} {'tile-level MSE':>14} {'improvement':>12}", flush=True)
    for name in tier_order:
        if name not in avg_tmse: continue
        d = tier_data[name]
        bpw = d["bpw"]
        uniform_mse = avg_tmse[name].mean().item()
        # Find tile-level at same bpw
        for p in pareto:
            if abs(p["target_bpw"] - bpw) < 0.01:
                p_mse = p["mse"]
                imp = (1 - p_mse / uniform_mse) * 100 if uniform_mse > 0 else 0
                print(f"  {bpw:>5.0f} {uniform_mse:>12.4e} {p_mse:>14.4e} {imp:>11.1f}%", flush=True)
                break

    results["pareto"] = pareto
    results["tiers"] = {name: {"bpw": d["bpw"], "avg_mse": avg_tmse[name].mean().item()}
                        for name, d in tier_data.items() if name in avg_tmse}
    return results

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--data-dir", default="/tmp/poc_residual/glm52_data")
    ap.add_argument("--out", default="/tmp/poc_residual/results_v43.json")
    args = ap.parse_args()
    dev = torch.device(args.device)
    print(f"Device: {dev}  GPU: {torch.cuda.get_device_name(0)}", flush=True)
    ext, ghd, tcp, tcpi, qtf, cbs_scale = _bootstrap()
    results = run_experiment(Path(args.data_dir), dev, ext, ghd, tcp, tcpi, qtf, cbs_scale)
    Path(args.out).write_text(json.dumps(results, indent=2, default=str))
    print(f"\nResults saved to {args.out}", flush=True)

if __name__ == "__main__":
    main()
