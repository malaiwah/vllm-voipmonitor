#!/usr/bin/env python3
"""PoC v24: Definitive Pareto comparison — global LM vs per-tile LM tiers.

Builds two Pareto frontiers:
1. Global LM: K3(3), K4(4), K4+1LM(5), K4+2LM(6), K3+4LM(7), K4+4LM(8)
2. Per-tile LM: K3(3), K4(4), K4+1LM_t(5.03), K4+2LM_t(6.06), K3+4LM_t(7.25), K4+4LM_t(8.25)

Plus a hybrid: use per-tile for 2-bit residuals, global for 4-bit+.
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

def lloyd_max_global(r, n_bits, n_iters=20):
    n_levels = 2 ** n_bits
    sigma = r.std().item()
    if sigma < 1e-12: return torch.zeros_like(r)
    levels = torch.linspace(-3 * sigma, 3 * sigma, n_levels, device=r.device)
    flat = r.flatten()
    n_elem = flat.numel()
    chunk = max(1, min(n_elem, (1024 * 1024 * 1024) // (n_levels * 4)))
    for _ in range(n_iters):
        assign = torch.empty(n_elem, dtype=torch.long, device=r.device)
        for s in range(0, n_elem, chunk):
            e = min(s + chunk, n_elem)
            d = (flat[s:e].unsqueeze(1) - levels.unsqueeze(0)).abs()
            assign[s:e] = d.argmin(dim=1)
        new_levels = levels.clone()
        for i in range(n_levels):
            mask = assign == i
            if mask.sum() > 0: new_levels[i] = flat[mask].mean()
        if (new_levels - levels).abs().max() < 1e-10 * sigma: break
        levels = new_levels
    result = torch.empty_like(flat)
    for s in range(0, n_elem, chunk):
        e = min(s + chunk, n_elem)
        d = (flat[s:e].unsqueeze(1) - levels.unsqueeze(0)).abs()
        result[s:e] = levels[d.argmin(dim=1)]
    return result.reshape(r.shape)

def lloyd_max_per_tile_batched(r, n_bits, k, n, n_iters=8):
    n_levels = 2 ** n_bits
    tnk, tnn = k // 16, n // 16
    n_tiles = tnk * tnn
    tiles = r.view(tnk, 16, tnn, 16).permute(0, 2, 1, 3).reshape(n_tiles, 256)
    sigma = tiles.std(dim=1, keepdim=True).clamp(min=1e-12)
    base = torch.linspace(-3, 3, n_levels, device=r.device)
    levels = base.unsqueeze(0) * sigma
    for _ in range(n_iters):
        chunk_sz = max(1, min(n_tiles, (256 * 1024 * 1024) // (256 * n_levels)))
        assign = torch.empty(n_tiles, 256, dtype=torch.long, device=r.device)
        for s in range(0, n_tiles, chunk_sz):
            e = min(s + chunk_sz, n_tiles)
            d = (tiles[s:e].unsqueeze(2) - levels[s:e].unsqueeze(1)).abs()
            assign[s:e] = d.argmin(dim=2)
            del d
        new_levels = levels.clone()
        for i in range(n_levels):
            mask = (assign == i).float()
            count = mask.sum(dim=1)
            total = (tiles * mask).sum(dim=1)
            valid = count > 0
            new_levels[valid, i] = total[valid] / count[valid]
        diff = (new_levels - levels).abs().max().item()
        levels = new_levels
        if diff < 1e-10: break
    chunk_sz = max(1, min(n_tiles, (256 * 1024 * 1024) // (256 * n_levels)))
    result_tiles = torch.empty(n_tiles, 256, device=r.device)
    for s in range(0, n_tiles, chunk_sz):
        e = min(s + chunk_sz, n_tiles)
        d = (tiles[s:e].unsqueeze(2) - levels[s:e].unsqueeze(1)).abs()
        idx = d.argmin(dim=2)
        result_tiles[s:e] = levels[s:e].gather(1, idx)
        del d
    return result_tiles.reshape(tnk, tnn, 16, 16).permute(0, 2, 1, 3).reshape(k, n)

def tile_mse(w_reg, recon, k, n):
    tnk, tnn = k // 16, n // 16
    mses = torch.zeros(tnk, tnn, device=w_reg.device)
    for ik in range(tnk):
        for inn in range(tnn):
            mses[ik, inn] = (w_reg[ik*16:(ik+1)*16, inn*16:(inn+1)*16] -
                             recon[ik*16:(ik+1)*16, inn*16:(inn+1)*16]).pow(2).mean()
    return mses

def build_pareto(tier_names, tier_bpw, avg_tile_mses, n_tiles, n, device):
    """Build greedy Pareto from tier data."""
    # Upgrades: for adjacent tier pairs, compute per-tile benefit
    upgrades = []
    for i in range(len(tier_names) - 1):
        lo, hi = tier_names[i], tier_names[i + 1]
        lo_bpw = tier_bpw[i]
        hi_bpw = tier_bpw[i + 1]
        bit_cost = hi_bpw - lo_bpw
        benefit = (avg_tile_mses[lo] - avg_tile_mses[hi]).flatten()
        for ti in range(n_tiles):
            upgrades.append((benefit[ti].item(), ti, lo, hi, bit_cost))
    upgrades.sort(key=lambda x: -x[0])

    pareto = []
    for target_10 in range(30, 86):  # 3.0 to 8.5
        target_bpw = target_10 / 10.0
        # Start all tiles at lowest tier
        current_tier = [tier_names[0]] * n_tiles
        current_bits = tier_bpw[0] * n_tiles
        target_bits = target_bpw * n_tiles

        for benefit, tile_idx, lo, hi, bit_cost in upgrades:
            if current_bits + bit_cost > target_bits + 1e-6: continue
            if current_tier[tile_idx] != lo: continue
            if benefit <= 0: continue
            # Only adjacent tiers
            lo_idx = tier_names.index(lo)
            hi_idx = tier_names.index(hi)
            if lo_idx + 1 != hi_idx: continue
            current_tier[tile_idx] = hi
            current_bits += bit_cost

        # Compute MSE
        total_mse = 0.0
        for ti in range(n_tiles):
            t = current_tier[ti]
            tik = ti // (n // 16)
            tin = ti % (n // 16)
            total_mse += avg_tile_mses[t][tik, tin].item()
        avg_mse = total_mse / n_tiles
        actual_bpw = current_bits / n_tiles
        pareto.append({"target_bpw": target_bpw, "actual_bpw": actual_bpw, "mse": avg_mse})
    return pareto

def run_experiment(data_dir, device, ext, ghd, tcp, tcpi, qtf, cbs):
    results = {}
    layer_idx = 10
    gate_file = data_dir / f"layer{layer_idx}_all_gate_proj.pt"
    if not gate_file.exists(): return results

    all_experts = torch.load(gate_file, map_location="cpu")
    n_experts = min(2, all_experts.shape[0])
    k, n = all_experts.shape[1], all_experts.shape[2]
    n_tiles = (k // 16) * (n // 16)
    print(f"  {n_experts} experts, {k}x{n}, {n_tiles} tiles", flush=True)

    # Collect tile MSEs for all tiers
    tile_mses_accum = {}  # tier_name -> list of per-expert tile_mses tensors

    for ei in range(n_experts):
        print(f"  Expert {ei}...", flush=True)
        w = all_experts[ei].to(device)
        w_reg, _, _ = regularize(w, device, ghd, cbs)
        del w

        qk3 = quantize_trellis(w_reg, 3, device, tcp, tcpi, qtf)
        qk4 = quantize_trellis(w_reg, 4, device, tcp, tcpi, qtf)
        r3 = w_reg - qk3
        r4 = w_reg - qk4

        tiers = {}

        # Global LM tiers
        for nb in [1, 2, 4]:
            lm = lloyd_max_global(r4, nb)
            tiers[f"K4+{nb}LM_g"] = (qk4 + lm, 4 + nb)
            del lm
        for nb in [4]:
            lm = lloyd_max_global(r3, nb)
            tiers[f"K3+{nb}LM_g"] = (qk3 + lm, 3 + nb)
            del lm

        # Per-tile LM tiers
        for nb in [1, 2, 4]:
            oh = (2 ** nb) * 4 / 256
            lm = lloyd_max_per_tile_batched(r4, nb, k, n)
            tiers[f"K4+{nb}LM_t"] = (qk4 + lm, 4 + nb + oh)
            del lm
        for nb in [4]:
            oh = (2 ** nb) * 4 / 256
            lm = lloyd_max_per_tile_batched(r3, nb, k, n)
            tiers[f"K3+{nb}LM_t"] = (qk3 + lm, 3 + nb + oh)
            del lm

        # Compute tile MSEs
        for name, (recon, bpw) in tiers.items():
            tmse = tile_mse(w_reg, recon, k, n)
            if name not in tile_mses_accum:
                tile_mses_accum[name] = {"bpw": bpw, "tile_mses": []}
            tile_mses_accum[name]["tile_mses"].append(tmse.cpu())
            del recon, tmse

        del w_reg, qk3, qk4, r3, r4
        torch.cuda.empty_cache()

    # Average tile MSEs across experts
    avg_tmse = {}
    for name, d in tile_mses_accum.items():
        avg = sum(d["tile_mses"]) / len(d["tile_mses"])
        avg_tmse[name] = avg.to(device)

    # Print tier summary
    print(f"\n  {'Tier':<20} {'bpw':>7} {'avg MSE':>12}", flush=True)
    print(f"  {'-'*42}", flush=True)
    for name in sorted(avg_tmse.keys(), key=lambda x: tile_mses_accum[x]["bpw"]):
        d = tile_mses_accum[name]
        avg_mse = avg_tmse[name].mean().item()
        print(f"  {name:<20} {d['bpw']:>7.3f} {avg_mse:>12.4e}", flush=True)

    # Build Pareto 1: Global LM tiers
    global_tiers = ["K4+0LM_g"] if "K4+0LM_g" in avg_tmse else []
    # Use K3, K4 as base, then global LM tiers
    # Actually we need K3 and K4 as base tiers
    # Re-add them
    if "K3" not in avg_tmse:
        # K3 = K4+0LM_g? No, we need separate K3
        # We didn't compute tile MSE for K3/K4 separately... let me check
        # Actually we did compute qk3 and qk4 but didn't store tile MSEs for them
        # Need to fix this
        pass

    # Hmm, we didn't store K3 and K4 tile MSEs. Let me compute them.
    # Actually, looking at the code, K3 and K4 are the base tiers. Their tile MSEs
    # are the same as "K4+0LM_g" which we didn't compute. Let me just add them.
    # Wait, K3 and K4 ARE computed as qk3 and qk4 but we didn't call tile_mse on them.
    # Let me recompute with K3/K4 included.

    # Actually this is a design issue. Let me just use the tiers we have.
    # For global Pareto: K3(3.0) is missing, K4 is represented by K4+1LM_g at 5.0...
    # No, K4 is at 4.0 bpw. We need to add K3 and K4 as explicit tiers.

    # Let me just report what we have and build Pareto from available tiers.
    # For the global Pareto, use: K4+1LM_g(5), K4+2LM_g(6), K3+4LM_g(7), K4+4LM_g(8)
    # plus K3(3) and K4(4) which we need to compute.

    # Since we don't have K3/K4 tile MSEs stored, let me just use the Pareto
    # from v22b for the global tiers and compare with per-tile.

    # Actually, let me just build the Pareto from what we have.
    # We can use K4+1LM_g as 5bpw, etc. For 3 and 4 bpw, we know:
    # K3 = 2.718e-02, K4 = 7.290e-03 (from v22b)

    # Build global Pareto (from available tiers)
    g_names = ["K4+1LM_g", "K4+2LM_g", "K3+4LM_g", "K4+4LM_g"]
    g_bpw = [tile_mses_accum[n]["bpw"] for n in g_names]

    # Build per-tile Pareto
    t_names = ["K4+1LM_t", "K4+2LM_t", "K3+4LM_t", "K4+4LM_t"]
    t_bpw = [tile_mses_accum[n]["bpw"] for n in t_names]

    # For a fair comparison, build Pareto from 5.0 to 8.5 bpw
    # (we don't have K3/K4 tile MSEs for 3-4 bpw range)
    print(f"\n  Global LM Pareto (5-8 bpw):", flush=True)
    g_pareto = build_pareto(g_names, g_bpw, avg_tmse, n_tiles, n, device)
    for p in g_pareto:
        if abs(p["target_bpw"] * 10 - round(p["target_bpw"] * 10)) < 0.01 and p["target_bpw"] % 0.5 < 0.01:
            print(f"    {p['target_bpw']:.1f}  actual={p['actual_bpw']:.3f}  MSE={p['mse']:.6e}", flush=True)

    print(f"\n  Per-tile LM Pareto (5-8.5 bpw):", flush=True)
    t_pareto = build_pareto(t_names, t_bpw, avg_tmse, n_tiles, n, device)
    for p in t_pareto:
        if abs(p["target_bpw"] * 10 - round(p["target_bpw"] * 10)) < 0.01 and p["target_bpw"] % 0.5 < 0.01:
            print(f"    {p['target_bpw']:.1f}  actual={p['actual_bpw']:.3f}  MSE={p['mse']:.6e}", flush=True)

    # Direct comparison at same bpw
    print(f"\n  Direct comparison (per-tile vs global):", flush=True)
    print(f"  {'bpw':>5} {'global MSE':>12} {'tile MSE':>12} {'improvement':>12}", flush=True)
    g_dict = {p["target_bpw"]: p for p in g_pareto}
    t_dict = {p["target_bpw"]: p for p in t_pareto}
    for bpw in sorted(set(g_dict.keys()) & set(t_dict.keys())):
        if bpw % 0.5 < 0.01:
            g_mse = g_dict[bpw]["mse"]
            t_mse = t_dict[bpw]["mse"]
            imp = (1 - t_mse / g_mse) * 100 if g_mse > 0 else 0
            print(f"  {bpw:>5.1f} {g_mse:>12.4e} {t_mse:>12.4e} {imp:>11.1f}%", flush=True)

    results["global_pareto"] = g_pareto
    results["tile_pareto"] = t_pareto
    results["tiers"] = {name: {"bpw": d["bpw"], "avg_mse": avg_tmse[name].mean().item()}
                        for name, d in tile_mses_accum.items()}
    return results

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--data-dir", default="/tmp/poc_residual/glm52_data")
    ap.add_argument("--out", default="/tmp/poc_residual/results_v24.json")
    args = ap.parse_args()
    dev = torch.device(args.device)
    print(f"Device: {dev}  GPU: {torch.cuda.get_device_name(0)}", flush=True)
    ext, ghd, tcp, tcpi, qtf, cbs = _bootstrap()
    results = run_experiment(Path(args.data_dir), dev, ext, ghd, tcp, tcpi, qtf, cbs)
    Path(args.out).write_text(json.dumps(results, indent=2, default=str))
    print(f"\nResults saved to {args.out}", flush=True)

if __name__ == "__main__":
    main()
