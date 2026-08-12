#!/usr/bin/env python3
"""PoC v39: Multi-stage rescaled trellis + K2+K6trsc + per-tile rescaling.

Three new ideas:
1. Two-stage rescaled trellis: K2 + K3trsc on residual, then K2trsc on 2nd residual.
   Total = 2+3+2 = 7 bpw. Compare with K2+K5trsc (7 bpw).
   Tests whether successive trellis refinement beats single large trellis.

2. K2+K6trsc (8 bpw): Does rescaled trellis beat LM at 8 bpw too?
   v37 showed LM wins at 8 bpw, but we only tested up to K5trsc.

3. Per-tile rescaling: Instead of global RMS, scale each 16×16 tile to |cbs|.
   Tests whether per-tile adaptation improves the rescaled trellis.
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
    """Global rescaling: scale residual RMS to |cbs|, quantize, scale back."""
    residual_rms = residual.square().mean().sqrt().item()
    if residual_rms < 1e-12: return base_q
    scale = abs(cbs) / residual_rms
    scaled = residual * scale
    quant = quantize_trellis_raw(scaled, K_res, device, tcp, tcpi, qtf)
    return base_q + quant / scale

def per_tile_rescaled_trellis(base_q, residual, K_res, k, n, device, tcp, tcpi, qtf, cbs):
    """Per-tile rescaling: scale each 16×16 tile to |cbs|, quantize, scale back."""
    tnk, tnn = k // 16, n // 16
    result = torch.zeros_like(residual)
    target_rms = abs(cbs)
    qa = {"K": K_res, "mcg": True}
    perm = tcp(device); perm_i = tcpi(device)

    # Process in blocks of 16 rows (trellis needs 16×N)
    for bi in range(0, k, 16):
        tik_start = bi // 16
        for tin in range(tnn):
            rs, re = bi, bi + 16
            cs, ce = tin * 16, (tin + 1) * 16
            tile = residual[rs:re, cs:ce]
            tile_rms = tile.square().mean().sqrt().item()
            if tile_rms < 1e-12:
                continue
            scale = target_rms / tile_rms
            scaled_tile = tile * scale
            # Quantize single tile: reshape to (1, 256), permute, quantize, unpermute
            t = scaled_tile.reshape(1, 256)[:, perm].contiguous()
            quant_t, _ = qtf(t, qa)
            quant_t = quant_t[:, perm_i].reshape(16, 16)
            result[rs:re, cs:ce] = quant_t / scale

    return base_q + result

def get_tiles(r, k, n):
    tnk, tnn = k // 16, n // 16
    return r.view(tnk, 16, tnn, 16).permute(0, 2, 1, 3).reshape(tnk * tnn, 256)

def tiles_to_matrix(tiles, k, n):
    tnk, tnn = k // 16, n // 16
    return tiles.reshape(tnk, tnn, 16, 16).permute(0, 2, 1, 3).reshape(k, n)

def cluster_by_sigma(tiles, n_clusters, device):
    n_tiles = tiles.shape[0]
    sigmas = tiles.std(dim=1).clamp(min=1e-12)
    sorted_sigmas, sort_idx = sigmas.sort()
    cluster_size = n_tiles // n_clusters
    cluster_id = torch.zeros(n_tiles, dtype=torch.long, device=device)
    cluster_id[sort_idx] = torch.arange(n_tiles, device=device) // cluster_size
    cluster_id = cluster_id.clamp(max=n_clusters - 1)
    return cluster_id

def lloyd_max_clustered(tiles, cluster_id, n_bits, n_clusters, device):
    n_levels = 2 ** n_bits
    result = torch.empty_like(tiles)
    for cid in range(n_clusters):
        mask = cluster_id == cid
        if mask.sum() == 0: continue
        ct = tiles[mask]; cf = ct.flatten()
        sigma = cf.std().item()
        if sigma < 1e-12: result[mask] = 0.0; continue
        levels = torch.linspace(-3 * sigma, 3 * sigma, n_levels, device=device)
        for _ in range(12):
            d = (cf.unsqueeze(1) - levels.unsqueeze(0)).abs()
            assign = d.argmin(dim=1)
            new_levels = levels.clone()
            for i in range(n_levels):
                m = assign == i
                if m.sum() > 0: new_levels[i] = cf[m].mean()
            if (new_levels - levels).abs().max() < 1e-10 * sigma: break
            levels = new_levels
        d = (ct.unsqueeze(2) - levels.unsqueeze(0).unsqueeze(0)).abs()
        idx = d.argmin(dim=2)
        result[mask] = levels[idx]
    return result

def run_experiment(data_dir, device, ext, ghd, tcp, tcpi, qtf, cbs_scale):
    results = {}
    n_clusters = 128
    f = data_dir / f"layer10_all_gate_proj.pt"
    if not f.exists(): return results
    all_experts = torch.load(f, map_location="cpu")
    n_experts = min(5, all_experts.shape[0])
    k, n = all_experts.shape[1], all_experts.shape[2]
    print(f"  {n_experts} experts, {k}x{n}", flush=True)

    all_methods = {}

    for ei in range(n_experts):
        print(f"  Expert {ei}...", flush=True)
        w = all_experts[ei].to(device)
        w_reg, _, _ = regularize(w, device, ghd, cbs_scale)
        del w

        qk2 = quantize_trellis_raw(w_reg, 2, device, tcp, tcpi, qtf)
        r2 = w_reg - qk2

        methods = {}

        # Baselines from v37
        methods["K2+K5trsc"] = (rescaled_trellis(qk2, r2, 5, device, tcp, tcpi, qtf, cbs_scale), 7.0)
        methods["K2+6LM"] = (None, 8.0)  # placeholder, compute below

        # LM baseline for 8 bpw
        tiles = get_tiles(r2, k, n); cid = cluster_by_sigma(tiles, n_clusters, device)
        quant6 = lloyd_max_clustered(tiles, cid, 6, n_clusters, device)
        methods["K2+6LM"] = (qk2 + tiles_to_matrix(quant6, k, n), 8.0)
        del tiles, cid, quant6

        # === Part 1: Two-stage rescaled trellis ===
        # Stage 1: K2 + K3trsc (5 bpw)
        recon_s1 = rescaled_trellis(qk2, r2, 3, device, tcp, tcpi, qtf, cbs_scale)
        r_s1 = w_reg - recon_s1  # residual after stage 1
        # Stage 2: K2trsc on residual of stage 1 (total 7 bpw)
        recon_s2 = rescaled_trellis(recon_s1, r_s1, 2, device, tcp, tcpi, qtf, cbs_scale)
        methods["K2+K3trsc+K2trsc (2-stage, 7bpw)"] = (recon_s2, 7.0)

        # Also: K2+K2trsc+K3trsc (2-stage, 7bpw, different order)
        recon_s1b = rescaled_trellis(qk2, r2, 2, device, tcp, tcpi, qtf, cbs_scale)
        r_s1b = w_reg - recon_s1b
        recon_s2b = rescaled_trellis(recon_s1b, r_s1b, 3, device, tcp, tcpi, qtf, cbs_scale)
        methods["K2+K2trsc+K3trsc (2-stage, 7bpw)"] = (recon_s2b, 7.0)

        # K2+K2trsc+K2trsc (6 bpw)
        recon_s2c = rescaled_trellis(recon_s1b, r_s1b, 2, device, tcp, tcpi, qtf, cbs_scale)
        methods["K2+K2trsc+K2trsc (2-stage, 6bpw)"] = (recon_s2c, 6.0)

        # === Part 2: K2+K6trsc (8 bpw) ===
        methods["K2+K6trsc"] = (rescaled_trellis(qk2, r2, 6, device, tcp, tcpi, qtf, cbs_scale), 8.0)

        # K2+K7trsc (9 bpw)
        methods["K2+K7trsc"] = (rescaled_trellis(qk2, r2, 7, device, tcp, tcpi, qtf, cbs_scale), 9.0)

        # === Part 3: Per-tile rescaled trellis ===
        # Only test at 6 bpw (K2+K4trsc) for speed
        recon_pt = per_tile_rescaled_trellis(qk2, r2, 4, k, n, device, tcp, tcpi, qtf, cbs_scale)
        methods["K2+K4trsc_pertile (6bpw)"] = (recon_pt, 6.0)
        del recon_pt

        # Per-tile at 7 bpw (K2+K5trsc)
        recon_pt7 = per_tile_rescaled_trellis(qk2, r2, 5, k, n, device, tcp, tcpi, qtf, cbs_scale)
        methods["K2+K5trsc_pertile (7bpw)"] = (recon_pt7, 7.0)
        del recon_pt7

        # Compute MSEs
        for name, (recon, bpw) in methods.items():
            if recon is None: continue
            mse = (w_reg - recon).pow(2).mean().item()
            if name not in all_methods:
                all_methods[name] = {"mses": [], "bpw": bpw}
            all_methods[name]["mses"].append(mse)
            del recon

        del w_reg, qk2, r2
        torch.cuda.empty_cache()

    # Print sorted by bpw
    print(f"\n  {'Method':<45} {'bpw':>5} {'avg MSE':>12}", flush=True)
    print(f"  {'-'*65}", flush=True)
    for name in sorted(all_methods.keys(), key=lambda x: (all_methods[x]["bpw"], x)):
        r = all_methods[name]
        avg = sum(r["mses"]) / len(r["mses"])
        print(f"  {name:<45} {r['bpw']:>5.0f} {avg:>12.4e}", flush=True)

    # Key comparisons
    print(f"\n  Key comparisons:", flush=True)
    comparisons = [
        (7, "K2+K5trsc", "K2+K3trsc+K2trsc (2-stage, 7bpw)", "K2+K2trsc+K3trsc (2-stage, 7bpw)"),
        (8, "K2+6LM", "K2+K6trsc"),
        (9, "K2+K7trsc"),
    ]
    for comp in comparisons:
        bpw = comp[0]
        print(f"\n  {bpw} bpw:", flush=True)
        for name in comp[1:]:
            if name in all_methods:
                avg = sum(all_methods[name]["mses"]) / len(all_methods[name]["mses"])
                print(f"    {name:<45} MSE={avg:.4e}", flush=True)

    # Per-tile vs global rescaling
    print(f"\n  Per-tile vs global rescaling:", flush=True)
    for bpw, global_name, pt_name in [(6, "K2+K4trsc_pertile (6bpw)", None), (7, "K2+K5trsc_pertile (7bpw)", "K2+K5trsc")]:
        if bpw == 6:
            # Need global K2+K4trsc from v37 — compute it
            pass
        for name in [global_name]:
            if name and name in all_methods:
                avg = sum(all_methods[name]["mses"]) / len(all_methods[name]["mses"])
                print(f"    {name:<45} MSE={avg:.4e}", flush=True)

    results["methods"] = {k: {"bpw": v["bpw"], "avg_mse": sum(v["mses"])/len(v["mses"])}
                          for k, v in all_methods.items()}
    return results

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--data-dir", default="/tmp/poc_residual/glm52_data")
    ap.add_argument("--out", default="/tmp/poc_residual/results_v39.json")
    args = ap.parse_args()
    dev = torch.device(args.device)
    print(f"Device: {dev}  GPU: {torch.cuda.get_device_name(0)}", flush=True)
    ext, ghd, tcp, tcpi, qtf, cbs_scale = _bootstrap()
    results = run_experiment(Path(args.data_dir), dev, ext, ghd, tcp, tcpi, qtf, cbs_scale)
    Path(args.out).write_text(json.dumps(results, indent=2, default=str))
    print(f"\nResults saved to {args.out}", flush=True)

if __name__ == "__main__":
    main()
