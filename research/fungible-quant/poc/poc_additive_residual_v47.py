#!/usr/bin/env python3
"""PoC v47: Dithering + different Hadamard seeds + per-tile K1 rescaling.

Three new ideas to improve MSRT:

1. Subtractive dithering: Add uniform noise before quantization, subtract after.
   For TCQ, dithering can break structured quantization error patterns.
   Theory: subtractive dithering makes quantization error independent of input.

2. Different Hadamard seeds per stage: Currently all MSRT stages use the same
   regularized space (seed 0). Using different random sign vectors per stage
   could decorrelate stage quantization errors, improving multi-stage refinement.

3. Per-tile rescaling for K1 stages: K1 stages handle the largest residual.
   Per-tile rescaling might give better codebook match for K1.
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

def regularize(w, device, ghd, cbs, seed=0):
    k, n = w.shape
    g = torch.Generator(device="cpu").manual_seed(seed)
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

def dithered_rescaled_trellis(base_q, residual, K_res, device, tcp, tcpi, qtf, cbs, seed=42):
    """Subtractive dithering: add uniform noise, quantize, subtract noise."""
    residual_rms = residual.square().mean().sqrt().item()
    if residual_rms < 1e-12: return base_q
    scale = abs(cbs) / residual_rms
    scaled = residual * scale
    # Dither: uniform noise in [-0.5, 0.5] per element
    g = torch.Generator(device=device).manual_seed(seed)
    dither = (torch.rand_like(scaled) - 0.5) * (2 * abs(cbs) / (2**K_res - 1))
    dithered = scaled + dither
    quant = quantize_trellis_raw(dithered, K_res, device, tcp, tcpi, qtf)
    # Subtractive: remove dither from quantized result
    quant_minus_dither = quant - dither
    return base_q + quant_minus_dither / scale

def get_tiles(r, k, n):
    tnk, tnn = k // 16, n // 16
    return r.view(tnk, 16, tnn, 16).permute(0, 2, 1, 3).reshape(tnk * tnn, 256)

def tiles_to_matrix(tiles, k, n):
    tnk, tnn = k // 16, n // 16
    return tiles.reshape(tnk, tnn, 16, 16).permute(0, 2, 1, 3).reshape(k, n)

def run_experiment(data_dir, device, ext, ghd, tcp, tcpi, qtf, cbs_scale):
    results = {}
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

        # Standard regularization (seed 0)
        w_reg, _, _ = regularize(w, device, ghd, cbs_scale, seed=0)
        del w

        qk2 = quantize_trellis_raw(w_reg, 2, device, tcp, tcpi, qtf)
        r2 = w_reg - qk2

        methods = {}

        # === Baseline MSRT 8bpw: K2+K1+K2+K3trsc ===
        s1 = rescaled_trellis(qk2, r2, 1, device, tcp, tcpi, qtf, cbs_scale)
        r_s1 = w_reg - s1
        s2 = rescaled_trellis(s1, r_s1, 2, device, tcp, tcpi, qtf, cbs_scale)
        r_s2 = w_reg - s2
        methods["MSRT_8bpw_baseline"] = (rescaled_trellis(s2, r_s2, 3, device, tcp, tcpi, qtf, cbs_scale), 8.0)

        # === Part 1: Dithered MSRT ===
        # Dither at each stage
        ds1 = dithered_rescaled_trellis(qk2, r2, 1, device, tcp, tcpi, qtf, cbs_scale, seed=42)
        r_ds1 = w_reg - ds1
        ds2 = dithered_rescaled_trellis(ds1, r_ds1, 2, device, tcp, tcpi, qtf, cbs_scale, seed=43)
        r_ds2 = w_reg - ds2
        methods["MSRT_8bpw_dithered"] = (dithered_rescaled_trellis(ds2, r_ds2, 3, device, tcp, tcpi, qtf, cbs_scale, seed=44), 8.0)

        # Dither only at K1 stage
        ds1b = dithered_rescaled_trellis(qk2, r2, 1, device, tcp, tcpi, qtf, cbs_scale, seed=42)
        r_ds1b = w_reg - ds1b
        ds2b = rescaled_trellis(ds1b, r_ds1b, 2, device, tcp, tcpi, qtf, cbs_scale)
        r_ds2b = w_reg - ds2b
        methods["MSRT_8bpw_dither_K1only"] = (rescaled_trellis(ds2b, r_ds2b, 3, device, tcp, tcpi, qtf, cbs_scale), 8.0)

        del s1, r_s1, s2, r_s2, ds1, r_ds1, ds2, r_ds2, ds1b, r_ds1b, ds2b, r_ds2b

        # === Part 2: Different Hadamard seeds per stage ===
        # Re-regularize residual with different seed before each stage
        # Stage 1: re-regularize r2 with seed=1
        # Note: this doesn't make sense for the residual — the residual is already
        # in regularized space. Re-regularizing would apply a different Hadamard
        # to the residual, which changes the space. We'd need to inverse-transform
        # back to original space, re-regularize with new seed, quantize, then
        # inverse back. This is complex and likely doesn't help because the
        # residual is already Gaussian.
        # Instead, test: apply a second Hadamard (random signs) to the residual
        # before rescaled trellis, then inverse after.
        def hadamard_residual(residual, device, ghd, seed):
            """Apply random Hadamard to residual, return transformed + inverse function."""
            k, n = residual.shape
            g = torch.Generator(device="cpu").manual_seed(seed)
            su = (torch.randn(k, generator=g).sign() + 1e-5).sign().float().to(device)
            sv = (torch.randn(n, generator=g).sign() + 1e-5).sign().float().to(device)
            # Apply Hadamard (same structure as regularization but on residual)
            had_n = ghd(HAD_N, device, torch.float, 1.0 / math.sqrt(HAD_N))
            transformed = (residual.view(k, n // HAD_N, HAD_N) @ had_n).view(k, n)
            transformed = transformed * sv.unsqueeze(0)  # apply column signs
            had_k = ghd(HAD_K, device, torch.float, 1.0 / math.sqrt(HAD_K))
            transformed = (had_k @ transformed.view(k // HAD_K, HAD_K, n)).view(k, n)
            transformed = transformed * su.unsqueeze(1)  # apply row signs
            return transformed

        # Apply different Hadamard to r2 before stage 1
        r2_hadamarded = hadamard_residual(r2, device, ghd, seed=100)
        hs1 = rescaled_trellis(qk2, r2_hadamarded, 1, device, tcp, tcpi, qtf, cbs_scale)
        # Note: this changes the space — the reconstruction is qk2 + quant(r2_hadamarded)
        # which is NOT w_reg. This won't work as expected because we need to
        # inverse the Hadamard. Skip this test — it's not well-defined.
        # methods["MSRT_8bpw_hadamard_per_stage"] = (hs1, 8.0)  # Won't be correct
        del r2_hadamarded, hs1

        # === Part 3: MSRT 6bpw with K1 per-tile rescaling ===
        # K2+K1pertile+K3trsc
        # Per-tile rescaling for K1
        tnk, tnn = k // 16, n // 16
        target_rms = abs(cbs_scale)
        r2_tiles = get_tiles(r2, k, n)
        tile_rms = r2_tiles.std(dim=1, keepdim=True).clamp(min=1e-12)
        tile_scale = target_rms / tile_rms
        r2_scaled_tiles = r2_tiles * tile_scale
        r2_scaled = tiles_to_matrix(r2_scaled_tiles, k, n)
        # Quantize scaled residual with K1
        quant_k1 = quantize_trellis_raw(r2_scaled, 1, device, tcp, tcpi, qtf)
        # Scale back per-tile
        quant_k1_tiles = get_tiles(quant_k1, k, n)
        quant_k1_unscaled_tiles = quant_k1_tiles / tile_scale
        recon_k1 = qk2 + tiles_to_matrix(quant_k1_unscaled_tiles, k, n)
        r_after_k1 = w_reg - recon_k1
        # Stage 2: K3trsc (global rescaling)
        methods["MSRT_6bpw_pertile_K1"] = (rescaled_trellis(recon_k1, r_after_k1, 3, device, tcp, tcpi, qtf, cbs_scale), 6.0)

        del r2_tiles, tile_rms, tile_scale, r2_scaled_tiles, r2_scaled
        del quant_k1, quant_k1_tiles, quant_k1_unscaled_tiles, recon_k1, r_after_k1

        # Compute MSEs
        for name, (recon, bpw) in methods.items():
            mse = (w_reg - recon).pow(2).mean().item()
            if name not in all_methods:
                all_methods[name] = {"mses": [], "bpw": bpw}
            all_methods[name]["mses"].append(mse)
            del recon

        del w_reg, qk2, r2
        torch.cuda.empty_cache()

    # Print
    print(f"\n  {'Method':<35} {'bpw':>5} {'avg MSE':>12}", flush=True)
    print(f"  {'-'*55}", flush=True)
    for name in sorted(all_methods.keys(), key=lambda x: (all_methods[x]["bpw"], sum(all_methods[x]["mses"])/len(all_methods[x]["mses"]))):
        r = all_methods[name]
        avg = sum(r["mses"]) / len(r["mses"])
        print(f"  {name:<35} {r['bpw']:>5.0f} {avg:>12.4e}", flush=True)

    # Compare dithered vs baseline
    if "MSRT_8bpw_baseline" in all_methods and "MSRT_8bpw_dithered" in all_methods:
        base_mse = sum(all_methods["MSRT_8bpw_baseline"]["mses"]) / len(all_methods["MSRT_8bpw_baseline"]["mses"])
        dith_mse = sum(all_methods["MSRT_8bpw_dithered"]["mses"]) / len(all_methods["MSRT_8bpw_dithered"]["mses"])
        ratio = dith_mse / base_mse
        print(f"\n  Dithered vs baseline: {ratio:.4f}x ({'better' if ratio < 1 else 'worse'})", flush=True)

    results["methods"] = {k: {"bpw": v["bpw"], "avg_mse": sum(v["mses"])/len(v["mses"])}
                          for k, v in all_methods.items()}
    return results

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--data-dir", default="/tmp/poc_residual/glm52_data")
    ap.add_argument("--out", default="/tmp/poc_residual/results_v47.json")
    args = ap.parse_args()
    dev = torch.device(args.device)
    print(f"Device: {dev}  GPU: {torch.cuda.get_device_name(0)}", flush=True)
    ext, ghd, tcp, tcpi, qtf, cbs_scale = _bootstrap()
    results = run_experiment(Path(args.data_dir), dev, ext, ghd, tcp, tcpi, qtf, cbs_scale)
    Path(args.out).write_text(json.dumps(results, indent=2, default=str))
    print(f"\nResults saved to {args.out}", flush=True)

if __name__ == "__main__":
    main()
