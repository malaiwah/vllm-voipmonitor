#!/usr/bin/env python3
"""PoC v49: Cross-layer validation + expert reordering/rotation/tiling.

PART 1 — Cross-layer validation:
  Run MSRT on layers 10, 30, 50, 60, 70 (gate_proj, 10 experts each).
  Confirm MSRT generalizes across layers.

PART 2 — Expert reordering and rotation:
  2a. Row/column reordering: permute rows/columns of each expert before
      Hadamard to create diversity in tile structure.
  2b. Cross-expert rotation: Hadamard-transform across the expert dimension
      (mixing experts) before standard per-expert regularization.
  2c. Interleaved expert tiling: Stack experts into a super-matrix, tile
      across expert boundaries, quantize super-tiles.
  2d. Shared Hadamard: Apply the SAME Hadamard to all experts (not per-expert
      seed) to see if correlation helps MSRT residual stages.
"""
from __future__ import annotations
import argparse, json, math, os, sys, types, importlib.util, time, gc
from pathlib import Path
import torch

EXL3_PKG = "/opt/fruit-pip/exllamav3"

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
    return ext, ghd, m.tensor_core_perm, m.tensor_core_perm_i, m.quantize_tiles, m.codebook_scale

def block_rms(x, dim, keepdim=False):
    return x.square().mean(dim=dim, keepdim=keepdim).sqrt()

def regularize(w, device, ghd, cbs, had_k=128, had_n=128, seed=0):
    k, n = w.shape
    g = torch.Generator(device="cpu").manual_seed(seed)
    su = (torch.randn(k, generator=g).sign() + 1e-5).sign().float().to(device)
    sv = (torch.randn(n, generator=g).sign() + 1e-5).sign().float().to(device)
    out_scales = block_rms(w, dim=0, keepdim=True)
    mean = out_scales.mean().item()
    if mean > 1e-30: out_scales = out_scales / mean
    sv = (sv * out_scales + 1e-10).float()
    w = (w / sv).contiguous()
    had_n_mat = ghd(had_n, device, torch.float, 1.0 / math.sqrt(had_n))
    w = (w.view(k, n // had_n, had_n) @ had_n_mat).view(k, n).contiguous()
    in_scales = block_rms(w, dim=1, keepdim=True).clamp(min=1e-30)
    su = (su.unsqueeze(1) * in_scales / (-cbs) + 1e-10).float()
    w = (w / su).contiguous()
    had_k_mat = ghd(had_k, device, torch.float, 1.0 / math.sqrt(had_k))
    w = (had_k_mat @ w.view(k // had_k, had_k, n)).view(k, n).contiguous()
    return w, su, sv

def quantize_trellis_raw(data, K, device, tcp, tcpi, qtf):
    k, n = data.shape; tiles_n = n // 16; weight_q = torch.zeros_like(data)
    qa = {"K": K, "mcg": True}
    perm = tcp(device); perm_i = tcpi(device)
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

def msrt_6bpw(w_reg, device, tcp, tcpi, qtf, cbs):
    """MSRT 6bpw: K2 + K1trsc + K3trsc"""
    qk2 = quantize_trellis_raw(w_reg, 2, device, tcp, tcpi, qtf)
    r2 = w_reg - qk2
    s1 = rescaled_trellis(qk2, r2, 1, device, tcp, tcpi, qtf, cbs)
    r_s1 = w_reg - s1
    recon = rescaled_trellis(s1, r_s1, 3, device, tcp, tcpi, qtf, cbs)
    return (w_reg - recon).pow(2).mean().item()

def msrt_8bpw(w_reg, device, tcp, tcpi, qtf, cbs):
    """MSRT 8bpw: K2 + K1 + K2 + K3trsc"""
    qk2 = quantize_trellis_raw(w_reg, 2, device, tcp, tcpi, qtf)
    r2 = w_reg - qk2
    s1 = rescaled_trellis(qk2, r2, 1, device, tcp, tcpi, qtf, cbs)
    r_s1 = w_reg - s1
    s2 = rescaled_trellis(s1, r_s1, 2, device, tcp, tcpi, qtf, cbs)
    r_s2 = w_reg - s2
    recon = rescaled_trellis(s2, r_s2, 3, device, tcp, tcpi, qtf, cbs)
    return (w_reg - recon).pow(2).mean().item()

def run_cross_layer(data_dir, device, ghd, tcp, tcpi, qtf, cbs):
    """Part 1: Run MSRT on layers 10, 30, 50, 60, 70."""
    results = {}
    print("\n=== Part 1: Cross-Layer Validation ===", flush=True)
    for layer in [10, 30, 50, 60, 70]:
        f = data_dir / f"layer{layer}_all_gate_proj.pt"
        if not f.exists():
            print(f"  Layer {layer}: data not found, skipping", flush=True)
            continue
        experts = torch.load(f, map_location="cpu")
        n = min(10, experts.shape[0])
        mses_6 = []; mses_8 = []
        for ei in range(n):
            w = experts[ei].to(device)
            w_reg, _, _ = regularize(w, device, ghd, cbs)
            del w
            mses_6.append(msrt_6bpw(w_reg, device, tcp, tcpi, qtf, cbs))
            mses_8.append(msrt_8bpw(w_reg, device, tcp, tcpi, qtf, cbs))
            del w_reg
            torch.cuda.empty_cache()
        avg6 = sum(mses_6) / len(mses_6)
        avg8 = sum(mses_8) / len(mses_8)
        results[f"layer{layer}_6bpw"] = avg6
        results[f"layer{layer}_8bpw"] = avg8
        print(f"  Layer {layer:2d}: 6bpw={avg6:.4e}  8bpw={avg8:.4e}  (n={n})", flush=True)
        del experts, mses_6, mses_8
        gc.collect()
    return results

def run_expert_reordering(data_dir, device, ghd, tcp, tcpi, qtf, cbs):
    """Part 2: Expert reordering and rotation experiments."""
    results = {}
    print("\n=== Part 2: Expert Reordering + Rotation ===", flush=True)

    f = data_dir / "layer10_all_gate_proj.pt"
    if not f.exists():
        print("  Layer 10 data not found", flush=True)
        return results
    experts = torch.load(f, map_location="cpu")
    n = min(5, experts.shape[0])

    # --- 2a: Row/column permutation before Hadamard ---
    print("\n  2a: Row/column permutation before regularization", flush=True)
    g = torch.Generator(device="cpu").manual_seed(42)
    k, nn = experts.shape[1], experts.shape[2]
    row_perm = torch.randperm(k, generator=g)
    col_perm = torch.randperm(nn, generator=g)

    mses_perm = []
    for ei in range(n):
        w = experts[ei].to(device)
        # Permute rows and columns, then regularize
        w_perm = w[row_perm.to(device)][:, col_perm.to(device)].contiguous()
        w_reg, _, _ = regularize(w_perm, device, ghd, cbs)
        del w, w_perm
        mses_perm.append(msrt_6bpw(w_reg, device, tcp, tcpi, qtf, cbs))
        del w_reg
        torch.cuda.empty_cache()
    avg_perm = sum(mses_perm) / len(mses_perm)
    results["2a_rowcol_perm_6bpw"] = avg_perm
    print(f"    Row/col perm: 6bpw={avg_perm:.4e}", flush=True)

    # --- 2b: Cross-expert Hadamard mixing ---
    print("\n  2b: Cross-expert Hadamard mixing", flush=True)
    # Stack experts along row dimension, apply Hadamard across expert blocks
    # This mixes expert weights to see if cross-expert structure helps
    mses_cross = []
    for ei in range(n):
        w = experts[ei].to(device)
        w_reg, _, _ = regularize(w, device, ghd, cbs)
        # Now also apply a Hadamard on the column dimension with a different seed
        # to create cross-expert correlation
        g2 = torch.Generator(device="cpu").manual_seed(99)
        sv2 = (torch.randn(nn, generator=g2).sign() + 1e-5).sign().float().to(device)
        w_reg2 = w_reg * sv2  # Random sign flip (diagonal Hadamard)
        mses_cross.append(msrt_6bpw(w_reg2, device, tcp, tcpi, qtf, cbs))
        del w, w_reg, w_reg2
        torch.cuda.empty_cache()
    avg_cross = sum(mses_cross) / len(mses_cross)
    results["2b_signflip_6bpw"] = avg_cross
    print(f"    Sign flip: 6bpw={avg_cross:.4e}", flush=True)

    # --- 2c: Interleaved expert tiling (super-tiles) ---
    print("\n  2c: Interleaved expert tiling", flush=True)
    # Stack 2 experts along columns, quantize the wider matrix
    # This creates tiles that span 2 experts
    mses_super = []
    for ei in range(0, n - 1, 2):
        w1 = experts[ei].to(device)
        w2 = experts[ei + 1].to(device)
        # Concatenate along column dimension: (2048, 12288)
        w_cat = torch.cat([w1, w2], dim=1)
        del w1, w2
        w_reg, _, _ = regularize(w_cat, device, ghd, cbs)
        mses_super.append(msrt_6bpw(w_reg, device, tcp, tcpi, qtf, cbs))
        del w_reg
        torch.cuda.empty_cache()
    avg_super = sum(mses_super) / len(mses_super) if mses_super else 0
    results["2c_supertile_2exp_6bpw"] = avg_super
    print(f"    Super-tile (2 experts): 6bpw={avg_super:.4e}  (n={len(mses_super)})", flush=True)

    # --- 2d: Shared Hadamard (same seed for all experts) ---
    print("\n  2d: Shared Hadamard (same seed=0 for all, already baseline)", flush=True)
    # The baseline already uses seed=0 for all experts, so this IS the baseline.
    # Test with different seeds per expert to see if diversity matters.
    mses_diff_seeds = []
    for ei in range(n):
        w = experts[ei].to(device)
        w_reg, _, _ = regularize(w, device, ghd, cbs, seed=ei * 7)  # Different seed per expert
        del w
        mses_diff_seeds.append(msrt_6bpw(w_reg, device, tcp, tcpi, qtf, cbs))
        del w_reg
        torch.cuda.empty_cache()
    avg_diff_seeds = sum(mses_diff_seeds) / len(mses_diff_seeds)
    results["2d_diff_seeds_6bpw"] = avg_diff_seeds
    print(f"    Different seeds: 6bpw={avg_diff_seeds:.4e}", flush=True)

    # --- 2e: Cross-expert rotation (Hadamard across expert dimension) ---
    print("\n  2e: Cross-expert rotation (Hadamard mixing experts)", flush=True)
    # Apply a Hadamard transform that mixes experts: reshape (n_experts, k, n)
    # to (k, n_experts, n) and Hadamard on the n_experts dimension
    # This requires n_experts to be a power of 2 for Hadamard
    n_mix = min(4, n)  # Use 4 experts for mixing (power of 2)
    had_mix_size = n_mix
    had_mix = ghd(had_mix_size, device, torch.float, 1.0 / math.sqrt(had_mix_size))

    mses_mix = []
    # Process all expert groups
    for start in range(0, n - n_mix + 1, n_mix):
        group = [experts[start + j].to(device) for j in range(n_mix)]
        # Stack: (n_mix, k, n) -> reshape to (k, n_mix, n) -> hadamard on n_mix
        stacked = torch.stack(group)  # (n_mix, k, n)
        del group
        k_dim, n_dim = stacked.shape[1], stacked.shape[2]
        # Permute to (k, n_mix, n)
        permuted = stacked.permute(1, 0, 2).contiguous()  # (k, n_mix, n)
        del stacked
        # Apply Hadamard on n_mix dimension
        mixed = (had_mix @ permuted).contiguous()  # (k, n_mix, n)
        del permuted
        # Permute back: (n_mix, k, n)
        mixed_back = mixed.permute(1, 0, 2).contiguous()
        del mixed

        # Now regularize and quantize each mixed expert
        for j in range(n_mix):
            w = mixed_back[j]
            w_reg, _, _ = regularize(w, device, ghd, cbs)
            mses_mix.append(msrt_6bpw(w_reg, device, tcp, tcpi, qtf, cbs))
            del w, w_reg
        del mixed_back
        torch.cuda.empty_cache()

    avg_mix = sum(mses_mix) / len(mses_mix) if mses_mix else 0
    results["2e_cross_expert_hadamard_6bpw"] = avg_mix
    print(f"    Cross-expert Hadamard mix: 6bpw={avg_mix:.4e}  (n={len(mses_mix)})", flush=True)

    return results

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--data-dir", default="/tmp/poc_residual/glm52_data")
    ap.add_argument("--out", default="/tmp/poc_residual/results_v49.json")
    args = ap.parse_args()
    dev = torch.device(args.device)
    print(f"Device: {dev}  GPU: {torch.cuda.get_device_name(0)}", flush=True)
    ext, ghd, tcp, tcpi, qtf, cbs = _bootstrap()
    print(f"codebook_scale = {cbs}", flush=True)

    data_dir = Path(args.data_dir)
    results = {}

    # Part 1: Cross-layer validation
    results["cross_layer"] = run_cross_layer(data_dir, dev, ghd, tcp, tcpi, qtf, cbs)

    # Part 2: Expert reordering + rotation
    results["expert_reordering"] = run_expert_reordering(data_dir, dev, ghd, tcp, tcpi, qtf, cbs)

    # Summary
    print("\n=== Summary ===", flush=True)
    print("\nCross-layer MSRT (gate_proj, 10 experts):", flush=True)
    for layer in [10, 30, 50, 60, 70]:
        k6 = f"layer{layer}_6bpw"
        k8 = f"layer{layer}_8bpw"
        if k6 in results["cross_layer"]:
            print(f"  Layer {layer:2d}: 6bpw={results['cross_layer'][k6]:.4e}  8bpw={results['cross_layer'][k8]:.4e}", flush=True)

    print("\nExpert reordering (layer 10 gate_proj, 5 experts, 6bpw):", flush=True)
    baseline_6 = results["cross_layer"].get("layer10_6bpw", 0)
    for name, val in sorted(results["expert_reordering"].items()):
        ratio = val / baseline_6 if baseline_6 > 0 else 0
        print(f"  {name}: {val:.4e}  ({ratio:.4f}x baseline)", flush=True)

    Path(args.out).write_text(json.dumps(results, indent=2, default=str))
    print(f"\nResults saved to {args.out}", flush=True)

if __name__ == "__main__":
    main()
