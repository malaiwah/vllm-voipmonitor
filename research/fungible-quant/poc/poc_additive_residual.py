#!/usr/bin/env python3
"""Proof-of-Concept: Additive Residual Encoding for EXL3 Trellis K2→K3→K4.

Question: Can a 1-bit scalar residual plane, added on top of a K-bit trellis
base encode, capture enough of the K→K+1 improvement to be worth pursuing?

This PoC simulates the EXL3 quantization pipeline (Hadamard incoherence
processing + K-bit quantization) using uniform round-to-nearest as a
conservative proxy for the Viterbi trellis search. Uniform quantization is
strictly worse than trellis at the same K (trellis exploits inter-weight
correlations), so the residuals here are LARGER than real EXL3 residuals —
making the 1-bit residual quality estimates CONSERVATIVE (if it works with
uniform, it will work better with trellis).

Pipeline (faithful to EXL3):
  1. Random sign flips (su, sv) — incoherence processing
  2. Blockwise 128×128 Hadamard transforms — spreads outliers
  3. K-bit uniform quantization (proxy for Viterbi trellis)
  4. Undo Hadamard and sign flips to get reconstructed weights
  5. Compute residuals and 1-bit scalar quantize them
  6. Measure: MSE, cosine, relative Frobenius, % of gap closed

Runs on CPU or MPS. No CUDA, no exllamav3 dependency.

Usage:
  python poc_additive_residual.py [--device mps] [--data-dir /tmp/glm52_expert]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Hadamard transform (pure PyTorch, same as EXL3's 128×128 block Hadamard)
# ---------------------------------------------------------------------------

_had_cache: dict[int, torch.Tensor] = {}


def hadamard_matrix(n: int) -> torch.Tensor:
    """Return an n×n normalized Hadamard matrix (Sylvester construction)."""
    if n in _had_cache:
        return _had_cache[n]
    assert (n & (n - 1)) == 0, f"n must be power of 2, got {n}"
    H = torch.tensor([[1.0]])
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], dim=1),
                        torch.cat([H, -H], dim=1)], dim=0)
    H = H / math.sqrt(n)
    _had_cache[n] = H
    return H


def apply_hadamard_left(w: torch.Tensor, block: int = 128) -> torch.Tensor:
    """Apply block Hadamard on the row dimension (in_features)."""
    H = hadamard_matrix(block).to(w.dtype).to(w.device)
    rows, cols = w.shape
    assert rows % block == 0
    w = w.view(rows // block, block, cols)
    # H is (block, block); w is (n_blocks, block, cols)
    # Need (n_blocks, block, block) @ (n_blocks, block, cols)
    w = H @ w  # broadcast: (block, block) @ (n_blocks, block, cols) → (n_blocks, block, cols)
    return w.view(rows, cols)

def apply_hadamard_right(w: torch.Tensor, block: int = 128) -> torch.Tensor:
    """Apply block Hadamard on the column dimension (out_features)."""
    H = hadamard_matrix(block).to(w.dtype).to(w.device)
    rows, cols = w.shape
    assert cols % block == 0
    w = w.view(rows, cols // block, block)
    # w is (rows, n_blocks, block); H is (block, block)
    # Need (rows, n_blocks, block) @ (block, block) → (rows, n_blocks, block)
    w = w @ H  # broadcast: (rows, n_blocks, block) @ (block, block) → (rows, n_blocks, block)
    return w.view(rows, cols)


# ---------------------------------------------------------------------------
# Uniform K-bit quantization (proxy for EXL3 trellis)
# ---------------------------------------------------------------------------

def quantize_uniform(w: torch.Tensor, K: int) -> torch.Tensor:
    """Uniform round-to-nearest K-bit quantization.

    Quantizes to 2^K levels symmetrically around zero.
    This is a conservative proxy for EXL3's Viterbi trellis search:
    - Same K bits/weight
    - Same number of reconstruction levels (2^K)
    - Worse quality (no inter-weight correlation exploitation)
    - Residuals are LARGER → 1-bit estimates are CONSERVATIVE
    """
    if K <= 0:
        return torch.zeros_like(w)
    levels = 2 ** K
    # Symmetric quantization: scale to [-1, 1) then round
    max_abs = w.abs().max()
    if max_abs < 1e-12:
        return torch.zeros_like(w)
    scale = max_abs / (levels / 2 - 1) if levels > 2 else max_abs
    q = torch.round(w / scale) * scale
    return q


# ---------------------------------------------------------------------------
# 1-bit residual quantization (the approach under test)
# ---------------------------------------------------------------------------

def quantize_1bit_global(r: torch.Tensor) -> tuple[torch.Tensor, float]:
    """1-bit scalar quantization with a single global scale.

    r_hat = sign(r) * mean(|r|)

    Returns (reconstructed residual, scale).
    """
    s = r.abs().mean().item()
    if s < 1e-12:
        return torch.zeros_like(r), 0.0
    r_hat = torch.sign(r) * s
    return r_hat, s


def quantize_1bit_grouped(r: torch.Tensor, group_size: int) -> torch.Tensor:
    """1-bit scalar quantization with per-group scales.

    Groups along the row dimension (in_features).
    """
    if group_size <= 0 or group_size >= r.shape[0]:
        r_hat, _ = quantize_1bit_global(r)
        return r_hat
    rows = r.shape[0]
    pad = (group_size - rows % group_size) % group_size
    if pad > 0:
        r = F.pad(r, (0, 0, 0, pad))
    rows_padded = r.shape[0]
    r = r.view(rows_padded // group_size, group_size, -1)
    s = r.abs().mean(dim=1, keepdim=True)
    s = torch.clamp(s, min=1e-12)
    r_hat = torch.sign(r) * s
    r_hat = r_hat.view(rows_padded, -1)
    if pad > 0:
        r_hat = r_hat[:rows, :]
    return r_hat


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def metrics(w_ref: torch.Tensor, w_approx: torch.Tensor) -> dict:
    err = w_ref - w_approx
    mse = err.pow(2).mean().item()
    cos = F.cosine_similarity(w_ref.flatten().unsqueeze(0),
                               w_approx.flatten().unsqueeze(0), dim=1).item()
    rel_frob = (err.norm() / w_ref.norm()).item()
    max_abs = err.abs().max().item()
    return {"mse": mse, "cos": cos, "rel_frob": rel_frob, "max_abs": max_abs}


# ---------------------------------------------------------------------------
# Main PoC
# ---------------------------------------------------------------------------

def run_poc(w: torch.Tensor, name: str, device: torch.device) -> dict:
    """Run the full PoC on one tensor.

    w: (in_features, out_features) float32 tensor (row-major, as EXL3 expects).
    """
    w = w.to(device)
    HAD_BLOCK = 128

    # --- Step 1: EXL3-like preprocessing ---
    # Random sign flips (reproducible)
    g = torch.Generator(device="cpu").manual_seed(42)
    su = (torch.randn(w.shape[0], generator=g).sign() + 1e-5).sign().float().to(device)
    sv = (torch.randn(w.shape[1], generator=g).sign() + 1e-5).sign().float().to(device)

    # Apply sign flips
    w_signed = w * su.unsqueeze(1) * sv.unsqueeze(0)

    # Apply block Hadamard on both dimensions
    w_reg = apply_hadamard_left(w_signed, HAD_BLOCK)
    w_reg = apply_hadamard_right(w_reg, HAD_BLOCK)

    # --- Step 2: Quantize at K=2,3,4 (proxy for trellis) ---
    qk = {}
    for K in [2, 3, 4]:
        wq_reg = quantize_uniform(w_reg, K)
        # Undo Hadamard
        wq = apply_hadamard_right(wq_reg, HAD_BLOCK)
        wq = apply_hadamard_left(wq, HAD_BLOCK)
        # Undo sign flips
        wq = wq * su.unsqueeze(1) * sv.unsqueeze(0)
        qk[K] = wq

    # --- Step 3: Compute residuals ---
    # Direct residuals (from BF16 reference)
    r_23 = w - qk[2]  # K2→K3 residual (what 1-bit needs to capture)
    r_34 = w - qk[3]  # K3→K4 residual

    # --- Step 4: 1-bit quantize residuals ---
    # Global scale
    r_23_hat_g, s_23 = quantize_1bit_global(r_23)
    r_34_hat_g, s_34 = quantize_1bit_global(r_34)

    # Per-group scales (group=128, matching Hadamard block)
    r_23_hat_128 = quantize_1bit_grouped(r_23, 128)
    r_34_hat_128 = quantize_1bit_grouped(r_34, 128)

    # Per-group scales (group=1024)
    r_23_hat_1024 = quantize_1bit_grouped(r_23, 1024)
    r_34_hat_1024 = quantize_1bit_grouped(r_34, 1024)

    # --- Step 5: Reconstruct ---
    w_k3_21_global = qk[2] + r_23_hat_g       # 2+1 (global)
    w_k3_21_g128   = qk[2] + r_23_hat_128      # 2+1 (group=128)
    w_k3_21_g1024  = qk[2] + r_23_hat_1024     # 2+1 (group=1024)

    w_k4_31_global = qk[3] + r_34_hat_g        # 3+1 (global)
    w_k4_31_g128   = qk[3] + r_34_hat_128       # 3+1 (group=128)
    w_k4_31_g1024  = qk[3] + r_34_hat_1024      # 3+1 (group=1024)

    # Chained: 2+1+1 (approximate K3 → approximate K4)
    r_chained = w - w_k3_21_g128  # residual from APPROXIMATE K3
    r_chained_hat, s_chain = quantize_1bit_global(r_chained)
    r_chained_hat_128 = quantize_1bit_grouped(r_chained, 128)
    w_k4_211_global = w_k3_21_g128 + r_chained_hat
    w_k4_211_g128   = w_k3_21_g128 + r_chained_hat_128

    # --- Step 6: Measure ---
    results = {}
    ref_metrics = metrics(w, w)  # sanity (should be perfect)

    # Standalone quantization (references)
    for K in [2, 3, 4]:
        results[f"K{K}_standalone"] = metrics(w, qk[K])

    # Residual approaches
    results["K3_2+1_global"]  = metrics(w, w_k3_21_global)
    results["K3_2+1_g128"]    = metrics(w, w_k3_21_g128)
    results["K3_2+1_g1024"]   = metrics(w, w_k3_21_g1024)

    results["K4_3+1_global"]  = metrics(w, w_k4_31_global)
    results["K4_3+1_g128"]    = metrics(w, w_k4_31_g128)
    results["K4_3+1_g1024"]   = metrics(w, w_k4_31_g1024)

    results["K4_2+1+1_global"] = metrics(w, w_k4_211_global)
    results["K4_2+1+1_g128"]   = metrics(w, w_k4_211_g128)

    # --- Gap closed analysis ---
    eps_k2 = results["K2_standalone"]["mse"]
    eps_k3 = results["K3_standalone"]["mse"]
    eps_k4 = results["K4_standalone"]["mse"]

    gap_23 = eps_k2 - eps_k3  # K2→K3 improvement
    gap_34 = eps_k3 - eps_k4  # K3→K4 improvement
    gap_24 = eps_k2 - eps_k4  # K2→K4 total

    gap_closed = {}
    for label, r in results.items():
        if label.startswith("K3_2+1"):
            gc = (eps_k2 - r["mse"]) / gap_23 if gap_23 > 0 else 0
            gap_closed[label] = gc
        elif label.startswith("K4_3+1"):
            gc = (eps_k3 - r["mse"]) / gap_34 if gap_34 > 0 else 0
            gap_closed[label] = gc
        elif label.startswith("K4_2+1+1"):
            gc = (eps_k2 - r["mse"]) / gap_24 if gap_24 > 0 else 0
            gap_closed[label] = gc

    r_23_mean = r_23.mean().item()
    r_23_std = r_23.std().item()
    res_stats = {
        "r_23": {
            "mean": r_23.mean().item(),
            "std": r_23.std().item(),
            "abs_mean": r_23.abs().mean().item(),
            "kurtosis": ((r_23 - r_23.mean()) ** 4).mean().item() /
                        (r_23.std().item() ** 4 + 1e-12) - 3.0,
            "sign_entropy": -(0.5 * (1 + r_23_mean/(r_23_std+1e-12)) *
                             math.log(0.5 * (1 + r_23_mean/(r_23_std+1e-12)) + 1e-12) +
                             0.5 * (1 - r_23_mean/(r_23_std+1e-12)) *
                             math.log(0.5 * (1 - r_23_mean/(r_23_std+1e-12)) + 1e-12))
                             if r_23_std > 1e-12 else 0.0,
        },
        "r_34": {
            "mean": r_34.mean().item(),
            "std": r_34.std().item(),
            "abs_mean": r_34.abs().mean().item(),
            "kurtosis": ((r_34 - r_34.mean()) ** 4).mean().item() /
                        (r_34.std().item() ** 4 + 1e-12) - 3.0,
        },
    }

    # --- Memory overhead analysis ---
    numel = w.numel()
    overhead = {
        "global_scale": 2 / numel * 8,  # 1 fp16 = 2 bytes, in bits/weight
        "group_128":   128 * 2 / numel * 8 / (numel / 128) * 0 + 2 * (numel // 128) / numel * 8,
        "group_1024":  2 * (numel // 1024) / numel * 8,
    }
    # Simpler: overhead = num_scales * 2 bytes * 8 bits / numel
    overhead = {
        "global":  2 * 8 / numel,                    # 1 scale
        "g128":    (numel // 128) * 2 * 8 / numel,   # 1 scale per 128 weights
        "g1024":   (numel // 1024) * 2 * 8 / numel,  # 1 scale per 1024 weights
    }

    return {
        "tensor": name,
        "shape": list(w.shape),
        "numel": numel,
        "metrics": results,
        "gap_closed": gap_closed,
        "residual_stats": res_stats,
        "eps": {"K2": eps_k2, "K3": eps_k3, "K4": eps_k4},
        "gap": {"K2_K3": gap_23, "K3_K4": gap_34, "K2_K4": gap_24},
        "memory_overhead_bits_per_weight": overhead,
        "improvement_ratios": {
            "K2_to_K3": eps_k2 / eps_k3 if eps_k3 > 0 else float("inf"),
            "K3_to_K4": eps_k3 / eps_k4 if eps_k4 > 0 else float("inf"),
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Additive residual encoding PoC")
    parser.add_argument("--device", default="cpu",
                        help="torch device (cpu, mps, cuda:0)")
    parser.add_argument("--data-dir", default="/tmp/glm52_expert",
                        help="Directory with gate_proj.pt, up_proj.pt, down_proj.pt")
    parser.add_argument("--out", default=None,
                        help="Output JSON path (default: stdout)")
    args = parser.parse_args()

    device = torch.device(args.device)
    data_dir = Path(args.data_dir)

    print(f"Device: {device}", flush=True)
    print(f"Data dir: {data_dir}", flush=True)

    all_results = {}
    for proj in ["gate_proj", "up_proj", "down_proj"]:
        path = data_dir / f"{proj}.pt"
        if not path.exists():
            print(f"SKIP {proj}: {path} not found", flush=True)
            continue
        w = torch.load(path, map_location="cpu")
        print(f"\n{'='*60}", flush=True)
        print(f"Processing {proj}: shape={tuple(w.shape)}, "
              f"mean={w.mean():.6f}, std={w.std():.6f}", flush=True)
        print(f"{'='*60}", flush=True)

        res = run_poc(w, proj, device)
        all_results[proj] = res

        # Print summary
        eps = res["eps"]
        gc = res["gap_closed"]
        ratios = res["improvement_ratios"]
        print(f"\n  Standalone (MSE):", flush=True)
        print(f"    K2: {eps['K2']:.6e}", flush=True)
        print(f"    K3: {eps['K3']:.6e}  ({ratios['K2_to_K3']:.2f}x better than K2)", flush=True)
        print(f"    K4: {eps['K4']:.6e}  ({ratios['K3_to_K4']:.2f}x better than K3)", flush=True)

        print(f"\n  Residual approach (MSE):", flush=True)
        for label in ["K3_2+1_global", "K3_2+1_g128", "K3_2+1_g1024",
                       "K4_3+1_global", "K4_3+1_g128", "K4_3+1_g1024",
                       "K4_2+1+1_global", "K4_2+1+1_g128"]:
            m = res["metrics"][label]
            g = gc.get(label, 0)
            print(f"    {label:20s}: MSE={m['mse']:.6e}  cos={m['cos']:.6f}  "
                  f"relFrob={m['rel_frob']:.6f}  gap_closed={g:.1%}", flush=True)

        print(f"\n  Residual statistics:", flush=True)
        for rn, rs in res["residual_stats"].items():
            print(f"    {rn}: mean={rs['mean']:.2e}  std={rs['std']:.2e}  "
                  f"abs_mean={rs['abs_mean']:.2e}  kurtosis={rs['kurtosis']:.2f}", flush=True)

        print(f"\n  Memory overhead (bits/weight):", flush=True)
        for on, ov in res["memory_overhead_bits_per_weight"].items():
            print(f"    {on}: {ov:.6f}", flush=True)

    # Aggregate across all projections
    print(f"\n{'='*60}", flush=True)
    print("AGGREGATE (mean across all projections)", flush=True)
    print(f"{'='*60}", flush=True)

    if all_results:
        agg = {}
        for proj, res in all_results.items():
            for label, m in res["metrics"].items():
                if label not in agg:
                    agg[label] = {"mse": [], "cos": [], "rel_frob": []}
                agg[label]["mse"].append(m["mse"])
                agg[label]["cos"].append(m["cos"])
                agg[label]["rel_frob"].append(m["rel_frob"])

        agg_gc = {}
        for proj, res in all_results.items():
            for label, gc in res["gap_closed"].items():
                if label not in agg_gc:
                    agg_gc[label] = []
                agg_gc[label].append(gc)

        eps_agg = {}
        for proj, res in all_results.items():
            for K, e in res["eps"].items():
                if K not in eps_agg:
                    eps_agg[K] = []
                eps_agg[K].append(e)

        print(f"\n  Standalone (mean MSE):", flush=True)
        for K in ["K2", "K3", "K4"]:
            vals = eps_agg.get(K, [0])
            print(f"    {K}: {sum(vals)/len(vals):.6e}", flush=True)

        print(f"\n  Residual approach (mean MSE, mean gap_closed):", flush=True)
        for label in ["K3_2+1_global", "K3_2+1_g128", "K3_2+1_g1024",
                       "K4_3+1_global", "K4_3+1_g128", "K4_3+1_g1024",
                       "K4_2+1+1_global", "K4_2+1+1_g128"]:
            if label in agg:
                mse = sum(agg[label]["mse"]) / len(agg[label]["mse"])
                cos = sum(agg[label]["cos"]) / len(agg[label]["cos"])
                rf = sum(agg[label]["rel_frob"]) / len(agg[label]["rel_frob"])
                gc_vals = agg_gc.get(label, [0])
                gc = sum(gc_vals) / len(gc_vals)
                print(f"    {label:20s}: MSE={mse:.6e}  cos={cos:.6f}  "
                      f"relFrob={rf:.6f}  gap_closed={gc:.1%}", flush=True)

        # Improvement ratio
        eps_k2 = sum(eps_agg["K2"]) / len(eps_agg["K2"])
        eps_k3 = sum(eps_agg["K3"]) / len(eps_agg["K3"])
        eps_k4 = sum(eps_agg["K4"]) / len(eps_agg["K4"])
        print(f"\n  Improvement ratios:", flush=True)
        print(f"    K2→K3: {eps_k2/eps_k3:.2f}x", flush=True)
        print(f"    K3→K4: {eps_k3/eps_k4:.2f}x", flush=True)

    # Save results
    output = {
        "poc": "additive_residual_encoding",
        "description": "1-bit scalar residual on top of K-bit uniform quantization (proxy for EXL3 trellis)",
        "note": "Uniform quantization is a conservative proxy. Real EXL3 trellis is better at each K, "
                "so residuals are smaller and 1-bit quality would be higher.",
        "device": str(device),
        "model": "zai-org/GLM-5.2 layer 30 expert 137",
        "per_tensor": all_results,
    }

    out_path = args.out or "poc_additive_residual_results.json"
    Path(out_path).write_text(json.dumps(output, indent=2, default=str))
    print(f"\nResults saved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
