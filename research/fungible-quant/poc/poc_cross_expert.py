#!/usr/bin/env python3
"""Cross-expert residual analysis: run on local Mac with 64 experts.

Loads 64 expert gate_proj weights, quantizes each with uniform K2 proxy
(same as the CPU PoC), computes residuals, and analyzes cross-expert
structure via SVD.

Uses uniform quantization (proxy for trellis) — the cross-expert structure
should be the same regardless of quantizer, since the regularization
(Hadamard + sign flips) is what creates any shared structure.
"""

from __future__ import annotations
import json, math, sys
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np

HAD_BLOCK = 128

def hadamard_matrix(n):
    H = torch.tensor([[1.0]])
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return H / math.sqrt(n)

def apply_had_left(w, block=128):
    H = hadamard_matrix(block).to(w.dtype)
    rows, cols = w.shape
    w = w.view(rows // block, block, cols)
    return (H @ w).view(rows, cols)

def apply_had_right(w, block=128):
    H = hadamard_matrix(block).to(w.dtype)
    rows, cols = w.shape
    w = w.view(rows, cols // block, block)
    return (w @ H).view(rows, cols)

def quantize_uniform(w, K):
    if K <= 0: return torch.zeros_like(w)
    levels = 2 ** K
    max_abs = w.abs().max()
    if max_abs < 1e-12: return torch.zeros_like(w)
    scale = max_abs / (levels / 2 - 1) if levels > 2 else max_abs
    return torch.round(w / scale) * scale

def regularize(w, device="cpu"):
    k, n = w.shape
    g = torch.Generator(device="cpu").manual_seed(0)
    su = (torch.randn(k, generator=g).sign() + 1e-5).sign().float()
    sv = (torch.randn(n, generator=g).sign() + 1e-5).sign().float()  # (n,) not (1,n)
    w = (w * su.unsqueeze(1) * sv.unsqueeze(0)).contiguous()
    w = apply_had_left(w, HAD_BLOCK).contiguous()
    w = apply_had_right(w, HAD_BLOCK).contiguous()
    return w
def q1b_scalar(r):
    s = r.abs().mean().item()
    return torch.zeros_like(r) if s < 1e-12 else torch.sign(r) * s

def main():
    data_dir = Path("/tmp/glm52_experts64")
    experts = sorted(data_dir.glob("expert_*_gate.pt"))
    print(f"Found {len(experts)} experts", flush=True)

    # Load, regularize, quantize at K2, compute residuals
    residuals = []
    for i, p in enumerate(experts):
        w = torch.load(p, map_location="cpu").float()
        w_reg = regularize(w)
        qk2 = quantize_uniform(w_reg, 2)
        r = (w_reg - qk2).flatten()
        residuals.append(r.numpy().astype(np.float32))
        if i < 3 or i >= len(experts) - 1:
            print(f"  expert {i}: shape={w.shape} residual std={r.std():.4f}", flush=True)

    R = np.stack(residuals)  # (n_experts, numel)
    n_experts, d = R.shape
    print(f"\nResidual matrix: {R.shape}", flush=True)
    print(f"Per-expert std: mean={R.std(axis=1).mean():.4f} min={R.std(axis=1).min():.4f} max={R.std(axis=1).max():.4f}", flush=True)

    # SVD (on CPU, numpy handles large matrices)
    print("\nComputing SVD...", flush=True)
    # R is (64, 12.6M) — too wide for full SVD. Use truncated SVD via randomized projection.
    # Actually, for a (64, d) matrix, SVD only needs the (64, 64) Gram matrix R @ R.T
    # Eigenvalues of R @ R.T = singular values squared
    G = R @ R.T  # (64, 64) Gram matrix
    eigvals, eigvecs = np.linalg.eigh(G)
    # Sort descending
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    singular_values = np.sqrt(np.maximum(eigvals, 0))

    total_var = singular_values.sum()
    total_energy = (singular_values ** 2).sum()
    print(f"\nSingular values (top 10): {['%.4f' % s for s in singular_values[:10]]}", flush=True)
    print(f"Total variance (sum of σ): {total_var:.4f}", flush=True)
    print(f"Total energy (sum of σ²): {total_energy:.4f}", flush=True)

    for r in [1, 2, 4, 8, 16, 32, 64]:
        var_captured = (singular_values[:r] ** 2).sum() / total_energy
        print(f"  Rank-{r:2d}: captures {var_captured:.1%} of residual energy", flush=True)

    # Common-mode (mean residual across experts)
    r_mean = R.mean(axis=0)
    r_mean_energy = (r_mean ** 2).sum()
    print(f"\nCommon-mode (mean) energy: {r_mean_energy:.4f} ({r_mean_energy/total_energy:.1%} of total)", flush=True)

    # Low-rank reconstruction quality
    # For rank-r: R_r = U_r @ S_r @ V_r^T
    # But we don't have V_r (would need R^T @ eigvecs). Instead, measure
    # how well rank-r captures each expert's residual.
    print("\nLow-rank residual reconstruction:", flush=True)
    for r in [1, 2, 4, 8, 16]:
        # Project each expert onto top-r principal components
        # R_proj = eigvecs[:, :r] @ eigvecs[:, :r].T @ R
        # But that's expensive. Use: for each expert i, the rank-r approx is
        # sum_j (eigvecs[i,j] * singular_values[j]) * V_j
        # The reconstruction error is ||R[i] - R_proj[i]||^2
        # = ||R[i]||^2 - sum_j (eigvecs[i,j] * singular_values[j])^2  (Pythagorean)
        per_expert_energy = (R ** 2).sum(axis=1)  # (n_experts,)
        captured = (eigvecs[:, :r] * singular_values[:r]) ** 2  # (n_experts, r)
        captured_per_expert = captured.sum(axis=1)  # (n_experts,)
        frac_captured = captured_per_expert / np.maximum(per_expert_energy, 1e-12)

        # Memory: rank-r stores r × (n_experts + d) values
        # vs n_experts × d values for per-expert residuals
        # In bits: r * (n_experts * 16 + d * 1) vs n_experts * d * 1
        per_expert_bits = n_experts * d  # 1 bit/weight per expert
        lowrank_bits = r * (n_experts * 16 + d)  # 16-bit for U, 1-bit for V
        compression = per_expert_bits / max(lowrank_bits, 1)

        print(f"  Rank-{r:2d}: mean captured {frac_captured.mean():.1%} "
              f"(min {frac_captured.min():.1%}, max {frac_captured.max():.1%})  "
              f"compression vs per-expert: {compression:.1f}x", flush=True)

    # Test: if we use rank-4 low-rank residual, how good is the K3 reconstruction?
    print("\nK3 via low-rank residual (per-expert K2 + rank-r shared residual):", flush=True)
    for r in [1, 2, 4, 8, 16]:
        # Reconstruct rank-r residual for each expert
        # R_r[i] = sum_j eigvecs[i,j] * singular_values[j] * V_j
        # V_j = R^T @ eigvecs[:,j] / singular_values[j]
        # This is expensive (d × 64), so compute in batches
        V_components = np.zeros((r, d), dtype=np.float32)
        for j in range(r):
            if singular_values[j] > 1e-12:
                V_components[j] = R.T @ eigvecs[:, j] / singular_values[j]

        # 1-bit quantize V components (sign × mean)
        V_quant = np.zeros_like(V_components)
        for j in range(r):
            v = V_components[j]
            s = np.abs(v).mean()
            V_quant[j] = np.sign(v) * s if s > 1e-12 else 0

        # Reconstruct: R_approx[i] = sum_j eigvecs[i,j] * singular_values[j] * V_quant[j]
        R_approx = eigvecs[:, :r] @ np.diag(singular_values[:r]) @ V_quant  # (n_experts, d)

        # Measure: for each expert, K2 + R_approx vs K2 + R_true
        # MSE of the residual approximation
        residual_mse = ((R - R_approx) ** 2).mean()
        true_residual_mse = (R ** 2).mean()  # MSE of residual itself (vs 0)
        frac_captured = 1 - residual_mse / true_residual_mse

        print(f"  Rank-{r:2d}: residual MSE={residual_mse:.6e}  "
              f"captures {frac_captured:.1%} of residual energy  "
              f"(1-bit V, {r} components)", flush=True)

    print("\nDone.", flush=True)

if __name__ == "__main__":
    main()
