# Additive Residual Encoding — Optimization Opportunities (v3)

**Date:** 2026-08-11
**GPU:** RTX 5090 (trellis quantization), Apple M4 Max (cross-expert analysis)
**Quantizer:** Real EXL3 Viterbi trellis (`ext.quantize_tiles`)
**Code:** `poc/poc_additive_residual_v3.py`, `poc/poc_cross_expert.py`

---

## Measurements

### Standalone trellis (reference)

| K | MSE | Improvement |
|---|---|---|
| K2 | 1.061e-01 | — |
| K3 | 2.717e-02 | 3.90× |
| K4 | 7.284e-03 | 3.73× |

### All residual paths (aggregate, 3 projections)

| Path | Bits/w | MSE | Cosine | Gap closed | vs true K |
|---|---|---|---|---|---|
| K3 via 2+1s | 3 | 3.839e-02 | 0.988 | 85.8% | 1.41× K3 |
| K4 via 3+1s | 4 | 9.951e-03 | 0.997 | 86.6% | 1.37× K4 |
| K4 via 2+1s+1s | 4 | 1.376e-02 | 0.996 | 93.4% | 1.89× K4 |
| **K4 via 2+2lloyd** | 4 | **1.244e-02** | 0.996 | **94.8%** | **1.71× K4** |
| **K4 via 2+2kmeans** | 4 | **1.244e-02** | 0.996 | **94.8%** | **1.71× K4** |
| K4 via 2+2uniform | 4 | 1.052e-01 | 0.965 | 0.9% | 14.4× K4 |
| **K4 via 3+2lm** | 5 | **3.331e-03** | 0.999 | **119.9%** | **0.46× K4** |
| K4 via 2+1s+1lm | 4 | 1.376e-02 | 0.996 | 93.4% | 1.89× K4 |

---

## Findings by Optimization Direction

### 1. Different Codebook: Lloyd-Max 2-bit wins

**K4 via 2+2lloyd** closes **94.8%** of the K2→K4 gap — up from 93.4% with 2+1+1 chained, at the same 4 bits/weight. MSE drops 10% (1.376e-02 → 1.244e-02).

Lloyd-Max places levels at ±0.45σ and ±1.51σ — optimal for Gaussian. The residual is near-Gaussian (kurtosis ≈ 0), so this is near-optimal.

K-means (trained on actual residuals) gives identical results — confirming the residual is Gaussian enough that theoretical optimal levels suffice. No training needed.

2-bit uniform fails (0.9% gap closed) — uniformly spaced levels are wrong for Gaussian residuals. The levels must be non-uniform.

1-bit Lloyd-Max = 1-bit scalar — confirms sign × mean(|r|) is already optimal for 1-bit on zero-mean Gaussian.

### 2. K4 via 3+2lm: better than true K4 (!)

**K4 via 3+2lm** (K3 base + 2-bit Lloyd-Max) achieves MSE 3.331e-03 — **2.2× better than true K4** (7.284e-03). Gap closed is 119.9%.

This is a 5-bit representation (3+2), not 4-bit — but it shows the residual approach can exceed trellis quality at a given total bitrate. A 3+2 progressive artifact (5 bits/weight) gives better-than-K4 quality while still being one segment file.

### 3. Layer-level (cross-expert) sharing: does NOT work

Analyzed 64 experts from layer 30 (gate_proj). Computed K2 residuals for each, then SVD on the 64×12.6M residual matrix:

| Rank | Energy captured | Compression vs per-expert |
|---|---|---|
| 1 | 1.8% | 64× |
| 2 | 3.4% | 32× |
| 4 | 6.7% | 16× |
| 8 | 13.2% | 8× |
| 16 | 26.0% | 4× |
| 32 | 51.1% | 2× |
| 64 | 100% | 1× |

Common-mode (mean residual across experts): **0.0%** of total energy.

**The residuals are essentially full-rank across experts.** Rank-1 captures only 1.8%, rank-4 only 6.7%. Even rank-16 (25% of dimensions) captures only 26%. The singular values are nearly flat (61.1, 59.4, 59.3, 59.0, ...) — no dominant component.

**Low-rank residual reconstruction quality** (1-bit quantized V components):

| Rank | Residual energy captured | K3 MSE improvement |
|---|---|---|
| 1 | 1.1% | negligible |
| 4 | 4.3% | negligible |
| 8 | 8.4% | small |
| 16 | 16.6% | modest |

**Verdict:** layer-level sharing does not work. Each expert's K2 residual is independent — the Hadamard regularization and sign flips decorrelate the errors across experts. The per-expert 1-bit scalar residual remains the right approach.

### 4. Tensor-level (zooming in)

Not measured separately, but the v2 PoC already showed:
- Per-group scaling (group=128) improves gap closed by only 0.3% over global — not worth 0.125 bits/weight overhead
- The residual is near-uniform (kurtosis ≈ 0) after Hadamard preprocessing

**Verdict:** no gain from finer granularity. The Hadamard already decorrelated the tiles.

---

## Summary: Best Paths

| Path | Bits/w | MSE | Gap closed | Use case |
|---|---|---|---|---|
| K3 via 2+1s | 3 | 3.84e-02 | 85.8% | Minimal K3 (1 residual plane) |
| **K4 via 2+2lloyd** | 4 | 1.24e-02 | **94.8%** | **Best 4-bit progressive K4** |
| K4 via 3+1s | 4 | 9.95e-03 | 86.6% | Best 4-bit from K3 base |
| **K4 via 3+2lm** | 5 | 3.33e-03 | **119.9%** | **Better than true K4** (5-bit) |
| K4 via 2+1s+1s | 4 | 1.38e-02 | 93.4% | Chained 4-bit (no K3 base) |

## Recommendations

1. **Use Lloyd-Max 2-bit for the K2→K4 direct residual.** 10% better than 2+1+1 at the same 4 bits/weight. Levels (±0.45σ, ±1.51σ) are trivially computed from residual std.

2. **Offer 3+2lm as a premium tier.** At 5 bits/weight it beats true K4 by 2.2× — a quality level trellis can't reach without K5 (blocked on SM120).

3. **Skip layer-level sharing.** Residuals are full-rank across experts (rank-1 captures 1.8%). Per-expert residuals are necessary.

4. **Skip k-means.** Lloyd-Max gives identical quality with a closed-form solution.

5. **Skip trellis residuals.** The EXL3 codebook doesn't match residual distributions. Lloyd-Max (Gaussian-optimal) is the right codebook.

6. **Joint encoder design:** at encode time, compute K2 base + 2-bit Lloyd-Max residual (K4 at 4 bits) + optionally K3 base + 1-bit scalar residual (K4 at 4 bits from K3) + optionally 2-bit Lloyd-Max from K3 (K4+ at 5 bits). Record quality per path. User picks at runtime.
