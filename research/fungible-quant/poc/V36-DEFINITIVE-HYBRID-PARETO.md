# PoC v36: Definitive hybrid Pareto — rescaled trellis + Lloyd-Max

## NEW BEST TIERS (rescaled trellis breakthrough)

The rescaled trellis-on-residual works even better with K2/K3 bases than K4!

| bpw | Best tier | MSE | vs previous best | Improvement |
|-----|-----------|-----|------------------|-------------|
| 2 | K2 | 1.061e-01 | — | — |
| 3 | K3 | 2.718e-02 | — | — |
| 4 | K4 | 7.286e-03 | — | — |
| **5** | **K2+K3trsc** | **1.892e-03** | K4+1LM: 2.795e-03 | **32% better** |
| **6** | **K2+K4trsc** | **5.276e-04** | K4+2LM: 9.885e-04 | **47% better** |
| **7** | **K3+K4trsc** | **2.139e-04** | K4+3LM: 3.085e-04 | **31% better** |
| 8 | K2+6LM | 8.767e-05 | K4+4LM: 9.540e-05 | 8% better |
| 9 | K3+6LM | 2.629e-05 | — | — |
| 10 | K4+6LM | 9.613e-06 | — | — |

## Why K2 base + rescaled trellis is best for 5-6 bpw

K2 trellis has a larger residual (σ ≈ 0.29) than K4 (σ ≈ 0.085).
The larger residual:
1. Gives the trellis more signal to work with
2. Is more Gaussian (closer to the trellis codebook design)
3. Allows the trellis's 2^L states to capture more information

The trellis codebook (designed for Gaussian σ ≈ |cbs|) works best when the
rescaled residual has similar characteristics to the original weights. K2's
larger residual (after rescaling) is closer to this ideal than K4's smaller
residual.

## Why Lloyd-Max wins at 8+ bpw

At 4+ bit residual (8+ bpw total):
- LM with c128 clusters has 128 × 16 = 2048 effective levels
- This adapts to per-cluster σ, which the fixed trellis codebook can't do
- LM's adaptive clustering gives it the edge at high bitrates

## Updated best method (v36)

Hybrid 9-tier system:
- K2 (2 bpw), K3 (3), K4 (4): trellis only
- K2+K3trsc (5 bpw): rescaled trellis on K2 residual
- K2+K4trsc (6 bpw): rescaled trellis on K2 residual
- K3+K4trsc (7 bpw): rescaled trellis on K3 residual
- K2+6LM (8 bpw): Lloyd-Max on K2 residual
- K3+6LM (9 bpw): Lloyd-Max on K3 residual
- K4+6LM (10 bpw): Lloyd-Max on K4 residual

This is a MAJOR improvement over the previous best (v31 Pareto):
- 5 bpw: 32% better
- 6 bpw: 47% better
- 7 bpw: 31% better
