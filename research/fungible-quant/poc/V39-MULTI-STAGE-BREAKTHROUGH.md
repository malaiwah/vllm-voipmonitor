# PoC v39: Multi-stage rescaled trellis — 17% better than single, K6trsc beats LM at 8bpw!

## THREE BREAKTHROUGHS

### 1. Two-stage rescaled trellis beats single large trellis

| Method | bpw | MSE | vs K2+K5trsc |
|--------|-----|-----|--------------|
| K2+K5trsc (single) | 7 | 1.733e-04 | baseline |
| K2+K3trsc+K2trsc (2-stage) | 7 | 1.531e-04 | **12% better** |
| K2+K2trsc+K3trsc (2-stage) | 7 | 1.437e-04 | **17% better!** |
| K2+K2trsc+K2trsc (2-stage) | 6 | 5.172e-04 | 2% better than K2+K4trsc |

**Key insight**: Splitting the residual budget across two trellis stages is
better than using one large trellis. This is the successive refinement
property of TCQ (Jafarkhani 1999) in action! The first stage captures the
bulk of the residual, and the second stage refines the remaining error.

**Order matters**: K2+K2trsc+K3trsc (small first, large second) is better
than K2+K3trsc+K2trsc (large first, small second). The first residual
stage should use fewer bits, leaving a more Gaussian residual for the
second stage.

### 2. K2+K6trsc beats LM at 8 bpw!

| Method | bpw | MSE |
|--------|-----|-----|
| K2+6LM (c128) | 8 | 8.773e-05 |
| K2+K6trsc | 8 | 8.129e-05 (**7% better**) |

**Rescaled trellis now wins at 8 bpw too!** This extends the trellis advantage
from 5-7 bpw to 5-8 bpw. The crossover with LM has shifted to 9+ bpw.

### 3. Per-tile rescaling slightly better than global

| Method | bpw | Global MSE | Per-tile MSE | Improvement |
|--------|-----|-----------|-------------|-------------|
| K2+K4trsc | 6 | 5.276e-04 | 5.124e-04 | 3% |
| K2+K5trsc | 7 | 1.733e-04 | 1.606e-04 | 7% |

Per-tile rescaling gives 3-7% improvement but is much slower (Python loop).
For production, global rescaling is practical; per-tile is an option for
maximum quality.

## Updated Best Tiers (v39)

| bpw | Best tier | MSE | vs prev best |
|-----|-----------|-----|--------------|
| 2 | K2 | 1.061e-01 | — |
| 3 | K3 | 2.718e-02 | — |
| 4 | K4 | 7.286e-03 | — |
| 5 | K2+K3trsc | 1.892e-03 | — |
| 6 | K2+K2trsc+K2trsc | 5.172e-04 | 2% better than K2+K4trsc |
| **7** | **K2+K2trsc+K3trsc** | **1.437e-04** | **17% better than K2+K5trsc!** |
| **8** | **K2+K6trsc** | **8.129e-05** | **7% better than K2+6LM!** |
| 9 | K3+6LM | 2.629e-05 | K2+K7trsc: 5.757e-05 (LM wins) |
| 10 | K4+6LM | 9.613e-06 | — |

## Why two-stage works

The first trellis stage (K2) captures the bulk of the residual distribution.
The second residual (after subtracting K2trsc) is:
1. Smaller in magnitude (less dynamic range)
2. More Gaussian (central limit effect from subtracting structured quantization)
3. Better matched to the trellis codebook (after rescaling)

This is analogous to residual vector quantization (RVQ) but using TCQ instead
of VQ for each stage. The successive refinement property of TCQ ensures
optimal rate allocation across stages.
