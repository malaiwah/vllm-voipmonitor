# PoC v50: Low-bitrate MSRT measurements (3-5 bpw) — MSRT advantage starts at 5bpw

## Purpose

Previous MSRT Pareto (v41-v46) started at 5bpw. This experiment fills the gap
at 3-4 bpw to determine whether MSRT provides any advantage at low bitrates.

## Results (10 experts, layer 10 gate_proj, validated on layer 40)

### 3 bpw — K3 wins

| Config | MSE | vs K3 |
|--------|-----|-------|
| **K3 only** | **2.718e-02** | **1.00×** |
| K2+K1trsc | 2.908e-02 | 1.07× worse |
| K2+K1 (no rescale) | 1.783e-01 | 6.56× worse |

K2+K1trsc is 7% worse than K3 at the same 3bpw. The 1-bit rescaled trellis
residual is too small to improve on K2's quantization error.

### 4 bpw — K4 and K2+K2trsc tied

| Config | MSE | vs K4 |
|--------|-----|-------|
| **K4 only** | **7.286e-03** | **1.00×** |
| K2+K2trsc | 7.305e-03 | 1.00× (tie, 0.3% worse) |
| K3+K1trsc | 7.515e-03 | 1.03× worse |
| K2+K1trsc+K1trsc | 7.964e-03 | 1.09× worse |

### 5 bpw — MSRT advantage begins

| Config | MSE | vs K3 | vs K4 |
|--------|-----|-------|-------|
| **K2+K3trsc** | **1.892e-03** | **0.070×** | **0.260×** |
| K3+K2trsc | 1.952e-03 | 0.072× | 0.268× |
| K2+K1trsc+K2trsc | 1.995e-03 | 0.073× | 0.274× |

## Key Finding

**MSRT provides no advantage below 5bpw.** The crossover point is at 5bpw:

| bpw | Best method | MSE |
|-----|------------|-----|
| 2 | K2 only | 1.061e-01 |
| 3 | K3 only | 2.718e-02 |
| 4 | K4 only (≈ K2+K2trsc) | 7.286e-03 |
| **5** | **K2+K3trsc (MSRT)** | **1.892e-03** |
| 6 | K2+K1trsc+K3trsc (MSRT) | 5.144e-04 |

At 3bpw, K3's single-tier trellis captures the weight distribution better than
K2+K1's two-stage approach because the K1 residual stage (1 bit) is too coarse
to meaningfully correct K2's errors. The rescaling doesn't help here because the
residual after K2 has too much structure for 1-bit trellis to capture.

At 4bpw, K2+K2trsc matches K4 (7.305e-03 vs 7.286e-03) — the 2-bit rescaled
residual is sufficient to match single-tier K4, but not beat it.

At 5bpw, the 3-bit rescaled residual (K3trsc) on K2's base finally provides
enough resolution to outperform single-tier approaches by 3.8× over K4.

## Correction to Cost Analysis

The previous cost analysis (commit cc38fc6487) incorrectly claimed MSRT K2+K1
at 3bpw gives MSE 5.144e-04 (53× better than K3). This was wrong — 5.144e-04
is the MSE for K2+K1+K3 at **6bpw**, not K2+K1 at 3bpw. The actual K2+K1trsc
at 3bpw MSE is 2.908e-02 (7% worse than K3). The corrected cost analysis is in
GLM52-MSRT-COST-ANALYSIS.md.
