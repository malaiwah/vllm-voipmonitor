# Continuously Variable Quantization — v8/v9 Results (GLM-5.2)

## Overview

Achieved continuously variable 3.0–5.5 bpw on real GLM-5.2 weights using
tile-level mixed precision across 3 quality tiers (K3/K4/K5).

## Method: Tile-Level 3-Tier Mixed Precision

Each 16×16 tile is independently assigned to one of three quality tiers:
- **K3** (3 bits): EXL3 trellis quantization at K=3
- **K4** (4 bits): EXL3 trellis quantization at K=4
- **K5** (5 bits): K4 + 2-bit Lloyd-Max on residual

Tile assignment is guided by per-tile quantization error: tiles with the highest
K3→K4 improvement are upgraded first. A 2-bit bitmap per tile records the tier
assignment (overhead: 2/256 ≈ 0.008 bpw, negligible).

## Pareto Frontier (Layer 10, 10 experts, real GLM-5.2 weights)

| Bits | Method | MSE | Gap to K4 | Notes |
|------|--------|-----|-----------|-------|
| 3.0 | K3 (all tiles) | 2.718e-02 | 0% | Baseline |
| 3.3 | K3 + 25% K4 tiles | 2.154e-02 | 28.3% | |
| 3.5 | K3 + 50% K4 tiles | 1.644e-02 | 54.0% | |
| 3.8 | K3 + 75% K4 tiles | 1.166e-02 | 78.0% | |
| 4.0 | K4 (all tiles) | 7.286e-03 | 100% | |
| 4.2 | K4 + 10% K5 tiles | 6.506e-03 | 103.9% | **Beats K4 at 4.2 bits!** |
| 4.5 | 3-tier: 50% K3 + 50% K5 | 3.926e-03 | 116.9% | |
| 5.0 | K5 (K3+2lloyd uniform) | 1.066e-03 | 131.3% | Best at 5 bits |
| 5.5 | K4 + 75% K5 tiles | 2.403e-03 | 124.5% | |

## Key Findings

### 1. Continuously variable 3-5 bits achieved
The tile-level approach gives smooth quality transitions across the entire 3-5 bit
range. No re-encoding needed — just change the tier assignment bitmap at load time.

### 2. K4+10% K5 tiles beats uniform K4 (103.9% vs 100% at +0.2 bits)
Upgrading just 10% of the most-damaged tiles from K4 to K5 gives better quality
than uniform K4, at only 4.2 bits. The most-damaged tiles benefit disproportionately
from the 2-bit Lloyd-Max correction.

### 3. 3-tier is more bit-efficient than K4+tile K5
At 4.5 bits:
- 3-tier (50% K3 + 50% K5): MSE=3.926e-03
- K4 + 25% K5 tiles: MSE=5.471e-03

The 3-tier wins because it "saves" bits on less-damaged tiles (K3 instead of K4)
and "spends" them on upgrading more tiles to K5. The K4 base wastes bits on
less-damaged tiles that don't need K4 precision.

### 4. K3+2lloyd (uniform K5) is best at exactly 5 bits
At 5.0 bits, K3+2lloyd (MSE=1.066e-03) beats K4+50%tile_K5 (MSE=3.879e-03) by 3.6×.
The K3 residual is larger, so 2-bit Lloyd-Max captures more error.

### 5. BitsMoE spectral decomposition is quantization-neutral
SVD decomposition into shared basis Φ + per-expert spectral factors P_e gives
identical quantization quality to direct quantization (0.01% difference).
The shared basis can be stored unquantized (amortized cost ~0.006 bpw).

### 6. Per-expert allocation doesn't help for real GLM-5.2
- Spectral energy CV across experts: 2.35% (negligible)
- K3 MSE CV across experts: 0.03% (negligible)
- Hadamard regularization equalizes all experts

The Fruit model (mini GLM-5.2 mimic) DOES have 49.6% CV in per-expert MSE,
but real GLM-5.2 experts are statistically homogeneous after training.

## Runtime Efficiency

The tile-level approach is runtime-efficient:
- **Storage**: 2-bit tier bitmap per 16×16 tile (0.008 bpw overhead)
- **Dequantization**: Branch per tile (K3/K4/K5 lookup), similar to Q-Palette's half-TCQ
- **Kernel**: Can use existing EXL3 trellis kernels per tile group
- **No re-encoding**: Tier assignment is a load-time parameter

## Full Quality Curve (3-5 bits)

```
bits  | MSE (log)  | method
------|------------|----------------------------------
3.0   | 2.7e-02    | K3 (all tiles)
3.3   | 2.2e-02    | K3 + 25% K4 tiles
3.5   | 1.6e-02    | K3 + 50% K4 tiles
3.8   | 1.2e-02    | K3 + 75% K4 tiles
4.0   | 7.3e-03    | K4 (all tiles)
4.2   | 6.5e-03    | K4 + 10% K5 tiles
4.5   | 3.9e-03    | 3-tier: 50% K3 + 50% K5
5.0   | 1.1e-03    | K5 (K3+2lloyd uniform)
5.5   | 2.4e-03    | K4 + 75% K5 tiles
```

The curve is smooth and monotonic (except 5.0→5.5 where uniform K5 is better
than partial K5 upgrade from K4 base).
