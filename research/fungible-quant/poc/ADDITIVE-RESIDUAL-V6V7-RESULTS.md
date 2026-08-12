# Continuously Variable Per-Expert Sparse Quantization — v6/v7 Results

## Overview

Pivoted from fixed-bitwidth-per-expert to **continuously variable per-expert allocation**
using two approaches:
- v6: K3 base + sparse fp16 corrections (top-k% of residual by magnitude)
- v7: K3 base + tile-level K3→K4 upgrades (16×16 tiles)

Tested on **real GLM-5.2 weights**: 70 experts from layer 10 and layer 40,
each expert gate_proj shape (2048, 6144), 12.6M weights.

## Key Finding: Per-Expert Damage is Uniform

After Hadamard regularization, all experts have nearly identical quantization damage:
- K3 MSE range across 70 experts: [2.7164e-02, 2.7191e-02] (0.1% variation)
- K4 MSE range: [7.266e-03, 7.302e-03] (0.5% variation)
- Disambiguation weight (distance from layer mean): 0.9919–0.9931 (0.1% variation)

**Implication:** Water-filling and disambiguation-weighted allocation provide NO benefit
over uniform allocation. The Hadamard transform equalizes per-expert damage.

## v6: Sparse fp16 Corrections — INEFFICIENT

K3 base + sparse fp16 correction on top-k% of K3 residual:

| Extra bpw | Sparse % | Total bits | MSE | Gap to K4 |
|-----------|----------|------------|-----|-----------|
| 0.1 | 0.45% | 3.114 | 2.570e-02 | 7.4% |
| 0.25 | 1.14% | 3.273 | 2.439e-02 | 14.0% |
| 0.5 | 2.27% | 3.523 | 2.265e-02 | 22.8% |
| 1.0 | 4.55% | 4.000 | 1.990e-02 | 36.6% |
| 2.0 | 9.09% | 4.909 | 1.582e-02 | 57.1% |

**Problem:** Each sparse correction costs ~22 bits/entry (16-bit value + 6-bit index),
making it 11× less bit-efficient than Lloyd-Max 2-bit (2 bits/weight).

## v7: Tile-Level K3→K4 Upgrade — EFFICIENT

K3 base + upgrade top-k% most-damaged 16×16 tiles to K4:

| Upgrade % | Total bits | MSE | Gap to K4 |
|------------|------------|-----|-----------|
| 1% | 3.014 | 2.691e-02 | 1.3% |
| 5% | 3.054 | 2.595e-02 | 6.2% |
| 10% | 3.104 | 2.480e-02 | 11.9% |
| 25% | 3.254 | 2.154e-02 | 28.3% |
| 50% | 3.504 | 1.644e-02 | 54.0% |
| 75% | 3.754 | 1.166e-02 | 78.0% |
| 100% | 4.000 | 7.286e-03 | 100.0% (= K4) |

**This is the continuously variable bpw the user envisioned:** smooth quality
curve from 3.0 to 4.0 bits, controlled by a single parameter (upgrade fraction).
Index overhead: 1 bit per tile = 0.004 bpw (negligible).

## 3-Tier: K3 + K4 tiles + K5 tiles

Extend to 3 quality tiers per tile:

| K4 frac | K5 frac | Total bits | MSE | Gap to K4 |
|---------|---------|------------|-----|-----------|
| 0.10 | 0.10 | 3.408 | 2.186e-02 | 26.7% |
| 0.25 | 0.25 | 4.008 | 1.467e-02 | 62.9% |
| 0.00 | 0.50 | 4.508 | 1.308e-02 | 70.9% |
| 0.50 | 0.25 | 4.258 | 9.894e-03 | 86.9% |
| 0.25 | 0.50 | 4.758 | 8.302e-03 | 94.9% |

## Pareto Frontier

| Bits | Best method | MSE |
|------|-------------|-----|
| 3.0 | K3+tile_K4_10% | 2.480e-02 |
| 3.5 | K3+tile_K4_50% | 1.644e-02 |
| 4.0 | K4 standalone | 7.286e-03 |
| 4.5 | K3+3tier K4_50%+K5_25% | 9.894e-03 |
| 5.0 | K3+2lloyd | 3.333e-03 |

**Note:** At 4.5 bits, K4+2lloyd uniformly applied would be better than 3-tier
(because K3 tiles in 3-tier drag down the average). The 3-tier approach is
most useful for 3-4 bit range where K3→K4 tile upgrades dominate.

## Conclusions

1. **Tile-level K3→K4 upgrade is the best continuously variable method for 3-4 bits.**
   Smooth quality curve, minimal index overhead, runtime-efficient (1-bit bitmap per tile).

2. **Per-expert allocation does NOT help for GLM-5.2** — Hadamard regularization
   makes all experts equally damaged. Water-filling reduces to uniform.

3. **Sparse fp16 corrections are 11× less efficient than tile upgrades** —
   22 bits/entry vs 1 bit/weight for tile K3→K4.

4. **For 4-5 bit range, K4+2lloyd uniformly applied is better than 3-tier mixing.**
   The 3-tier approach only wins when the base is K3 (3-4 bit range).

5. **The fungibility story:** A single encoded model with tile-level bitmap
   supports continuously variable 3.0-4.0 bpw by controlling the K3→K4 upgrade
   fraction at load time. No re-encoding needed.
