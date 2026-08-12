# GLM-5.2 MSRT Cost Analysis: Memory, Performance, and Configuration

## Model Parameters

| Component | Parameters | Notes |
|-----------|-----------|-------|
| MoE experts (routed) | 724.8B | 256 experts × 75 layers × 3 projections × (2048×6144) |
| Non-MoE (attn, embed, shared expert) | 34.5B | MLA attention, embeddings, shared expert |
| Total | 759.3B | |

## Existing EXL3 Checkpoints (Real Measurements)

| Quant | Tier split | bpw | GiB/rank (TP4) | Source |
|-------|-----------|-----|-----------------|--------|
| brandonmusic 3.0 | 256 K3 | 3.000 | 64.97 | Real checkpoint |
| willfalco 3.36 | 160 K3 + 96 K4 | 3.375 | ~73.09 | Real checkpoint |
| willfalco 3.42 | 148 K3 + 108 K4 | 3.422 | 74.07 | Real checkpoint |

Scaling factor: **21.66 GiB per bpw** (at TP4, from 64.97 / 3.0).

## Card Budgets (TP4, per-rank GiB for MoE weights)

Non-MoE at FP8 consumes ~9.3 GiB/rank. Remaining budget:

| Card | Total VRAM | Weight budget | Non-MoE (FP8) | MoE budget | Max MoE bpw |
|------|-----------|---------------|---------------|------------|-------------|
| RTX 5090 | 32 GB | ~24 GiB | 9.3 GiB | 14.7 GiB | 0.68 |
| H100 | 80 GB | ~70 GiB | 9.3 GiB | 60.7 GiB | 2.80 |
| H200 | 141 GB | ~120 GiB | 9.3 GiB | 110.7 GiB | 5.11 |
| B200 | 192 GB | ~165 GiB | 9.3 GiB | 155.7 GiB | 7.19 |

**RTX 5090 cannot fit GLM-5.2 at any useful MoE bitrate at TP4.** Needs TP8+ or datacenter GPU.

## Measured Quality — Complete Pareto (v50, 10 experts, layer 10 + layer 40)

All values measured with real EXL3 trellis on RTX 5090. MSRT = rescaled trellis
residual stages. "trsc" = rescaled trellis on residual.

### 2 bpw

| Config | MSE | GEMM passes |
|--------|-----|-------------|
| K2 only | 1.061e-01 | 1 |

### 3 bpw — K3 wins, MSRT gives no advantage

| Config | MSE | vs K3 | GEMM passes |
|--------|-----|-------|-------------|
| **K3 only** | **2.718e-02** | **1.00×** | **1** |
| K2+K1trsc | 2.908e-02 | 1.07× worse | 2 |
| K2+K1 (no rescale) | 1.783e-01 | 6.56× worse | 2 |

**At 3bpw, K3 is strictly better than K2+K1trsc (by 7%).** The 1-bit rescaled
trellis residual is too small to improve on K2's quantization error.

### 4 bpw — K4 and K2+K2trsc tied

| Config | MSE | vs K4 | GEMM passes |
|--------|-----|-------|-------------|
| **K4 only** | **7.286e-03** | **1.00×** | **1** |
| K2+K2trsc | 7.305e-03 | 1.00× (tie) | 2 |
| K3+K1trsc | 7.515e-03 | 1.03× worse | 2 |
| K2+K1trsc+K1trsc | 7.964e-03 | 1.09× worse | 3 |

### 5 bpw — MSRT advantage begins

| Config | MSE | vs K3 | vs K4 | GEMM passes |
|--------|-----|-------|-------|-------------|
| **K2+K3trsc** | **1.892e-03** | **0.070×** | **0.260×** | **2** |
| K3+K2trsc | 1.952e-03 | 0.072× | 0.268× | 2 |
| K2+K1trsc+K2trsc | 1.995e-03 | 0.073× | 0.274× | 3 |

**At 5bpw, MSRT is 14× better than K3 and 3.8× better than K4.**

### 6+ bpw (from v41-v46)

| Config | bpw | MSE | vs K3 | vs K4 | GEMM passes |
|--------|-----|-----|-------|-------|-------------|
| K2+K1trsc+K3trsc | 6.0 | 5.144e-04 | 0.019× | 0.071× | 3 |
| K2+K1trsc+K4trsc | 7.0 | 1.415e-04 | 0.005× | 0.019× | 3 |
| K2+K1+K2+K3trsc | 8.0 | 3.868e-05 | 0.001× | 0.005× | 4 |
| K2+K1+K1+K2+K3trsc | 9.0 | 1.095e-05 | 0.0004× | 0.0015× | 5 |
| K2+K1+K1+K1+K2+K3trsc | 10.0 | 3.381e-06 | 0.0001× | 0.0005× | 6 |

### Key Insight: Where MSRT Wins

| Bitrate range | Best method | Why |
|---------------|------------|-----|
| 2-4 bpw | Single-tier trellis (K2, K3, K4) | Residual too large for small K to help |
| 5-7 bpw | MSRT (K2 base + K3-K5 rescaled) | Residual fits trellis codebook well after rescaling |
| 8-10 bpw | MSRT (K2 base + progressive K1 + K2/K3) | Successive refinement optimal |

**MSRT provides no advantage below 5bpw.** At 3bpw, K3 is 7% better than
K2+K1trsc. At 4bpw, K4 and K2+K2trsc are tied. The MSRT advantage starts
at 5bpw and grows with bitrate.

## Feasible Strategies Per Card

### H100 80GB (MoE budget: ~60 GiB/rank)

| Strategy | eff bpw | GiB/rank | Fits? | MSE | Notes |
|----------|---------|----------|-------|-----|-------|
| K2 base only | 2.00 | 43.3 | ✓ | 1.06e-01 | Lowest quality |
| **K2 + K2trsc (top 96)** | **2.75** | **54.1** | **✓** | **~7e-03** | **K4-quality on 37.5% of experts** |
| K3 base only | 3.00 | 65.0 | ✗ | 2.72e-02 | Doesn't fit with FP8 non-MoE |

**Best H100 option: K2 base (all) + K2trsc cartridge (top 96 experts)**
- 54.1 GiB MoE + 9.3 GiB non-MoE = 63.4 GiB — fits in 70 GiB budget
- Cartridge experts get K4-equivalent quality (MSE 7.3e-03)
- Non-cartridge experts stay at K2 (MSE 1.06e-01)
- 2 GEMM passes for cartridge experts, 1 for non-cartridge

**Note: K3 at 3bpw (65.0 GiB) does NOT fit** on H100 with FP8 non-MoE
(65.0 + 9.3 = 74.3 GiB > 70 GiB budget). It fits only with FP4 non-MoE
(65.0 + 4.3 = 69.3 GiB), leaving almost no room for KV cache.

### H200 141GB (MoE budget: ~110 GiB/rank)

| Strategy | eff bpw | GiB/rank | Fits? | MSE | Notes |
|----------|---------|----------|-------|-----|-------|
| K3 base only | 3.00 | 65.0 | ✓ | 2.72e-02 | Comfortable fit |
| Mixed K3/K4 (160/96) | 3.38 | 73.1 | ✓ | ~mixed | Current willfalco quant |
| K4 base only | 4.00 | 86.6 | ✓ | 7.29e-03 | |
| **K2+K3trsc (5bpw, all)** | **5.00** | **108.3** | **✓** | **1.89e-03** | **MSRT: 14× better than K3** |
| K2+K1trsc+K3trsc (6bpw) | 6.00 | 129.9 | ✗ | 5.14e-04 | Doesn't fit |

**Best H200 option: MSRT K2+K3trsc (5bpw, all experts)**
- 108.3 GiB MoE + 9.3 GiB non-MoE = 117.6 GiB — fits in 120 GiB budget
- MSE 1.892e-03 — 14× better than K3, 3.8× better than K4
- 2 GEMM passes per expert

### B200 192GB (MoE budget: ~155 GiB/rank)

| Strategy | eff bpw | GiB/rank | Fits? | MSE | Notes |
|----------|---------|----------|-------|-----|-------|
| K2+K1trsc+K3trsc (6bpw) | 6.00 | 129.9 | ✓ | 5.14e-04 | 53× better than K3 |
| K2+K1+K2+K3trsc (8bpw) | 8.00 | 173.3 | ✗ | 3.87e-05 | Needs TP8 |

**Best B200 option: MSRT K2+K1trsc+K3trsc (6bpw, all experts)**
- 129.9 GiB MoE + 9.3 GiB non-MoE = 139.2 GiB — fits in 155 GiB budget
- MSE 5.144e-04 — 53× better than K3, 14× better than K4
- 3 GEMM passes per expert

## Recommendation: Base K and Additive Bits

### Base weight choice depends on target bitrate

| Target bpw | Base K | Rationale |
|------------|--------|-----------|
| 2-4 bpw | K2, K3, or K4 (single-tier) | MSRT gives no advantage below 5bpw |
| 5-7 bpw | **K2** (2 bits) | K2's larger residual gives MSRT more signal |
| 8-10 bpw | **K2** (2 bits) | K2 base + progressive refinement optimal |

**K2 is the best base for MSRT at 5+ bpw** because its larger residual
(σ≈0.29 vs K3's σ≈0.09) gives the rescaled trellis more signal to work with.

### Additive cartridge options (for 5+ bpw only):

| Cartridge | Add bits | Total bpw | Extra GEMMs | Quality vs K3 | Quality vs K4 |
|-----------|---------|-----------|-------------|---------------|---------------|
| +K3 | 3 | 5.0 | +1 | 14× better | 3.8× better |
| +K1+K3 | 4 | 6.0 | +2 | 53× better | 14× better |
| +K1+K2+K3 | 6 | 8.0 | +3 | 703× better | 188× better |

**Warning: These cartridge options only make sense at 5+ bpw.**
At 3-4 bpw, single-tier trellis (K3 or K4) is equal or better.

### Cartridge selectivity (group of experts):

The cartridge can be applied to a subset of experts. Since GLM-5.2 experts
are statistically homogeneous (CV=0.11% across 70 experts, v48/v49), expert
selection should be based on **routing frequency** (which experts are activated
most often by the router), not on per-expert weight difficulty.

- **top 96 (37.5%)**: Matches Fruit model's K4 tier — the most-routed experts
- **top 160 (62.5%)**: Wider coverage, better quality at higher memory cost
- **all 256 (100%)**: Full MSRT, highest quality, maximum memory

### Runtime cost of cartridge:

Each cartridge stage adds 1 GEMM launch per expert per token batch. For decode
with 8 active experts and cartridge on 96 experts (37.5%):
- Avg cartridge experts per token: 8 × 96/256 = 3
- Extra GEMM launches: 3 per token (vs 8 base GEMMs)
- Overhead: 37.5% more launches, same bandwidth (total bits unchanged)
- **Negligible for decode** (launches are μs-scale, GEMMs are ms-scale)
