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

## MSRT Quality (PoC Measurements, 10 experts, layer 10 gate_proj)

| Config | bpw | MSE | vs K3 | vs K4 | GEMM passes |
|--------|-----|-----|-------|-------|-------------|
| K2 only | 2.0 | 1.061e-01 | 3.90× | 14.6× | 1 |
| K3 only | 3.0 | 2.718e-02 | 1.00× | 3.73× | 1 |
| K4 only | 4.0 | 7.286e-03 | 0.268× | 1.00× | 1 |
| **K2+K1 (MSRT 3bpw)** | **3.0** | **5.144e-04** | **0.019×** | **0.071×** | **2** |
| K2+K3 (MSRT 5bpw) | 5.0 | 1.892e-03 | 0.070× | 0.260× | 2 |
| K2+K1+K3 (MSRT 6bpw) | 6.0 | 5.144e-04 | 0.019× | 0.071× | 3 |
| K2+K1+K2+K3 (MSRT 8bpw) | 8.0 | 3.868e-05 | 0.001× | 0.005× | 4 |

## Feasible Strategies Per Card

### H100 80GB (MoE budget: ~60 GiB/rank)

| Strategy | eff bpw | GiB/rank | Fits? | MSE | vs K3 |
|----------|---------|----------|-------|-----|-------|
| K2 base only | 2.00 | 43.3 | ✓ | 1.06e-01 | 3.90× |
| **K2 + K1 cartridge (top 96)** | **2.38** | **51.4** | **✓** | **~6e-02** | **~2.2×** |
| K2 + K1 cartridge (all 256) | 3.00 | 65.0 | ✗ | 5.14e-04 | 0.019× |
| K3 base only | 3.00 | 65.0 | ✗ | 2.72e-02 | 1.00× |

**Best H100 option: K2 base + K1 cartridge on top 96 experts (37.5%)**
- 51.4 GiB MoE + 9.3 GiB non-MoE = 60.7 GiB total
- Cartridge adds 1 GEMM pass for 96/256 experts (avg 3 per token of 8)
- Runtime overhead: ~0.38 extra GEMM launches per token batch

### H200 141GB (MoE budget: ~110 GiB/rank)

| Strategy | eff bpw | GiB/rank | Fits? | MSE | vs K3 |
|----------|---------|----------|-------|-----|-------|
| K2 + K1 (all) MSRT 3bpw | 3.00 | 65.0 | ✓ | 5.14e-04 | 0.019× |
| Mixed K3/K4 (160/96) | 3.38 | 73.1 | ✓ | ~mixed | ~0.5× |
| K2 + K1+K3 cartridge (top 96) | 3.50 | 75.8 | ✓ | ~1e-03 | ~0.04× |
| K4 base only | 4.00 | 86.6 | ✓ | 7.29e-03 | 0.268× |
| **K2 + K1+K3 cartridge (top 160)** | **4.50** | **97.5** | **✓** | **~5e-04** | **~0.019×** |
| MSRT K2+K3 (5bpw) | 5.00 | 108.3 | ✓ | 1.89e-03 | 0.070× |

**Best H200 option: K2 base + K1+K3 cartridge on top 160 experts (62.5%)**
- 97.5 GiB MoE + 9.3 GiB non-MoE = 106.8 GiB total
- Cartridge: 2 extra GEMM passes for 160/256 experts (avg 5 per token)
- Quality approaches full MSRT 6bpw at 75% of the memory

### B200 192GB (MoE budget: ~155 GiB/rank)

| Strategy | eff bpw | GiB/rank | Fits? | MSE | vs K3 |
|----------|---------|----------|-------|-----|-------|
| MSRT K2+K1+K3 (6bpw) | 6.00 | 129.9 | ✓ | 5.14e-04 | 0.019× |
| MSRT K2+K1+K2+K3 (8bpw) | 8.00 | 173.3 | ✗ | 3.87e-05 | 0.001× |

**Best B200 option: MSRT K2+K1+K3 (6bpw, all experts)**
- 129.9 GiB MoE + 9.3 GiB non-MoE = 139.2 GiB total
- 3 GEMM passes per expert (base + 2 cartridge stages)
- 53× better than K3 at 2× the memory

## Recommendation: Base K and Additive Bits

### Base weight: **K2 (2 bits)**

- K2 is the lowest viable base (K1 has MSE 0.106 — too lossy)
- K2 uses 48.6 GiB/rank, leaving budget for cartridge
- K2+K1 MSRT at same 3bpw as K3 is **53× better** in MSE

### Additive cartridge options (per expert, selectively applied):

| Cartridge | Add bits | Total bpw | Extra GEMMs | Quality gain |
|-----------|---------|-----------|-------------|-------------|
| +K1 | 1 | 3.0 | +1 | 53× over K3 (at same bpw) |
| +K1+K3 | 4 | 6.0 | +2 | 14× over K4 |
| +K1+K2+K3 | 6 | 8.0 | +3 | 188× over K4 |

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

For prefill (large batch), GEMMs are compute-bound. Extra stages at same total
bitrate have same total FLOPs — overhead is just launch latency.
