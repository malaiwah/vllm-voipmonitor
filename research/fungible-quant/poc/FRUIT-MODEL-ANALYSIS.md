# Fruit Model Analysis: Per-Expert Heterogeneity is a Tier Artifact

## Finding

The Fruit model (GLM-5.2-SIQ-Fruit-Instruct) has CV=49.66% in per-expert
MSE — but this is NOT evidence of genuine per-expert quality variation.

The CV is caused by the K3/K4 tier split within each layer (160 K3 + 96 K4),
not by individual expert quality differences:

| K tier | n_experts | mean MSE    | within-tier CV |
|--------|-----------|-------------|----------------|
| K3     | 160       | 1.7434e-02  | <0.1%          |
| K4     | 96        | 4.5378e-03  | <0.1%          |
| Mixed  | 256       | 1.2598e-02  | 49.66% (artifact)|

The 3.88× MSE ratio between K3 and K4 experts creates the apparent
heterogeneity. Within each tier, experts are just as homogeneous as
real GLM-5.2 (CV < 0.1%).

## Implication

Per-expert allocation provides **zero benefit even for the Fruit model**.
The only useful allocation dimension is the tier (K3 vs K4) itself,
which is already a per-layer decision in the Fruit model.

## Fruit model structure

- Layers 3-12: 160 K3 + 96 K4 (62.5%/37.5%) → 3.375 bpw avg
- Layer 13: 256 K3 (3.0 bpw)
- All MoE layers have identical tier split
- Per-expert MSE within each tier varies <0.1% (statistically homogeneous)

This confirms the universal finding across all our experiments:
**GLM-5.2 experts (both real and Fruit) are statistically homogeneous
after Hadamard regularization. Per-expert allocation never helps.**
