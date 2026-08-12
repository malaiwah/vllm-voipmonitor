# PoC v26: Cross-layer codebook sharing — universal normalized codebooks

## Finding

Normalized Lloyd-Max codebooks (divided by per-cluster sigma) are **universal**
across all layers and projections. A single set of 64 codebooks trained on
layer 10 gate_proj gives identical quality (ratio = 1.000x) when applied to
any layer (10/40) and any projection (gate/up/down).

## Results

| Train on       | Apply to       | MSE       | vs own  |
|----------------|----------------|-----------|---------|
| L10_gate       | L10_gate       | 1.360e-02 | self    |
| L10_gate       | L10_down       | 1.359e-02 | 1.000x  |
| L10_gate       | L40_gate       | 1.359e-02 | 1.000x  |
| L10_gate       | L40_down       | 1.359e-02 | 1.000x  |
| L40_up         | L10_gate       | 1.359e-02 | 1.000x  |
| ...            | ...            | ...       | 1.000x  |

All 36 combinations (6 train × 6 apply) give ratio = 1.000x.

## Why it works

Hadamard regularization makes the residual distribution identical across:
- Different layers (10 vs 40)
- Different projections (gate vs up vs down)
- Different tensor shapes (2048×6144 vs 6144×2048)

The normalized codebook captures the shape of the distribution (Gaussian-like
after Hadamard), which is universal. The scale (sigma) is per-tile, computed
at runtime.

## Storage impact

- Per-model: 64 codebooks × 4 levels × 4 bytes = 1KB total
- Per-expert: 0 (codebooks are shared across all experts/layers/projections)
- Per-tile: only sigma (1 float) and cluster_id (6 bits for 64 clusters)

This eliminates all codebook overhead — the 0.00008-0.00033 bpw from v25
drops to essentially zero.

## Updated best method

**6-tier tile-level mixed precision with universal normalized codebooks**:
- K3 (3.0 bpw), K4 (4.0 bpw)
- K4+1LM (5.0 bpw), K4+2LM (6.0 bpw) — 64-cluster codebooks
- K3+4LM (7.0 bpw), K4+4LM (8.0 bpw) — 64-cluster codebooks
- Codebook storage: 1KB per model (universal)
- Per-tile storage: sigma (4B) + cluster_id (1B) + tier (2 bits)
- Total overhead: <0.001 bpw
- Quality: 93-70% of per-tile (from v25), better than global
