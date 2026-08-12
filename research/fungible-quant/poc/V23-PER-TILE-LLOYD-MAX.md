# PoC v23b: Per-tile Lloyd-Max codebooks

## Finding

Per-tile Lloyd-Max codebooks give 17-41% MSE improvement over per-expert
(global) codebooks, with manageable codebook overhead.

| Method              | bpw   | MSE       | vs global |
|---------------------|-------|-----------|-----------|
| K4+2LM_global       | 6.000 | 1.070e-03 | 100%      |
| K4+2LM_row128       | 6.000 | 1.070e-03 | 100% (no benefit) |
| K4+2LM_tile         | 6.062 | 8.894e-04 | 83.1% (17% better) |
| K4+4LM_global       | 8.000 | 1.360e-04 | 100%      |
| K4+4LM_tile         | 8.250 | 7.958e-05 | 58.5% (41% better) |

## Codebook overhead

Per-tile codebook: 2^N levels × 4 bytes / 256 weights per tile:
- 2-bit: 4 × 4 / 256 = 0.0625 bpw
- 4-bit: 16 × 4 / 256 = 0.25 bpw
- 6-bit: 64 × 4 / 256 = 1.0 bpw (too much)
- 8-bit: 256 × 4 / 256 = 4.0 bpw (prohibitive)

For 6+ bit codebooks, per-expert (global) is better (overhead amortized).
For 2-4 bit codebooks, per-tile is better (quality gain >> overhead).

## Per-row-group provides no benefit

Per-row-group LM (128 rows = 8 tile-rows per group) gives identical results
to global LM. This confirms that Hadamard regularization equalizes the
residual distribution across all row groups — there's no spatial structure
left to exploit.

## Vectorized implementation

v23 used a Python for-loop over 49152 tiles (900s timeout).
v23b uses batched GPU operations (reshape to (n_tiles, 256), vectorized
distance/assignment). Runtime: <0.1s per expert. 9000× speedup.

## Updated best method

For tiers with ≤4-bit residual codebooks, use per-tile codebooks:
- K4+2LM_tile (6.06 bpw): 17% better than K4+2LM_global
- K3+4LM_tile (7.25 bpw): 18% better than K3+4LM_global

For tiers with 6+ bit codebooks, use per-expert (global) codebooks:
- K4+6LM_global (10 bpw): overhead of per-tile would be 1.0 bpw
- K4+8LM_global (12 bpw): overhead of per-tile would be 4.0 bpw

## Runtime considerations

Per-tile codebook storage: 4-16 float values per tile. For 49152 tiles:
- 2-bit: 4 floats × 49152 = 786KB per expert
- 4-bit: 16 floats × 49152 = 3.1MB per expert

Dequantization: lookup codebook[level_index] per tile — same cost as
global codebook lookup, just different codebook per tile. Can be
implemented as a gather operation.
