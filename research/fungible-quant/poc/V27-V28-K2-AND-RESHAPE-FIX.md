# PoC v27b-v28: K2 base tier + reshape bug fix + cross-layer re-verification

## Critical Bug Fix

v26-v27 had a reshape bug: `tiles.reshape(k, n)` doesn't correctly invert
the tile permutation `r.view(tnk, 16, tnn, 16).permute(0, 2, 1, 3).reshape(n_tiles, 256)`.
The correct inverse is `tiles.reshape(tnk, tnn, 16, 16).permute(0, 2, 1, 3).reshape(k, n)`.

This bug caused v26's "universal 1.000x codebook" to be an artifact —
scrambled residuals gave identical garbage for all codebooks.
v27b-v28 fix this with a `tiles_to_matrix()` helper.

## v27b: K2 base tier (first time tested)

K2 (2-bit trellis) as base tier with 64-cluster Lloyd-Max residual:

| Method | bpw | MSE | vs K4 |
|--------|-----|-----|-------|
| K2 | 2 | 1.061e-01 | 1456% |
| K2+1LM | 3 | 3.827e-02 | 525% |
| K2+2LM | 4 | 1.235e-02 | 170% |
| K2+3LM | 5 | 3.780e-03 | 52% |
| K2+4LM | 6 | 1.200e-03 | 16.5% |
| K2+6LM | 8 | 8.811e-05 | 1.2% |

### Best tier at each bpw

| bpw | Best | MSE | Runner-up | Ratio |
|-----|------|-----|-----------|-------|
| 2 | K2 | 1.061e-01 | — | — |
| 3 | K3 | 2.718e-02 | K2+1LM | 1.41x |
| 4 | K4 | 7.288e-03 | K3+1LM | 1.36x |
| 5 | K4+1LM | 2.798e-03 | K3+2LM | 1.18x |
| 6 | K4+2LM | 9.926e-04 | K3+3LM | 1.06x |
| 7 | K4+3LM | 3.109e-04 | K3+4LM | 1.06x |
| **8** | **K2+6LM** | **8.811e-05** | K4+4LM | **1.10x** |

**Crossover at 8 bpw**: K2+6LM beats K4+4LM by 10%! At high bitrates,
K2's larger residual gives the 6-bit LM more signal to capture.

## v28: Cross-layer codebook re-verification (correct reshape)

With the reshape fix, cross-layer codebook sharing still works:
- All 36 combinations (6 train × 6 apply) give ratio ≤ 1.001x
- Codebooks are universal (within 0.01% degradation)
- MSE values now correct (~9.93e-04, matching v25)

## Updated best method

8-tier system covering 2-10 bpw:
- K2 (2 bpw), K3 (3), K4 (4), K4+1LM (5), K4+2LM (6), K4+3LM (7)
- K2+6LM (8 bpw, crossover winner), K3+6LM (9), K4+6LM (10)
- 64-cluster codebooks, universal across layers/projections
- 3-bit tier bitmap (8 tiers, 0.012 bpw overhead)
