# PoC v24: Per-tile vs global LM Pareto comparison

## Result

Per-tile Lloyd-Max codebooks dominate global LM on the Pareto frontier,
with improvement growing from 1.6% (5 bpw) to 70.3% (8.5 bpw).

| bpw | Global LM MSE | Per-tile LM MSE | Improvement |
|-----|---------------|-----------------|-------------|
| 5.0 | 2.813e-03     | 2.768e-03       | 1.6%        |
| 5.5 | 1.860e-03     | 1.761e-03       | 5.3%        |
| 6.0 | 1.058e-03     | 9.750e-04       | 7.8%        |
| 6.5 | 6.849e-04     | 5.907e-04       | 13.8%       |
| 7.0 | 4.733e-04     | 3.803e-04       | 19.6%       |
| 7.5 | 3.485e-04     | 2.340e-04       | 32.8%       |
| 8.0 | 2.693e-04     | 1.264e-04       | 53.1%       |
| 8.5 | 2.693e-04     | 8.001e-05       | 70.3%       |

## Why improvement grows with bpw

At low bpw (5-6), most tiles are at the base tier (K4+1LM), so the
per-tile codebook only affects a few upgraded tiles. At high bpw (7-8+),
most tiles use 4-bit codebooks where per-tile gives 41% improvement vs
global, and the codebook overhead (0.25 bpw) is small relative to the
total bitrate.

## Updated best method

**6-tier tile-level mixed precision with per-tile codebooks**:
- K3 (3.0 bpw), K4 (4.0 bpw)
- K4+1LM_tile (5.03 bpw), K4+2LM_tile (6.06 bpw)
- K3+4LM_tile (7.25 bpw), K4+4LM_tile (8.25 bpw)
- Per-tile codebook for ≤4-bit LM, global codebook for 6+ bit LM
- DP-optimal tile assignment, continuously variable 3.0-8.25 bpw
- 2-bit tier bitmap (0.0078 bpw) + per-tile codebook overhead (0.03-0.25 bpw)
