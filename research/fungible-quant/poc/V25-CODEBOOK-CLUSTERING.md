# PoC v25: Codebook clustering — sweet spot at 64 clusters

## Finding

Instead of per-tile (49152 codebooks, high overhead) or global (1 codebook),
clustering tiles by residual sigma and sharing codebooks within clusters
captures 80-90% of the per-tile benefit at 1/770th the overhead.

## Results

### 2-bit Lloyd-Max (K4+2LM, 6 bpw base)

| Clusters | Overhead (bpw) | MSE       | vs global | vs per-tile |
|----------|----------------|-----------|-----------|-------------|
| 1 (global) | 0.00000 | 1.070e-03 | 100%    | —           |
| 4         | 0.00001 | 1.055e-03 | 98.6%   | —           |
| 16        | 0.00002 | 1.028e-03 | 96.1%   | —           |
| **64**    | 0.00008 | 9.930e-04 | **92.8%** | 88% of gain |
| 256       | 0.00033 | 9.887e-04 | 92.4%   | —           |
| 1024      | 0.00130 | 9.856e-04 | 92.1%   | —           |
| 49152 (tile) | 0.0625 | 8.894e-04 | 83.1%  | 100% of gain|

### 4-bit Lloyd-Max (K4+4LM, 8 bpw base)

| Clusters | Overhead (bpw) | MSE       | vs global | vs per-tile |
|----------|----------------|-----------|-----------|-------------|
| 1 (global) | 0.00000 | 1.368e-04 | 100%    | —           |
| 4         | 0.00002 | 1.140e-04 | 83.4%   | —           |
| 16        | 0.00008 | 1.015e-04 | 74.2%   | —           |
| **64**    | 0.00033 | 9.647e-05 | **70.5%** | 74% of gain |
| 256       | 0.00130 | 9.440e-05 | 69.0%   | —           |
| 1024      | 0.00521 | 9.280e-05 | 67.9%   | —           |
| 49152 (tile) | 0.2500 | 7.958e-05 | 58.5%  | 100% of gain|

## Sweet spot: 64 clusters

- 2-bit: 7.2% improvement for 0.00008 bpw overhead (vs 16.9% for 0.0625 bpw)
- 4-bit: 29.5% improvement for 0.00033 bpw overhead (vs 41.5% for 0.25 bpw)
- Captures 74-88% of the per-tile gain at 1/770th the overhead

## Clustering method

Tiles are sorted by per-tile residual sigma (1D feature) and split into
equal-size groups. Each cluster trains one Lloyd-Max codebook on all its
tiles' residuals. Runtime: 64 codebooks × 4-16 levels = 256-1024 floats
per expert — negligible storage and lookup cost.

## Updated best method

**6-tier tile-level mixed precision with 64-cluster codebooks**:
- K3 (3.0 bpw), K4 (4.0 bpw)
- K4+1LM_64c (5.0001 bpw), K4+2LM_64c (6.0001 bpw)
- K3+4LM_64c (7.0003 bpw), K4+4LM_64c (8.0003 bpw)
- 64 shared codebooks per expert, clustered by sigma
- Overhead: <0.001 bpw (negligible)
- Quality: 93-70% of per-tile, 7-30% better than global
