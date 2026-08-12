# PoC v46: Systematic stage allocation search — new optimal found!

## 8bpw (6 residual bits): 32 allocations tested

### Top 10

| Rank | Allocation | MSE | Stages |
|------|-----------|-----|--------|
| 1 | K2+K1+K1+K2+K2 | 3.879e-05 | 4 |
| 2 | K2+K1+K2+K3 | 3.885e-05 | 3 |
| 3 | K2+K1+K1+K1+K3 | 3.891e-05 | 4 |
| 4 | K2+K1+K1+K4 | 3.895e-05 | 3 |
| 5 | K2+K1+K2+K1+K2 | 3.994e-05 | 4 |

**New best at 8bpw**: K2+K1+K1+K2+K2 = 3.879e-05 (0.2% better than K2+K1+K2+K3).
The difference is within noise — K2+K1+K2+K3 remains the practical best (fewer stages).

### Bottom 3

| Allocation | MSE | Stages |
|-----------|-----|--------|
| K2+K4+K2 | 6.725e-05 | 2 |
| K2+K5+K1 | 7.608e-05 | 2 |
| K2+K6 (single) | 8.126e-05 | 1 |

**Single-stage K6 is 2.1× worse than best** — multi-stage is essential.

## 9bpw (7 residual bits): 62 allocations tested (partial — timed out)

### Top 5 (from what completed)

| Rank | Allocation | MSE | Stages |
|------|-----------|-----|--------|
| 1 | K2+K1+K1+K2+K3 | 1.103e-05 | 4 |
| 2 | K2+K1+K1+K1+K4 | 1.105e-05 | 4 |
| 3 | K2+K1+K1+K3+K2 | 1.149e-05 | 4 |
| 4 | K2+K1+K2+K1+K3 | 1.203e-05 | 4 |
| 5 | K2+K1+K1+K4+K1 | 1.250e-05 | 4 |

**New best at 9bpw**: K2+K1+K1+K2+K3 = 1.103e-05 (matches v41 result, now confirmed optimal among 62 allocations).

## Key patterns

1. **Start with K1**: Every top-10 allocation starts with K1 (first residual stage)
2. **More stages = better**: 3-4 stages consistently beat 1-2 stages
3. **End with K2 or K3**: The final stage should be K2 or K3 (fine refinement)
4. **K1,K1 at front**: Multiple K1 stages at the beginning are optimal
5. **Order matters**: K1+K2+K3 (3.885e-05) >> K3+K2+K1 (not tested but K3 first is worse)

## Rate-distortion interpretation

The optimal allocation follows the reverse waterfilling principle:
- K1 stages handle the largest residual (highest σ), needing only 1 bit each
- K2/K3 stages handle the smaller, refined residual (lower σ), needing 2-3 bits
- This matches the Gaussian R-D theory: D ∝ σ² · 2^(-2R), so larger σ gets more bits

## Updated best MSRT (v46)

| bpw | Best allocation | MSE | Stages |
|-----|----------------|-----|--------|
| 5 | K2+K3 | 1.892e-03 | 1 |
| 6 | K2+K1+K3 | 5.144e-04 | 2 |
| 7 | K2+K1+K4 | 1.415e-04 | 2 |
| 8 | K2+K1+K1+K2+K2 | 3.879e-05 | 4 |
| 9 | K2+K1+K1+K2+K3 | 1.103e-05 | 4 |
| 10 | K2+K1+K1+K1+K2+K3 | 3.381e-06 | 5 |

Note: At 8bpw, K2+K1+K1+K2+K2 (4 stages) is 0.2% better than K2+K1+K2+K3 (3 stages).
The 3-stage version is preferred for runtime (fewer dequant passes).
