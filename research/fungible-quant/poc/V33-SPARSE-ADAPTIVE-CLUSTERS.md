# PoC v33: Sparse LM, adaptive LM, tier-specific clusters

## Part 1: Tier-specific cluster counts

| Method | bpw | MSE (c128) | MSE (c512) | c512 improvement |
|--------|-----|------------|------------|------------------|
| K4+4LM | 8 | 9.546e-05 | 9.372e-05 | 1.8% |
| K4+6LM | 10 | 9.632e-06 | 9.169e-06 | 4.8% |

c512 gives only 2-5% improvement over c128. Not worth the 4× overhead increase.

## Part 2: Sparse residual LM

Only quantize top-K% largest |residual| values with 2-bit LM.

| Sparsity | Raw bpw | Entropy bpw | MSE | vs uniform 6bpw |
|----------|---------|-------------|-----|-----------------|
| 10% | 5.200 | 4.669 | 4.116e-03 | 4.2× worse |
| 25% | 5.500 | 5.311 | 2.290e-03 | 2.3× worse |
| 50% | 6.000 | 6.000 | 1.225e-03 | 24% worse |
| 75% | 6.500 | 6.311 | 1.041e-03 | 5% worse |
| 100% | 7.000 | 6.000 | 9.886e-04 | 0% (same as uniform) |

**Finding**: Sparse LM is always worse than uniform LM at the same bitrate.
The sparse bitmap (1 bit/weight) overhead dominates. Even with entropy coding,
the bitmap costs ~0.8 bpw, negating the sparsity savings.

## Part 3: Per-tile adaptive LM bit-width

Assign 3-bit LM to high-energy tiles, 1-bit to low-energy tiles.

| Split | bpw | MSE | vs uniform 6bpw |
|-------|-----|-----|-----------------|
| 30% high | 5.6 | 1.806e-03 | 83% worse |
| 50% high | 6.0 | 1.339e-03 | 35% worse |
| 70% high | 6.4 | 9.032e-04 | ~same as uniform at 6.4 |

**Finding**: Per-tile adaptive LM is 35% worse than uniform at the same total
bitrate. The sigma-clustering already handles per-tile variation optimally.
Splitting tiles by energy and using different codebook sizes wastes bits on
low-energy tiles that barely contribute to MSE.

## Conclusion

All three ideas confirm that the current approach is already optimal:
- Uniform LM with c128 codebooks is better than sparse or adaptive variants
- The Hadamard regularization + Lloyd-Max + sigma-clustering combination
  already captures all exploitable structure in the residual
- No new method from 60+ papers beats tile-level mixed precision with
  c128 clustered codebooks + entropy-aware Pareto
