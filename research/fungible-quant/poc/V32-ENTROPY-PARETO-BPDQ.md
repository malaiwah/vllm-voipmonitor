# PoC v32: Entropy-aware Pareto + BPDQ bit-plane test

## BPDQ Bit-Plane vs Lloyd-Max

| Method | 2-bit MSE | vs LM | 4-bit MSE | vs LM |
|--------|-----------|-------|-----------|-------|
| Lloyd-Max | 9.886e-04 | baseline | 9.546e-05 | baseline |
| Bit-plane (BPDQ) | 1.213e-03 | 1.23× worse | 9.918e-05 | 1.04× worse |

**Finding**: BPDQ bit-plane decomposition is 4-23% worse than Lloyd-Max.
The sign-magnitude split introduces an artificial constraint. Lloyd-Max
is already optimal for Gaussian residuals (Hadamard-regularized).

## Entropy-Aware Pareto — MAJOR IMPROVEMENT

Using entropy_bpw instead of raw_bpw as the tier budget allows upgrading
more tiles to higher tiers at the same target bitrate.

| Target bpw | Raw MSE | Entropy MSE | Improvement |
|------------|---------|-------------|-------------|
| 5.0 | 2.795e-03 | 2.793e-03 | 0.1% |
| 5.5 | 1.812e-03 | 1.715e-03 | 5.3% |
| 6.0 | 9.914e-04 | 8.546e-04 | 13.8% |
| 6.5 | 5.851e-04 | 4.425e-04 | 24.4% |
| 7.0 | 3.129e-04 | 1.994e-04 | 36.3% |
| 7.5 | 1.793e-04 | 9.154e-05 | 48.9% |
| 8.0 | 9.539e-05 | 7.015e-05 | 26.5% |
| 9.0 | 5.331e-05 | 3.230e-05 | 39.4% |
| 9.5 | 3.491e-05 | 1.905e-05 | 45.4% |

**At 7.5 bpw, entropy-aware Pareto gives 49% lower MSE!** This is because
K4+3LM costs 6.700 entropy-bpw (not 7.0), leaving 0.300 entropy-bpw
for upgrading tiles to K2+6LM (7.441 entropy-bpw).

## Why entropy-aware Pareto works

Without entropy coding, tier upgrades cost their full raw bit width.
With entropy coding, the effective cost is lower:
- K4+2LM: 2 raw bits → 1.899 entropy bits (5.1% savings)
- K4+3LM: 3 raw bits → 2.700 entropy bits (10.0% savings)
- K4+4LM: 4 raw bits → 3.495 entropy bits (12.6% savings)

The savings compound: at 7.0 entropy-bpw, you can afford K4+3LM (6.700)
plus 0.300 entropy-bpw of upgrades, vs only K4+3LM (7.000 raw) with no
upgrades in the raw Pareto.

## Updated best method

The entropy-aware Pareto should be the default mode for fungible quantization.
At decode time, entropy coding is optional (raw indices work correctly,
just use more bandwidth). For bandwidth-constrained scenarios, entropy
coding gives 14-49% MSE improvement at the same effective bitrate.
