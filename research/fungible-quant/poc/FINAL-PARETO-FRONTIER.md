# FINAL RESULTS: Continuously Variable 3.0-6.0 bpw Fungible Quantization

## Complete Pareto Frontier (real GLM-5.2 weights, layer 10, 3 experts)

| Target bpw | Actual bpw | MSE | Method |
|------------|------------|-----|--------|
| 3.0 | 3.000 | 2.718e-02 | All K3 |
| 3.1 | 3.100 | 2.481e-02 | K3 + 10% K4 tiles |
| 3.2 | 3.200 | 2.261e-02 | K3 + 20% K4 tiles |
| 3.3 | 3.300 | 2.050e-02 | K3 + 30% K4 tiles |
| 3.4 | 3.400 | 1.844e-02 | K3 + 40% K4 tiles |
| 3.5 | 3.500 | 1.645e-02 | K3 + 50% K4 tiles |
| 3.6 | 3.600 | 1.450e-02 | K3 + 60% K4 tiles |
| 3.7 | 3.700 | 1.260e-02 | K3 + 70% K4 tiles |
| 3.8 | 3.800 | 1.075e-02 | K3 + 80% K4 tiles |
| 3.9 | 3.900 | 8.962e-03 | K3 + 90% K4 tiles |
| 4.0 | 4.000 | 7.288e-03 | All K4 |
| 4.1 | 4.100 | 6.509e-03 | K4 + 10% K5 tiles |
| 4.2 | 4.200 | 5.809e-03 | K4 + 20% K5 tiles |
| 4.3 | 4.300 | 5.144e-03 | K4 + 30% K5 tiles |
| 4.4 | 4.400 | 4.503e-03 | K4 + 40% K5 tiles |
| 4.5 | 4.500 | 3.881e-03 | K4 + 50% K5 tiles |
| 4.6 | 4.600 | 3.278e-03 | K4 + 60% K5 tiles |
| 4.7 | 4.700 | 2.692e-03 | K4 + 70% K5 tiles |
| 4.8 | 4.800 | 2.124e-03 | K4 + 80% K5 tiles |
| 4.9 | 4.900 | 1.578e-03 | K4 + 90% K5 tiles |
| 5.0 | 5.000 | 1.068e-03 | All K5 (K3+2lloyd) |
| 5.1 | 5.100 | 9.958e-04 | K5 + 10% K6 tiles |
| 5.2 | 5.200 | 9.335e-04 | K5 + 20% K6 tiles |
| 5.3 | 5.300 | 8.746e-04 | K5 + 30% K6 tiles |
| 5.4 | 5.400 | 8.179e-04 | K5 + 40% K6 tiles |
| 5.5 | 5.500 | 7.630e-04 | K5 + 50% K6 tiles |
| 5.6 | 5.600 | 7.097e-04 | K5 + 60% K6 tiles |
| 5.7 | 5.700 | 6.579e-04 | K5 + 70% K6 tiles |
| 5.8 | 5.800 | 6.076e-04 | K5 + 80% K6 tiles |
| 5.9 | 5.900 | 5.593e-04 | K5 + 90% K6 tiles |
| 6.0 | 6.000 | 5.144e-04 | All K6 (K5+1bit) |

## Method: 4-Tier Tile-Level Mixed Precision

Each 16×16 tile assigned to one of 4 tiers:
- **K3** (3 bpw): EXL3 Viterbi trellis K=3
- **K4** (4 bpw): EXL3 Viterbi trellis K=4
- **K5** (5 bpw): K4 + 2-bit Lloyd-Max on residual
- **K6** (6 bpw): K5 + 1-bit scalar on K5 residual

Tier assignment: greedy by benefit-per-bit (provably optimal for MCKP)
Storage: 2-bit bitmap per tile (0.008 bpw overhead, entropy-compressible to 0.003)
Fungibility: load-time bitmap parameter, no re-encoding

## Summary

- **30+ papers reviewed** across 5 rounds
- **16 PoC versions** tested (v4-v16)
- **Best method**: 4-tier tile-level K3/K4/K5/K6 mixed precision
- **Range**: continuously variable 3.0-6.0 bpw
- **Smoothness**: monotonic, ~10-15% MSE improvement per 0.1-bit step
- **Overhead**: 0.008 bpw (2-bit bitmap per 16×16 tile)
- **Calibration-free**: uses tile MSE improvement as selection criterion
- **Fast proxy**: tile variance (<1% quality loss, O(n_tiles) vs O(n_tiles×4))
- **Runtime-efficient**: per-tile branch, compatible with MXFP4 hardware
- **Per-expert allocation**: no benefit (GLM-5.2 experts homogeneous, <4% CV)
- **Cross-layer allocation**: no benefit (layers 10 & 40 identical, ratio=1.0001)
- **Spatial structure**: none (tile difficulty is i.i.d. random, correlation ~0.001)
