# PoC v29: Entropy coding of LM indices + product quantization

## Entropy Coding of LM Indices

Lloyd-Max indices have non-uniform distributions. For Gaussian residuals,
inner levels are more probable than outer levels.

| LM bits | Raw bits | Entropy | Savings | Effective bpw (K4+LM) |
|---------|----------|---------|---------|----------------------|
| 1       | 1.000    | 1.000   | 0.0%    | 5.000                |
| 2       | 2.000    | 1.898   | 5.1%    | 5.898                |
| 3       | 3.000    | 2.700   | 10.0%   | 6.700                |
| 4       | 4.000    | 3.494   | 12.6%   | 7.494                |

Savings grow with bit width because more levels → more non-uniform distribution.
At 4-bit LM, entropy coding saves 0.506 bpw — significant.

## Product Quantization

Split 256-element tile into sub-vectors, quantize each with separate codebook.

| Method         | vs Scalar LM | Conclusion |
|----------------|-------------|------------|
| PQ 2SV × 2bit  | 1.077x worse | PQ doesn't help |
| PQ 4SV × 2bit  | 1.077x worse | Same |
| PQ 8SV × 2bit  | 1.077x worse | Same |
| PQ 16SV × 2bit | 1.076x worse | Same |
| PQ 2SV × 4bit  | 1.417x worse | PQ much worse |
| PQ 16SV × 4bit | 1.411x worse | Slight improvement with more SVs |

**Finding**: Product quantization is 7-42% WORSE than scalar LM. After Hadamard
regularization, the residual is fully decorrelated — splitting into sub-vectors
doesn't capture any additional structure. Scalar LM on the full tile is optimal.

## Updated Pareto with entropy coding

With entropy-coded LM indices, the effective bpw shifts left:

| Method | Raw bpw | Entropy bpw | MSE |
|--------|---------|-------------|-----|
| K4+2LM | 6.0     | 5.898       | 9.93e-04 |
| K4+3LM | 7.0     | 6.700       | 3.11e-04 |
| K4+4LM | 8.0     | 7.494       | 9.65e-05 |

This means at 6.0 bpw target, we can use K4+2LM with entropy coding
and have 0.102 bpw to spare for tier upgrades.
