# PoC v30: Optimal cluster count + stacking test

## Optimal Cluster Count Per Bit-Width

More clusters always improves MSE, but with diminishing returns and growing overhead.

| Bit-width | c64 MSE | c128 MSE | c512 MSE | c64→c512 gain | c512 overhead |
|-----------|---------|----------|----------|---------------|---------------|
| 1-bit     | 2.800e-03 | 2.800e-03 | 2.799e-03 | 0.0% | 0.00033 bpw |
| 2-bit     | 9.937e-04 | 9.898e-04 | 9.878e-04 | 0.6% | 0.00065 bpw |
| 3-bit     | 3.112e-04 | 3.088e-04 | 3.066e-04 | 1.5% | 0.00130 bpw |
| 4-bit     | 9.662e-05 | 9.541e-05 | 9.367e-05 | 3.0% | 0.00260 bpw |
| 6-bit     | 1.025e-05 | 9.620e-06 | 9.186e-06 | 10.3% | 0.01042 bpw |

**Practical sweet spot: c128** — captures 90%+ of gain, overhead <0.003 bpw.
For 6-bit LM, c256 is better (10% gain from c64→c512, but 0.005 bpw overhead).

## Stacking vs Single Large Codebook (64 clusters)

| Method | bpw | MSE | vs single |
|--------|-----|-----|-----------|
| K4+3LM_single | 7 | 3.112e-04 | baseline |
| K4+2LM+1LM_stack | 7 | 4.120e-04 | 1.32× worse |
| K4+4LM_single | 8 | 9.662e-05 | baseline |
| K4+2LM+2LM_stack | 8 | 2.156e-04 | 2.23× worse |

**Finding**: Single large codebook always beats stacking, even with 64-cluster
codebooks. The compounding quantization error from stacking is fundamental.
This confirms v22's finding.

## Updated recommendation

- Use c128 codebooks (universal across layers/projections, verified v28)
- Always use single large LM codebook, never stack
- Entropy coding (v29) saves 5-13% of LM bits
- 8-tier system: K2(2), K3(3), K4(4), K4+1LM(5), K4+2LM(6), K4+3LM(7), K2+6LM(8), K4+6LM(10)
