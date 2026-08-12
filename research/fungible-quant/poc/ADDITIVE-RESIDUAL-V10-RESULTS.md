# AlphaQ PL_Alpha_Hill Per-Expert Allocation — v10 Results

## Finding: Per-Expert Allocation Does NOT Help for GLM-5.2

PL_Alpha_Hill (spectral heavy-tailedness) variation across 20 experts:
- Range: [1.0624, 1.0665] (0.4% variation)
- CV: 0.0011 (0.11%)

AlphaQ allocation vs uniform at 3.5 bpw:
- Uniform (10×K4 + 10×K3): MSE = 1.723e-02
- AlphaQ (4×K5 + 16×K3): MSE = 2.092e-02 → **21% WORSE**

AlphaQ allocation vs uniform at 4.0 bpw:
- Uniform (20×K4): MSE = 7.286e-03
- AlphaQ (8×K5 + 12×K3): MSE = 1.465e-02 → **101% WORSE**

## Root Cause: GLM-5.2 Experts Are Spectrally Homogeneous

All per-expert metrics show negligible variation:
| Metric | CV across experts |
|--------|-------------------|
| K3 MSE | 0.03% |
| K4 MSE | 0.05% |
| Disambiguation (dist from mean) | 0.06% |
| PL_Alpha_Hill | 0.11% |
| Spectral energy (BitsMoE) | 2.35% |
| Weight variance | 3.80% |

The Hadamard regularization further equalizes quantization damage.
Per-expert allocation requires heterogeneity that doesn't exist in well-trained MoEs.

## Conclusion

The tile-level approach (within each expert) is the correct granularity for quality
differentiation. Per-expert allocation is not beneficial for GLM-5.2 because all
experts have nearly identical statistical properties after training.

The continuously variable 3-5 bpw achieved by tile-level K3/K4/K5 mixing (v9)
remains the best method.
