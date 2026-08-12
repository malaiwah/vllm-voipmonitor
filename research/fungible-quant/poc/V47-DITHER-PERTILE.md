# PoC v47: Dithering and per-tile K1 rescaling — both worse than baseline

## Results

| Method | bpw | MSE | vs baseline |
|--------|-----|-----|-------------|
| MSRT_8bpw_baseline | 8 | 3.890e-05 | — |
| MSRT_8bpw_dithered | 8 | 5.271e-05 | 1.36× worse |
| MSRT_8bpw_dither_K1only | 8 | 5.030e-05 | 1.29× worse |
| MSRT_6bpw_pertile_K1 | 6 | 5.123e-04 | 0.4% better than baseline (5.144e-04) |

## Dithering: 29-36% worse

Subtractive dithering makes MSRT 29-36% worse. The dither noise adds to the
quantization error rather than reducing it. This is because:
1. TCQ (Viterbi) already optimizes the quantization path — dithering disrupts
   this optimization by adding noise to the input
2. The trellis codebook is designed for clean Gaussian input, not dithered input
3. Unlike scalar quantization (where dithering helps by breaking structured
   error patterns), TCQ's Viterbi already handles structure optimally

## Per-tile K1 rescaling: 0.4% better (negligible)

Per-tile rescaling for K1 stages gives only 0.4% improvement over global RMS.
This is consistent with v39 (per-tile rescaling 3-7% better) and v44 (per-row
0.04-0.36% better) — the gains are too small to justify the complexity.

## Conclusion

Neither dithering nor per-tile rescaling improves MSRT meaningfully:
- Dithering is actively harmful (disrupts Viterbi optimization)
- Per-tile rescaling gives negligible gain (0.4%)
- The Hadamard regularization already makes the residual distribution uniform
  enough that global RMS rescaling is sufficient

MSRT with global RMS rescaling and no dithering remains optimal.
