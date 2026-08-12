# Round 18: ECTCQ, embedded entropy-constrained TCQ — final literature

## Entropy-Constrained TCQ (ECTCQ)

From IEEE literature (early 1990s):
- Generalizes TCQ to allow noiseless coding of trellis branch symbols
- For memoryless Gaussian source: within 0.21 dB of distortion-rate bound
- Uses variable-rate encoding (entropy coding of branch labels)

**Tested in v45**: The EXL3 trellis indices are int16 Viterbi path values,
not per-weight branch labels. The trellis structure means the "indices" are
structured path information, not independent compressible symbols. Unlike
scalar quantization (LM), TCQ's efficiency comes from the Viterbi optimal
path, not from index compressibility.

**Conclusion**: ECTCQ theory applies to the branch label level, but the
EXL3 trellis implementation encodes the full path as int16 values, making
direct entropy coding impractical. The raw K bits/weight already captures
the Viterbi efficiency.

## Embedded E-ECTCQ

- Variable-rate embedded quantization (progressive bitstream)
- Combines successive refinement with entropy-constrained TCQ
- Theoretically achieves near-optimal R-D for progressive transmission

**Relevance**: Our MSRT achieves progressive refinement through multi-stage
rescaled trellis, which is the structural (non-entropy-coded) version of
E-ECTCQ. The entropy coding aspect doesn't apply to our trellis indices (v45).

## Key Insight from Round 18

TCQ and scalar quantization have fundamentally different entropy properties:
- **Scalar (LM)**: Per-weight indices are non-uniform → entropy savings
- **Trellis (TCQ)**: Path indices are structured → no entropy savings

This means MSRT's raw-bits Pareto is already optimal — no entropy coding
benefit. The efficiency of MSRT comes entirely from the multi-stage
rescaled trellis structure, not from compressibility.

## Total: 80+ papers reviewed across 18 rounds

MSRT (Multi-Stage Rescaled Trellis) remains the definitive best method.
No published method or theoretical improvement beats it for fungible
quantization of GLM-5.2.
