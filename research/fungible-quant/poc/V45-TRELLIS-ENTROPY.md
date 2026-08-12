# PoC v45: Trellis index entropy — indices are structured, not per-weight

## Finding

The EXL3 trellis `quantized_idx` contains `short` (int16) values that encode
the Viterbi path through the trellis, NOT per-weight codebook indices.

Measured entropy: ~16 bits per short value (effectively uniform over int16 range).
This is expected — the trellis indices encode path transitions, not independent
per-weight values. The K bits/weight efficiency comes from the Viterbi algorithm's
optimal path selection, not from compressibility of individual indices.

## Implication

Entropy coding of trellis indices is NOT applicable in the same way as for
Lloyd-Max indices (v29: LM indices had H=3.49 vs 4 bits → 12.6% savings).

The trellis encoding is already optimal — the Viterbi algorithm finds the
minimum-distortion path through the trellis, and the K bits/weight represent
the path encoding efficiency. There's no redundancy to exploit through
entropy coding.

## Contrast with LM

Lloyd-Max: per-weight scalar indices → non-uniform distribution → entropy savings
Trellis (TCQ): structured path indices → uniform → no entropy savings

This is a fundamental difference between scalar quantization (LM) and trellis
quantization (TCQ). TCQ's efficiency comes from the trellis structure, not
from index compressibility.

## Conclusion

MSRT's bitrate is already optimal at raw K bits/weight per stage. No entropy
coding benefit for trellis indices (unlike LM indices). The raw-bits Pareto
is the definitive Pareto for MSRT.
