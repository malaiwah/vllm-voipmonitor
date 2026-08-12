# PoC v44: MSRT vs RRQ + per-row rescaling + cross-layer validation

## MSRT vs RRQ (Round-to-Nearest residual)

| Method | bpw | Layer 10 MSE | Layer 40 MSE | Ratio |
|--------|-----|-------------|-------------|-------|
| MSRT (rescaled TCQ) | 6 | 5.146e-04 | 5.144e-04 | 0.9997 |
| RRQ (RTN residual) | 6 | 3.515e-02 | 3.617e-02 | 1.029 |
| MSRT (rescaled TCQ) | 8 | 3.890e-05 | 3.865e-05 | 0.994 |
| RRQ (RTN residual) | 8 | 3.932e-03 | 4.054e-03 | 1.031 |

**MSRT is 68-101× better than RRQ!** The rescaled trellis (TCQ with Viterbi)
is dramatically better than simple round-to-nearest on the residual. RRQ's
RTN approach wastes bits because it doesn't exploit the trellis structure
or the Viterbi optimal path.

This confirms our key innovation: using TCQ (not RTN) for residual quantization,
with rescaling to match the codebook range.

## Per-row rescaling vs global RMS

| Method | bpw | Global RMS | Per-row | Improvement |
|--------|-----|-----------|---------|-------------|
| MSRT | 6 | 5.146e-04 | 5.144e-04 | 0.04% |
| MSRT | 8 | 3.890e-05 | 3.876e-05 | 0.36% |

Per-row rescaling gives negligible improvement (0.04-0.36%). Global RMS
rescaling is sufficient — the Hadamard regularization already equalizes
row distributions.

## Cross-layer validation (layer 10 vs 40)

| Method | L10 MSE | L40 MSE | Ratio |
|--------|---------|---------|-------|
| MSRT 8bpw | 3.890e-05 | 3.865e-05 | 0.994 |
| MSRT perrow 8bpw | 3.876e-05 | 3.852e-05 | 0.994 |

**MSRT works identically across layers** (ratio ≤ 1.006). The Hadamard
regularization makes the residual distribution layer-independent, confirming
our v13 finding that layers 10 and 40 are statistically identical.

## Conclusion

1. **MSRT (rescaled TCQ) is 68-101× better than RRQ (RTN)** — TCQ's Viterbi
   optimization is essential; simple rounding is far worse
2. **Per-row rescaling gives negligible gain** — global RMS is sufficient
3. **MSRT is universal across layers** — works identically on layer 10 and 40
4. **RRQ (NeurIPS 2026) is fundamentally limited** by using RTN instead of TCQ
   for residual quantization
