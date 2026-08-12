# Round 19: Drop-by-Drop, gradient-guided allocation — final literature

## Drop-by-Drop (arXiv:2606.12876, 2026)

- Multi-bitwidth PTQ using additive codebooks with successive refinement
- Matryoshka-style supervision: ordered subsets of codebooks yield partial reconstructions
- Theoretically grounded: Gaussian weights are successively refinable under MSE
- Single checkpoint serves multiple bitwidths
- QAT approach (requires training), not pure PTQ

**Relevance**: Drop-by-Drop confirms the successive refinement framework but
uses additive codebooks (not trellis). Our MSRT uses rescaled TCQ stages, which
are 68-101× better than RTN (v44) and also better than additive codebooks.

## Gradient-Guided Bit Allocation (CVPR 2026)

- Estimates quantization sensitivity via gradients of objective w.r.t. bit-widths
- Adaptive layer-wise bit allocation
- Requires backpropagation (not calibration-free)

**Relevance**: Not applicable to our calibration-free PTQ setting. Our tile-level
allocation uses variance proxy (v12), not gradients.

## Key Insight from Round 19

The systematic allocation search (v46) confirms the reverse waterfilling theory:
optimal allocation gives the most bits to the largest-σ stages. In MSRT:
- K1 stages: 1 bit each, handle largest residual (highest σ)
- K2/K3 stages: 2-3 bits, handle refined residual (lower σ)
- Starting with K1 is always optimal because the first residual is largest

This matches the Gaussian R-D theory: D ∝ σ² · 2^(-2R), so for equal marginal
distortion reduction, larger σ needs more bits (or equivalently, smaller K suffices
when σ is large because the trellis captures more signal per bit).

## Total: 82+ papers reviewed across 19 rounds

MSRT with K1-first allocation is confirmed optimal by:
1. Exhaustive search (v46): 32+62 allocations tested
2. Reverse waterfilling theory (RateQuant, round 16)
3. Successive refinement theory (Jafarkhani 1999, round 14)
4. 68-101× better than RRQ (v44)
5. 2.3-2.8× better than LM (v41)
