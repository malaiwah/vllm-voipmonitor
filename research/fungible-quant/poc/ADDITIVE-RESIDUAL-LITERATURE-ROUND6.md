# Round 6 Literature: GLVQ, MoBiQuant details, expert merging, tensor networks

## GLVQ (NeurIPS 2025) — Learnable Lattice VQ
- Per-group learnable generation matrix for lattice codebook
- Babai rounding for differentiable nearest-lattice-point search
- Decoding: simple matrix-vector multiply
- Better than uniform quantization at low bits

**Relevance**: Could improve per-tile quality by learning tile-specific codebooks.
But requires training (not calibration-free). EXL3 trellis is already near-optimal
for Gaussian. GLVQ's advantage is mainly at 2 bpw, not 3-6 bpw.

## MoBiQuant (detailed) — Token-Adaptive Any-Precision
- Recursive residual quantization (like RRQ) + token-aware router
- "Outlier migration": PTQ-sensitive tokens change across precisions
- Token router selects optimal precision per token at runtime
- 1.34× throughput over SOTA any-precision

**Relevance**: Most advanced any-precision method. Our tile-level approach is
simpler (static bitmap vs dynamic router) but MoBiQuant's token-aware routing
could be a future extension: use different tier bitmaps for different token types.

## Expert Merging (PuzzleMoE, MergeMoE, REAM, LightMoE)
- Combine similar experts to reduce total expert count
- Different from quantization (reduces N, not bpw)
- Could be combined with our approach: merge experts → fewer experts →
  more budget per expert → higher tier

**Relevance**: Orthogonal compression axis. Could stack with our tile-level
quantization for additional memory savings.

## Tensor Networks (EinSort, CompactifAI, Minima)
- Decompose weight matrices into tensor network (TT, TR, CP)
- Index ordering (sorting) exposes hidden low-rank structure
- CompactifAI: Matrix Product Operator (MPO) decomposition

**Relevance**: Low-rank methods don't help after Hadamard (our v4/v8 confirmed
residuals are full-rank). But tensor networks operate on raw weights, not
residuals — could be an alternative to Hadamard+trellis.

## Polar Coordinate Quantization (HIT Shenzhen)
- Quantize in polar coordinates: separate direction and magnitude codebooks
- Direction: spherical code (like PVQ/LLVQ)
- Magnitude: scalar quantizer

**Relevance**: Similar to LLVQ's shape-gain decomposition. After Hadamard,
weights are approximately Gaussian (direction × magnitude). Polar quantization
is theoretically optimal for Gaussian but implementation complexity is high.

## Summary (Round 6)

Total papers: 37. No new method beats tile-level 4-tier K3/K4/K5/K6.
The main future directions are:
1. Token-aware precision (MoBiQuant) — dynamic per-token tier selection
2. Expert merging + quantization — orthogonal compression axis
3. Learnable per-tile codebooks (GLVQ) — but requires training
4. MXFP4 hardware path — native on RTX 5090
