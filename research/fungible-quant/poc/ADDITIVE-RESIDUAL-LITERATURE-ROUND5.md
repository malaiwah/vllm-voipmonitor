# Round 5 Literature: Microscaling, CodeQuant, MoBiQuant, GAMMA

## New Papers (Round 5)

### Microscaling (MX) Formats — MXFP4, NVFP4
- Block-level shared scaling factors (group of 32 elements shares one E8M0 exponent)
- Native hardware support on NVIDIA Blackwell (RTX 5090!)
- NVFP4: FP8-e4m3 scale per 16 elements, MXFP4: E8M0 scale per 32 elements
- Fused dequantization in tensor memory → no overhead

**Relevance to our tile-level approach**:
- Our 16×16 tiles (256 elements) could use MX-style shared scales
- NVFP4 uses 16-element groups — matches our tile dimension
- The 2-bit tier bitmap could select between MXFP4 (≈4 bits), MXFP6 (≈6 bits) tiers
- Hardware-native: RTX 5090 supports MXFP4 natively

### MoBiQuant (arXiv:2602.20191, Feb 2026)
- Recursive residual quantization (like RRQ) + token-aware precision router
- "Outlier migration" phenomenon: PTQ-sensitive tokens change across precisions
- Token-aware router selects optimal inference precision per token
- 1.34× throughput over SOTA any-precision methods

**Relevance**: Token-aware precision is the next step beyond tile-level. Instead of
a static tier bitmap, the precision could adapt per token at runtime. This is more
complex but could provide better quality-latency trade-offs.

### GAMMA (arXiv:2605.18475, May 2026)
- Global bit allocation for mixed-precision under arbitrary budgets
- Solves MCKP globally across all layers (not per-layer)
- Data-driven sensitivity estimation

**Relevance**: Our tile-level approach is per-tile, not per-layer. GAMMA's global
optimization could be applied to our tile tiers — but we showed cross-layer
allocation provides no benefit for GLM-5.2 (v13).

### CodeQuant (ICLR 2026)
- Unified clustering + quantization for MoE
- Learnable rotation (Cayley transform) for activation outlier smoothing
- K-means clustering of weights with centroid fine-tuning
- LUT-based kernel for GPU/CPU
- KL divergence loss to preserve router assignments

**Relevance**: Clustering is an alternative to trellis quantization. Our v4 showed
codebook methods (AQLM, multi-codebook) fail for i.i.d. Gaussian residuals.
CodeQuant uses clustering on raw weights (before Hadamard), which is different.
Could be combined with our tile-level approach: cluster per tile instead of
per matrix.

### WUSH (arXiv:2512.00956)
- Data-aware transforms per block (not just global Hadamard)
- Adaptive transforms that match local weight structure
- 5.8× per-layer speedup over optimized blockwise Hadamard kernels

**Relevance**: Per-tile different rotations could create per-tile diversity that
enables differentiated allocation. But this contradicts our finding that Hadamard
equalizes everything — perhaps WUSH's data-aware transforms would create MORE
variation, not less.

## Summary of All Rounds (1-5)

Total papers reviewed: **28**

| Round | Papers | Key Insight |
|-------|--------|-------------|
| 1 | RRQ, Drop-by-Drop, ResQ, R2Q, AQLM, AnyBCQ, MoPEQ, MatGPTQ | Residual refinement theory |
| 2 | HyperQuant, ICQuant, Q-Palette, GLVQ, Radio, RateQuant | Rate-distortion + index coding |
| 3 | BitsMoE, MxMoE, TileQ, AlphaQ, MoPEQ, EAQuant | Per-expert allocation (doesn't help) |
| 4 | RRQ-Intel, MoBiQuant, FlexQuant, BCJR-QAT, RUQuant, PolarQuant | Progressive bitstream + token-aware |
| 5 | MXFP4, CodeQuant, GAMMA, WUSH, MoBiQuant | Hardware formats + clustering |

## Best Method (Confirmed across all rounds)

**Tile-level 3-tier K3/K4/K5 mixed precision** remains the best method:
- Continuously variable 3.0-5.5 bpw
- Load-time bitmap parameter (no re-encoding)
- 0.008 bpw bitmap overhead (entropy-compressible to 0.003)
- Runtime-efficient (per-tile branch, compatible with MXFP4 hardware)
- Calibration-free
- <1% quality loss with variance proxy for tile selection

## Future Directions (from round 5)

1. **MXFP4 hardware path**: Implement tile tiers using NVFP4 (16-element groups)
   native on RTX 5090 — tier 0 = K3 (INT4), tier 1 = NVFP4, tier 2 = NVFP4 + 2-bit residual
2. **Token-aware precision** (MoBiQuant): Route tokens to different tier bitmaps
   at runtime based on token sensitivity
3. **Data-aware per-tile rotation** (WUSH): Use different rotations per tile to
   create tile-level diversity for differentiated allocation
4. **Clustering per tile** (CodeQuant): K-means within each tile instead of
   global trellis — may capture local structure better
