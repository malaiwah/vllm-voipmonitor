# Round 5b: Leech Lattice (LLVQ) and Entropy-Constrained Quantization

## Leech Lattice VQ (LLVQ, ICML 2026, arXiv:2603.11021)

- 24-dimensional lattice with optimal sphere packing (Fields Medal 2022)
- Codebook-free: uses Golay code construction for indexing
- Supports fractional bitrates naturally (shell selection controls bitrate)
- Outperforms QTIP, QuIP#, PVQ at 2 bpw
- Shape-gain quantization: separate magnitude (scalar) and direction (Leech)

**Relevance**: LLVQ is strongest at 2 bpw where EXL3 trellis also operates.
At 3-5 bpw (our range), EXL3 trellis is already near-optimal for Gaussian sources.
LLVQ could potentially be used as an alternative to K3 (3 bpw) if it achieves
better quality at 3 bpw, but the implementation complexity (24-D Golay code
search) is much higher than trellis.

## Entropy-Constrained Quantization (arXiv:2505.18758, 2505.02380)

- Rate-constrained quantization with entropy coding
- Quadratic rate estimation for layer-wise loss
- EntroLLM: mixed quantization + entropy coding for edge deployment
- Rate-aware: accounts for actual entropy of quantized codes, not just bitwidth

**Relevance**: Our v5 measured entropy of Lloyd-Max codes (H=1.91 vs 2.0 fixed).
The gain is marginal (~4.5%) for Gaussian sources where all codes are
roughly equiprobable. More relevant for non-uniform distributions.

## Neural Weight Compression (arXiv:2510.11234)

- Survey of weight compression methods
- Compares VQ, lattice, trellis, and scalar approaches
- Finds VQ methods (Leech, E8) outperform scalar at low rates
- At higher rates (>3 bpw), the gap narrows significantly

**Relevance**: Confirms that at 3+ bpw, trellis quantization (EXL3) is
competitive with more complex VQ methods. Our tile-level approach on top
of EXL3 trellis is the right choice for the 3-5 bpw range.

## Conclusion

After 5 rounds of research (30+ papers) and 14 PoC versions, the **tile-level
3-tier K3/K4/K5 mixed precision** method remains the best approach for
continuously variable 3-5.5 bpw fungible quantization of GLM-5.2.

Key reasons no alternative beats it:
1. EXL3 trellis is near-optimal for Gaussian sources at 3-5 bpw
2. Tile-level mixing (Q-Palette's half-TCQ analog) provides fractional bits
3. Per-expert/per-layer allocation doesn't help (GLM-5.2 is homogeneous)
4. The 2-bit tier bitmap (0.008 bpw) is negligible overhead
5. It's calibration-free and runtime-efficient
