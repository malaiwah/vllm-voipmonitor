# Round 13: HARP, MSQ, QTIP, Proteus, NanoQuant, ReSpinQuant

## New Papers Reviewed

### HARP (arXiv:2605.29843, 2026)
- Learnable structured two-sided orthogonal rotation replacing fixed Hadamard
- Butterfly-like block-orthogonal stages, Mixed-Radix for non-power-of-2
- Requires calibration data (we're calibration-free)
- Improves 2-4 bit quantization over fixed RHT
- **Not applicable**: Our approach is calibration-free; HARP needs calibration

### MSQ (ICCV 2025, arXiv:2507.22349)
- Memory-efficient bit sparsification: removes LSBs without bit decomposition
- LSB regularization induces sparsity in least significant bits
- QAT approach (training-aware), not PTQ
- **Not applicable**: QAT, not PTQ. Our v33 showed sparse approaches are worse

### QTIP (NeurIPS 2024, arXiv:2406.11235)
- Trellis coded quantization (TCQ) with bitshift trellis
- TCQ achieves 40% lower distortion than scalar on Gaussian sources
- 256-dimensional quantization (vs 8D VQ in QuIP#)
- Compute-based codes for fast GPU decoding
- **Tested in v34**: TCQ on RESIDUAL is 2-34× WORSE than Lloyd-Max
  - Codebook mismatch: trellis optimized for weight σ, not residual σ
  - Confirms: trellis for base + LM for residual is optimal combination

### Proteus (ICML 2026)
- Lookup-free trellis-coded quantization
- Eliminates codebook lookups for faster decoding
- **Relevant for runtime**, not for quality improvement

### NanoQuant (ICML 2026)
- Sub-1-bit quantization of LLMs
- Extreme compression below 1 bit/weight
- **Below our range** (we operate at 2-10 bpw)

### ReSpinQuant (ICML 2026)
- Subspace residual rotation approximation
- Learnable layer-wise rotation
- Requires calibration data
- **Not applicable**: Calibration-free approach

## Key Insight from Round 13

QTIP's TCQ is optimal for the **weight distribution** but NOT for the
**residual distribution**. This is because:
1. TCQ codebook is designed for σ ≈ 1 (weight scale after Hadamard)
2. Residual has σ ≈ 0.1-0.3 (much smaller, different shape)
3. Lloyd-Max adapts codebook to residual σ → optimal for residual

Our two-stage approach (TCQ for base + LM for residual) exploits each
quantizer's strength optimally. No single quantizer can handle both
distributions as well.

## Total: 65+ papers reviewed across 13 rounds
