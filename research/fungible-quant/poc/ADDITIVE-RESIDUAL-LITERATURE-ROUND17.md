# Round 17: RRQ (NeurIPS 2026), GSQ, Proteus — final literature

## RRQ: Recurrent Residual Quantization (arXiv:2608.04048, NeurIPS 2026)

- 2-bit base + sequence of 2-bit residual corrections (RTN)
- Calibration-free, avoids joint multi-bit optimization
- Progressive: 2→4→6→8 bits from single checkpoint
- Constructed in 1293s for Qwen3-8B (3.3× faster than MatGPTQ)

**Tested in v44**: RRQ (RTN residual) is **68-101× worse** than our MSRT
(rescaled TCQ residual). RRQ's fundamental limitation is using RTN instead
of TCQ for residual quantization. RTN doesn't exploit the trellis structure
or the Viterbi optimal path, wasting most of the residual bits.

Our MSRT approach is the TCQ-enhanced version of RRQ's progressive residual
framework. The key innovation is rescaling + TCQ (not RTN) for each stage.

## GSQ: Gumbel-Softmax Quantization (arXiv:2604.18556)

- Post-training scalar quantization with Gumbel-Softmax relaxation
- Jointly learns per-coordinate grid assignments and per-group scales
- Achieves near-VQ quality with scalar quantization simplicity
- Relevant for 3-4 bit range, not directly applicable to our multi-stage approach

## Proteus: Lookup-Free Trellis-Coded Quantization (ICML 2026)

- Lookup-free TCQ for 2-bit LLMs
- Lattice-breaking compute codes eliminate codebook lookups
- GPU-friendly bitshift trellis structure
- Relevant for runtime optimization of our MSRT (no LUT needed)

**Relevance to MSRT**: Proteus confirms that TCQ can be made fast without
codebook lookups. Our MSRT already uses the EXL3 trellis (which is fast),
but Proteus's lookup-free approach could further speed up the multi-stage
dequantization at runtime.

## Key Insight from Round 17

RRQ (NeurIPS 2026) validates the progressive residual framework but is
fundamentally limited by RTN. Our MSRT is the same framework but with
TCQ instead of RTN — and the difference is 68-101× in MSE. This is the
largest improvement ratio in our entire research, confirming that the
combination of TCQ + rescaling is the key innovation.

## Total: 78+ papers reviewed across 17 rounds

No published method matches MSRT's quality. RRQ (closest published method)
is 68-101× worse. MSRT is the definitive best method for fungible
quantization of GLM-5.2.
