# Round 11: ParetoQ, HBLLM, CARVQ, FraQAT — new literature

## ParetoQ (arXiv:2502.02631, NeurIPS 2025)

- First unified framework for 1-bit, 1.58-bit, 2-bit, 3-bit, 4-bit quantization
- Learning transition between 2 and 3 bits: above 3 bits, models stay close
  to pre-trained distributions; below 2 bits, representations change drastically
- Ternary, 2-bit, and 3-bit maintain comparable performance in size-accuracy trade-off
- 2-bit offers promising potential for memory reduction and speedup
- This is QAT (quantization-aware training), not PTQ — less directly applicable
- Validates our K2 tier: 2-bit is a viable base with different characteristics

## HBLLM (arXiv:2512.00862, NeurIPS 2025)

- Wavelet-enhanced 1-bit quantization using Haar wavelet transforms
- Frequency decomposition: separates low-freq (average) and high-freq (detail)
- Frequency-aware multi-parameter intra-row grouping
- L2-norm-based saliency-driven column selection
- Shared mean for non-salient weights within frequency bands
- 1.08 bits average on LLaMA2-13B

**Relevance**: Haar wavelets give multi-resolution decomposition (successive
refinement property). However, our Hadamard regularization already decorrelates
weights similarly. The wavelet approach could complement our method if applied
to the residual (before LM), but since Hadamard already removes spatial structure,
wavelets on the residual would give no additional benefit.

## CARVQ (arXiv:2510.12721)

- Corrective Adaptor with Group Residual Vector Quantization
- For LLM embedding compression (not weight compression)
- Post-training, combines linear and non-linear maps
- Group RVQ: quantize groups of channels with shared codebooks
- Less relevant to our weight quantization task

## FraQAT (arXiv:2510.14823)

- Quantization-Aware Training with fractional bits
- Progressively reduces parameter precision during training
- QAT approach, not applicable to our PTQ setting
- But validates fractional-bit quantization as viable

## Conclusions from Round 11

1. **No new method beats our approach**: All papers either use QAT (not PTQ),
   target embeddings (not weights), or use VQ (which we proved doesn't help
   after Hadamard).

2. **ParetoQ validates K2**: 2-bit is a distinct regime with different
   characteristics — our K2 crossover finding at 8 bpw aligns with their
   "learning transition" between 2 and 3 bits.

3. **HBLLM wavelets**: Interesting but Hadamard already provides the
   decorrelation that wavelets would give. No additional benefit expected.

4. **Total papers reviewed: 55+** across 11 rounds. No alternative found
   that beats tile-level mixed precision with clustered Lloyd-Max codebooks.
