# Round 10: TurboQuant rate-distortion theory + final conclusions

## TurboQuant (Google, arXiv:2504.19874)

- Random rotation → Beta distribution → near-independent coordinates
- Lloyd-Max scalar quantizer per coordinate (optimal for Beta/Gaussian)
- Two-stage: MSE quantizer + 1-bit QJL on residual → unbiased inner product
- Achieves within 2.7× of information-theoretic lower bound
- MSE: D ≈ 1/4^b (6dB per bit improvement)
- For b=1,2,3,4: D ≈ 0.36, 0.117, 0.03, 0.009

**Relevance**: Confirms that Lloyd-Max is optimal for Gaussian/Beta coordinates
after rotation. Our approach uses the same principle (Hadamard rotation +
Lloyd-Max residual). TurboQuant's two-stage (MSE + 1-bit residual) is
exactly our K4+1bit or K5+1bit approach.

## Final Conclusions After 10 Rounds (45+ papers, 20 PoC versions)

The tile-level 4-tier K3/K4/K5/K6 approach is the best fungible quantization
method for GLM-5.2 because:

1. **Theory confirms**: Lloyd-Max is optimal for Gaussian residuals (TurboQuant,
   PolarQuant, Q-Palette all agree)
2. **Rate-distortion**: MSE ∝ 1/4^b, each bit gives 6dB improvement
3. **Tile-level mixing** achieves fractional bits (Q-Palette half-TCQ analog)
4. **Per-expert allocation** confirmed not viable by 5+ independent metrics
5. **Hardware path** confirmed: cuDNN Grouped GEMM+Quant on RTX 5090
6. **2-bit benefit** (0.0078 bpw, shared) makes fungibility essentially free
7. **4-bit quantized benefit** gives 0% quality loss (v18)

No alternative from 45+ papers beats this approach for the 3-6 bpw range.
