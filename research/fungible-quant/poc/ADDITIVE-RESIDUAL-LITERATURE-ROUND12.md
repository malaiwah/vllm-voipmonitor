# Round 12: BPDQ (ICML 2026) + final conclusions

## BPDQ: Bit-Plane Decomposition Quantization (arXiv:2602.04163, ICML 2026)

- Constructs variable quantization grid via bit-planes and scalar coefficients
- Sign-magnitude decomposition: sign plane (1 bit) + magnitude planes
- Iteratively refines using second-order information
- 2-bit regime: enables Qwen2.5-72B on single RTX 3090 with 83.85% GSM8K

**Tested in v32**: BPDQ-style bit-plane on our residual:
- 2-bit: 22.7% worse than Lloyd-Max
- 4-bit: 3.9% worse than Lloyd-Max

**Conclusion**: Bit-plane decomposition is suboptimal for Gaussian residuals.
Lloyd-Max already provides optimal non-uniform spacing. BPDQ's advantage is
for non-Gaussian weight distributions where fixed grids are suboptimal —
but Hadamard regularization makes our residuals Gaussian.

## Final Conclusions After 12 Rounds (60+ papers, 32 PoC versions)

The tile-level 8-tier K2/K3/K4/K4+1LM/K4+2LM/K4+3LM/K2+6LM/K4+6LM approach
with c128 clustered codebooks and entropy-aware tier allocation is the optimal
fungible quantization method for GLM-5.2.

### Key results (entropy-aware Pareto, c128, 10 experts):

| Target bpw | MSE | vs K4 |
|------------|-----|-------|
| 3.0 | 2.718e-02 | 373% |
| 4.0 | 7.286e-03 | 100% |
| 5.0 | 2.793e-03 | 38% |
| 6.0 | 8.546e-04 | 12% |
| 7.0 | 1.994e-04 | 2.7% |
| 8.0 | 7.015e-05 | 1.0% |
| 9.0 | 3.230e-05 | 0.4% |
| 10.0 | 1.905e-05 | 0.3% |

### No alternative from 60+ papers beats this approach

- BPDQ (ICML 2026): 4-23% worse than Lloyd-Max
- ParetoQ (NeurIPS 2025): QAT, not PTQ
- HBLLM (NeurIPS 2025): Wavelets don't help after Hadamard
- VPTQ: VQ doesn't help after Hadamard decorrelates
- Product quantization: 7-42% worse than scalar LM
- Stacking: 32-123% worse than single large LM
- Per-expert allocation: 0% gain (experts homogeneous)
