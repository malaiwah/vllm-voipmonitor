# Round 8: Joint Pruning+Quantization, Stochastic Rounding, Final Thoughts

## Joint Structural Pruning + Mixed-Precision Quantization (arXiv:2606.07819)
- Combines expert pruning with per-layer mixed precision
- Prunes unimportant experts, quantizes remaining at variable bitwidth
- Unified optimization framework

**Relevance**: Orthogonal to our tile-level approach. Could stack: prune experts
→ fungible quantize remaining experts at tile level.

## Stochastic Rounding (arXiv:2606.00312)
- Stochastic rounding increases small singular values
- Uniformly dithered quantizer with stochastic rounding
- Benefits for training stability, not inference

**Relevance**: Our v5 tested dithering — no gain for inference. Stochastic
rounding is mainly useful during training (QAT), not PTQ.

## AutoQRA (ICML 2026)
- Joint optimization of mixed-precision quantization + LoRA adapters
- For efficient LLM fine-tuning

**Relevance**: Fine-tuning direction. Our approach is PTQ (no fine-tuning).

## Final Thoughts After 8 Rounds (40+ papers, 19 PoC versions)

The tile-level 4-tier K3/K4/K5/K6 approach is robust because:
1. It works within the existing EXL3 trellis format (no format changes)
2. It's calibration-free (uses only weight statistics)
3. The 10% tile difficulty CV is a fundamental property of Gaussian sources
4. Per-expert, per-layer, and spatial structure are all removed by Hadamard
5. The 2-bit benefit (0.0078 bpw, shared) makes fungibility essentially free

No alternative approach from 40+ papers beats it for the 3-6 bpw range.
The only improvements would come from:
- Hardware-specific kernels (MXFP4, FLUTE)
- Training-based methods (GLVQ, QAT) — but these are not PTQ
- Token-aware precision (MoBiQuant) — future runtime extension
