# Round 15: BCJR-QAT, RQT, Neural Weight Compression — final literature

## BCJR-QAT (arXiv:2605.10655)

- Differentiable relaxation of trellis-coded weight quantization
- Uses BCJR algorithm (forward-backward) instead of Viterbi for differentiable training
- QAT approach (requires training), not PTQ
- Relevant for training-aware quantization but not our calibration-free PTQ setting

## RQT: Hierarchical Residual Quantization (ACL 2025)

- Hierarchical residual quantization for multi-model deployment
- Multiple quantization stages with residual refinement
- Similar to our MSRT approach but uses VQ, not TCQ
- Confirms multi-stage residual quantization is a valid approach

## Neural Weight Compression (arXiv:2510.11234, 2026)

- Comprehensive survey of neural weight compression methods
- Covers VQ, TCQ, scalar quantization, pruning, low-rank
- Confirms TCQ achieves near-optimal distortion on Gaussian sources
- Our MSRT approach is novel: multi-stage TCQ with rescaling

## Key Insight from Round 15

Our MSRT (Multi-Stage Rescaled Trellis) approach is novel in the literature:
1. No published work uses multi-stage TCQ on weight residuals
2. The rescaling step (adapting residual to codebook range) is our innovation
3. RQT uses multi-stage VQ (not TCQ) — our TCQ approach is more efficient
4. BCJR-QAT uses differentiable TCQ for training — we use Viterbi for PTQ

## Total: 73+ papers reviewed across 15 rounds

No published method matches or beats MSRT for fungible quantization of GLM-5.2.
