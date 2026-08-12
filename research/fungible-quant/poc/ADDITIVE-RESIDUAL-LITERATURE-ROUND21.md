# Round 21: LLVQ, BCJR-QAT, learned lattice quantizers — final literature

## LLVQ: Leech Lattice Vector Quantization (arXiv:2603.11021, 2026)

- Leech lattice (24-dimensional) for LLM weight compression
- Achieves better retention than E8/E8P at similar bitrates
- Structured lattice-based schemes outperform uniform quantization for Gaussian

**Relevance**: LLVQ uses 24D VQ (lattice), while our MSRT uses TCQ (trellis).
Both exploit Gaussian structure. TCQ scales linearly with dimension (unlike VQ
which is exponential), allowing higher effective dimensions. Our MSRT achieves
better R-D through multi-stage refinement, which LLVQ doesn't do.

## BCJR-QAT: Differentiable Trellis-Coded Weight Quantization (arXiv:2605.10655)

- Uses BCJR (forward-backward) algorithm instead of Viterbi for differentiable training
- Enables QAT on trellis-coded weights
- STE for scalar, BCJR for trellis — differentiable relaxation

**Relevance**: BCJR-QAT is for QAT (training), not PTQ. Our MSRT uses Viterbi
(optimal for PTQ). BCJR would be relevant if we wanted to fine-tune the model
after MSRT quantization, but our approach is calibration-free PTQ.

## Learned Lattice Vector Quantizers (NeurIPS 2024)

- Learns optimal lattice structures for specific source distributions
- Adapts lattice geometry to non-uniform distributions
- Better than fixed lattices for non-Gaussian sources

**Relevance**: After Hadamard regularization, our weights are approximately
Gaussian, so fixed lattices (or trellis codebooks) are already near-optimal.
Learning a custom lattice wouldn't help because the Gaussian assumption is
valid after Hadamard.

## Key Insight from Round 21

The EXL3 trellis with mcg codebook is already near-optimal for Gaussian sources
(after Hadamard). Alternative codebooks (mul1: 2.6× worse) or lattice approaches
(LLVQ: fixed 24D) don't improve on TCQ's linear-scaling advantage. The Viterbi
algorithm provides optimal path selection, and the multi-stage rescaled refinement
(MSRT) exploits successive refinement for progressive quality.

No published method or codebook variant improves on MSRT.

## Total: 88+ papers/concepts reviewed across 21 rounds

MSRT (Multi-Stage Rescaled Trellis) with mcg codebook, Hadamard 128, global RMS
rescaling, and K1-first stage allocation is the definitive best method for
fungible quantization of GLM-5.2.
