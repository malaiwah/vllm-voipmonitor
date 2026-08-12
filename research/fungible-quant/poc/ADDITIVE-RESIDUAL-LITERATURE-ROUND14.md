# Round 14: Successive refinement of TCQ + Drop-by-Drop + rate-distortion

## Successive Refinement of TCQ (Jafarkhani, IEEE TIT 1999)

- TCQ is successively refinable: by selecting different sub-streams, various
  rates and distortions can be achieved from a single encoding
- This is exactly our fungible quantization goal!
- The patent US6125149A describes successively refinable TCQ in detail
- Key: the trellis structure allows progressive bit addition without re-encoding

**Relevance**: Our rescaled trellis-on-residual (v35-v37) inherits this
successive refinement property. By encoding K2 base + K5 rescaled trellis,
we can progressively refine by reading more trellis bits.

## Drop-by-Drop (arXiv:2606.12876, 2026)

- Multi-bitwidth PTQ using additive codebooks with successive refinement
- Matryoshka-style supervision in loss function
- Ordered subsets of codebooks yield accurate partial reconstructions
- Theoretically grounded: Gaussian weights can be optimally reconstructed
  with increasing fidelity as additional bits are incorporated
- QAT approach (requires training), not pure PTQ

**Relevance**: Their theoretical result confirms our approach is sound:
Gaussian weights (after Hadamard) ARE successively refinable under MSE.
Our trellis+rescaled-trellis achieves this without training.

## Nested-Lattice Quantized Matrix Multiplication (arXiv:2505.13164)

- Successive refinement framework with nested lattice quantization
- Quadratic Gaussian rate-distortion bound achieved
- Small lookup tables for fast decoding

**Relevance**: Nested lattices are the VQ analog of our trellis-on-residual.
Our approach (trellis for base + rescaled trellis for residual) is a form
of nested quantization where both stages use the same quantizer family.

## Key Insight from Round 14

Our approach (K2 base + K5 rescaled trellis residual) is a form of
**multi-stage trellis quantization** — the same trellis quantizer applied
to the residual of the first stage. This is:
1. Successively refinable (Jafarkhani 1999)
2. Optimal for Gaussian sources (Drop-by-Drop theory)
3. A form of nested quantization (lattice theory)

The rescaling step is our key innovation: it adapts the residual to the
trellis codebook's expected input range, making the same codebook work
for both the weight and the residual distributions.

## Total: 70+ papers reviewed across 14 rounds
