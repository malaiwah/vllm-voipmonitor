# Round 2-4 Literature Review — Additional Findings

## New Papers Found (Round 2-4)

### RRQ: Recurrent Residual Quantization (arXiv:2608.04048, Aug 2026, Intel)
- **Core idea**: Weights = 2-bit base + sequence of 2-bit residual corrections
- Each prefix of stages gives a usable model (2, 4, 6, 8 bits)
- Calibration-free, builds on existing quantized checkpoints
- Key insight: RRQ works best when outliers dominate (Peak-to-Mean Ratio > 10)
- For flatter distributions (like our Hadamard-regularized weights), direct fixed-bit is better

**Relevance to our work**: This is exactly our additive residual approach but with
uniform quantization stages instead of trellis. Our PoC already shows that
EXL3 trellis + Lloyd-Max residuals outperforms uniform residual stages.
The tile-level K3/K4/K5 approach is a more flexible version of RRQ that
allows per-tile precision selection rather than global stage-wise refinement.

### HyperQuant (arXiv:2606.23406, Jun 2026)
- **Core idea**: RHT + lattice quantization (E8/D4/A2) + Rice entropy coding + dither
- Rate-distortion optimal for Gaussian sources
- E8 lattice gives 0.49 dB granular gain over A2 at low rates
- Bit-stripping + Rice coding recovers 0.6-5.9% rate redundancy from HIGGS
- Subtractive dither is strictly inner-product unbiased

**Relevance**: Our v5 tested dithering (no gain) and entropy estimation (marginal).
The lattice quantization (D4) was tested in v5 but was much worse than trellis
(no Viterbi optimization). HyperQuant's E8 lattice would need custom CUDA kernels.

### ICQuant (arXiv:2505.00850, May 2025, UCLA/Meta)
- **Core idea**: Separate outliers from inliers, encode outlier indices via gap coding
- Index cost: ~0.3 bits/weight for 5% outliers (vs 1 bit/weight for bitmap)
- Key insight: outlier positions are uniformly distributed (can be enforced via permutation)
- Lemma 1: E[B] ≤ γ·b·(1 + 1/(exp(γ·(2^b-1)) - 1))

**Relevance**: Our v5 tested ICQuant gap-index coding. It reduces sparse index
cost from 24 bits/entry to ~6 bits/entry, but sparse fp16 corrections are still
11× less efficient than tile-level K3→K4 upgrades (22 bits/entry vs 1 bit/weight).

### Q-Palette (arXiv:2509.20214, Sep 2025, NeurIPS 2025)
- **Core idea**: Fractional-bit quantizers (1.5, 2.0, 2.5, ..., 5.0 bits) + half-TCQ
- Theorem 3.1: Optimal bit allocation for Gaussian: b*_l = max(η, 0.5·log2(a_l/(d_l·2^(2C))) + C)
- Half-TCQ: mix two TCQ bitwidths within a layer (e.g., 50% at 2.5b, 50% at 3.0b → 2.75b)
- Fusion-aware MSQ: jointly optimize quantizer + layer fusion

**Relevance**: Our tile-level K3/K4/K5 is analogous to half-TCQ but at tile
granularity (16×16) rather than row granularity. Q-Palette's optimal bit
allocation theorem confirms that fractional bits are essential for approaching
the Gaussian rate-distortion bound.

### BitsMoE (arXiv:2606.00079, May 2026)
- **Core idea**: SVD decomposition → shared basis Φ (unquantized) + per-expert spectral factors P_e
- ILP allocates bits per spectral component based on activation-aware reconstruction
- 2-bit on Qwen3-30B: +27.83 pp accuracy over GPTQ

**Relevance**: Our v8 tested this — decomposition is quantization-neutral (0.01%
difference), and spectral energy variation across GLM-5.2 experts is too small
(2.35% CV) for differentiated allocation.

### AlphaQ (arXiv:2606.04980, Jun 2026)
- **Core idea**: Calibration-free bit allocation using PL_Alpha_Hill (spectral heavy-tailedness)
- Smaller α = heavier tail = more important = higher precision
- Fine-grained MoEs (256 experts) show larger variance

**Relevance**: Our v10 tested this — PL_Alpha_Hill CV across GLM-5.2 experts is
0.11% (negligible). AlphaQ allocation HURTS by 21-101% because the variation
is too small. GLM-5.2 experts are spectrally homogeneous.

### TileQ (arXiv:2605.09281, May 2026, ICML)
- **Core idea**: 2D tiling layout for MoE — cluster experts by U and V similarity
- Shared U (row) and V (column) factors across expert clusters
- Fused inference kernel for multiple low-rank experts

**Relevance**: 2D expert clustering is orthogonal to our tile-level precision
mixing. Could be combined: cluster experts → shared basis → tile-level
precision within each cluster.

### MxMoE (arXiv:2505.05799, May 2025)
- **Core idea**: Joint accuracy-performance optimization for MoE mixed precision
- Generates custom GroupGEMM kernels for mixed bitwidths
- Per-block sensitivity + activation frequency → bit allocation

**Relevance**: Confirms tile-level/per-block mixed precision is the right
approach. Our tile-level K3/K4/K5 is simpler and doesn't need calibration data.

## Summary of Round 2-4 Findings

| Paper | Key Technique | Tested? | Result |
|-------|--------------|---------|--------|
| RRQ | 2-bit base + 2-bit residual stages | v4 (similar) | Lloyd-Max 2-bit beats uniform 2-bit |
| HyperQuant | E8 lattice + Rice + dither | v5 (D4, dither) | D4 worse than trellis; dither no gain |
| ICQuant | Gap-index outlier coding | v5, v6 | Reduces index cost but still inefficient |
| Q-Palette | Half-TCQ fractional bits | v7, v9 (tile-level) | Tile mixing = half-TCQ at tile granularity |
| BitsMoE | SVD shared basis | v8 | Quantization-neutral; experts too uniform |
| AlphaQ | PL_Alpha_Hill allocation | v10 | CV 0.11%, allocation HURTS |
| TileQ | 2D expert clustering | Not tested | Orthogonal; could combine with tile mixing |
| MxMoE | Calibration-based mixed precision | N/A | Confirms tile-level approach; we're calibration-free |

## Key Conclusions

1. **Our tile-level 3-tier K3/K4/K5 is a novel combination** of Q-Palette's
   half-TCQ idea (fractional bits via mixing) applied at tile granularity
   with EXL3 trellis quantization.

2. **Per-expert allocation is confirmed not viable** for GLM-5.2 by multiple
   metrics (MSE, spectral energy, PL_Alpha_Hill — all < 4% CV).

3. **RRQ's analysis confirms our finding**: residual refinement works best
   with outliers. After Hadamard regularization (which removes outliers),
   direct trellis quantization at each tier is better than residual stages.

4. **The fungibility story is validated** by RRQ, MatQuant, MatGPTQ, and
   Drop-by-Drop — single-checkpoint multi-precision is an active research
   direction. Our tile-level approach is the most flexible: continuously
   variable 3.0-5.5 bpw with a load-time bitmap parameter.
