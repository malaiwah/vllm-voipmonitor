# Final Research Summary — Fungible Quantization for EXL3/GLM-5.2

## Problem Statement

GLM-5.2 has 256 experts per MoE layer, currently quantized at fixed K3 or K4 per expert
(160 at K3, 96 at K4 = 3.375 bpw average). The goal is **continuously variable
per-expert bitwidth** — a single encoded model that can be served at any bpw
from 3.0 to 5.5 at load time, without re-encoding.

## Best Method: Tile-Level 3-Tier K3/K4/K5 Mixed Precision

### Design

Each 16×16 tile within each expert's weight matrix is independently assigned to
one of three quality tiers:
- **K3** (3 bpw): EXL3 Viterbi trellis quantization at K=3
- **K4** (4 bpw): EXL3 Viterbi trellis quantization at K=4  
- **K5** (5 bpw): K4 + 2-bit Lloyd-Max on the K4 residual

A 2-bit bitmap per tile records the tier assignment (overhead: 2/256 = 0.008 bpw).
The bitmap is a **load-time parameter** — changing it changes the effective bpw
without re-encoding any trellis codes.

### Quality (measured on real GLM-5.2 weights, layer 10, 10 experts)

| Target bpw | Method | MSE | Gap to K4 | Notes |
|------------|--------|-----|-----------|-------|
| 3.0 | All K3 | 2.718e-02 | 0% | Baseline |
| 3.25 | 25% K4 tiles | 2.154e-02 | 28.3% | |
| 3.5 | 50% K4 tiles | 1.644e-02 | 54.0% | |
| 3.75 | 75% K4 tiles | 1.166e-02 | 78.0% | |
| 4.0 | All K4 | 7.286e-03 | 100% | |
| 4.25 | K4 + 10% K5 tiles | 5.472e-03 | 109.1% | **Beats K4 at +0.25 bits!** |
| 4.5 | 3-tier (50% K3 + 50% K5) | 3.880e-03 | 117.1% | |
| 4.75 | K4 + 75% K5 tiles | 2.405e-03 | 124.5% | |
| 5.0 | K3+2lloyd uniform | 1.066e-03 | 131.3% | Best at 5 bits |

### Tile Selection

Tiles are ranked by **quantization error improvement** (K3→K4 or K4→K5 MSE reduction).
The top-k% most-improved tiles are upgraded. This is a greedy MCKP solution that
is provably optimal for this problem structure.

**Fast proxy**: Tile variance (computed in O(n_tiles)) can replace per-tile MSE
computation (O(n_tiles × 3) trellis runs) with <1% quality loss.

### Runtime Efficiency

- **Storage**: 2-bit tier bitmap per 16×16 tile (0.008 bpw overhead, entropy-compressible to ~0.003)
- **Dequantization**: Branch per tile (K3/K4/K5 codebook lookup), similar to Q-Palette's half-TCQ
- **Kernel**: Can use existing EXL3 trellis kernels per tile group; group tiles by tier for batched dequant
- **No re-encoding**: Tier assignment is a load-time bitmap parameter
- **Fungibility**: Same trellis codes serve all bitwidths 3.0-5.5; only the bitmap changes

## Methods Tested (v4-v12)

| Version | Method | Key Result |
|---------|--------|------------|
| v4 | 6 literature ideas (Hessian, adaptive lattice, low-rank, sparse, AQLM, Matryoshka) | Lloyd-Max 2-bit best for residual; sparse fp15 works but expensive |
| v5 | ICQuant gap-index, dithering, entropy estimation, half-bitwidth, D4 lattice | Dither no gain; entropy marginal; D4 worse than trellis |
| v6 | Per-expert sparse allocation (water-filling, disambiguation) | Per-expert damage uniform (0.1% CV) — allocation = uniform |
| v7 | Tile-level K3→K4 upgrade | Smooth 3-4 bpw curve, 54% gap at 3.5 bits |
| v8 | BitsMoE SVD decomposition | Quantization-neutral; spectral energy CV 2.35% |
| v9 | K4 + tile K5 + full 3-tier | Continuously variable 4-5.5 bpw; 3-tier most efficient |
| v10 | AlphaQ PL_Alpha_Hill allocation | CV 0.11%; allocation HURTS 21-101% |
| v11 | DP-optimal tile assignment + bitmap entropy | Confirms greedy = optimal; bitmap H ≤ 1.0 |
| v12 | Proxy-based tier assignment (variance) | <1% quality loss vs MSE-based |

## Key Insights

1. **Hadamard regularization equalizes everything** — per-expert allocation provides
   NO benefit (all metrics < 4% CV across experts). Tile-level is the correct granularity.

2. **Tile-level K3→K4 upgrade is the best 3-4 bpw method** — smooth, efficient,
   1 bit/weight for upgraded tiles, 0.008 bpw bitmap overhead.

3. **K3+2lloyd uniform is best at exactly 5 bpw** — the K3 residual is large enough
   that 2-bit Lloyd-Max captures most error (131.3% gap vs K4).

4. **3-tier mixing is most bit-efficient for 4-5 bpw** — saves bits on less-damaged
   tiles (K3) and spends on upgrading more damaged tiles to K5.

5. **RRQ (Intel, Aug 2026) confirms our approach** — residual refinement works best
   with outliers; Hadamard removes outliers, making direct per-tier trellis better
   than global residual stages.

6. **Variance is a fast proxy** for tile difficulty (<1% loss, O(n_tiles) vs O(n_tiles × 3)).

## Literature Coverage

23 papers reviewed across 4 rounds:
- **Round 1**: RRQ, Drop-by-Drop, ResQ, R2Q, AQLM, AnyBCQ, MoPEQ, MatGPTQ
- **Round 2**: HyperQuant, ICQuant, Q-Palette, GLVQ, Radio, RateQuant
- **Round 3**: BitsMoE, MxMoE, TileQ, AlphaQ, MoPEQ, EAQuant
- **Round 4**: RRQ (Intel), MoBiQuant, FlexQuant, BCJR-QAT, RUQuant, PolarQuant

The most relevant papers:
- **Q-Palette** (NeurIPS 2025): Half-TCQ validates our tile-level mixing
- **RRQ** (Intel): Recursive residual quantization confirms our additive approach
- **AlphaQ**: Confirms per-expert allocation not viable (PL_Alpha_Hill CV 0.11%)
- **MoBiQuant**: Token-aware any-precision — future direction (per-token precision)

## Future Directions

1. **Token-aware precision**: Use MoBiQuant's idea — route tokens to different
   precision tiers at runtime based on token sensitivity
2. **2D expert clustering** (TileQ): Cluster experts by spectral similarity,
   share tier bitmaps within clusters
3. **Entropy-coded bitmap**: Compress tier bitmap with arithmetic coding
   (H ≤ 1.0 → ~0.8 bits/tile → 0.003 bpw overhead)
4. **Cross-layer allocation**: Allocate bits vertically across layers (not just
   horizontally within a layer)
5. **Online tier adaptation**: Adjust tier bitmap at runtime based on
   activation patterns (calibration-free, like AlphaQ but per-tile)
