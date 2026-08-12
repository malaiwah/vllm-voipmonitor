# COMPLETE RESEARCH REPORT: Fungible Quantization for EXL3/GLM-5.2

## Final Answer: Best Method

**4-tier tile-level K3/K4/K5/K6 mixed precision** provides continuously variable
3.0-6.0 bpw from a single encoded model, with smooth monotonic quality improvement.

## Pareto Frontier (real GLM-5.2 weights, 3 experts, layer 10)

```
bpw  | MSE        | vs K4
-----|------------|--------
3.0  | 2.718e-02  | 0.0x (baseline K3)
3.5  | 1.645e-02  | 2.3x worse
4.0  | 7.288e-03  | 1.0x (K4 reference)
4.5  | 3.881e-03  | 1.9x better
5.0  | 1.068e-03  | 6.8x better
5.5  | 7.630e-04  | 9.6x better
6.0  | 5.144e-04  | 14.2x better
```

## Method Details

- **4 tiers**: K3 (trellis), K4 (trellis), K5 (K4+2bit Lloyd-Max), K6 (K5+1bit scalar)
- **Tile size**: 16×16 (256 weights per tile, matches EXL3 trellis)
- **Selection**: Greedy by benefit-per-bit (provably optimal MCKP solution)
- **Bitmap**: 2-bit per tile (0.008 bpw overhead, entropy-compressible to 0.003)
- **Fungibility**: Load-time bitmap, no re-encoding needed
- **Proxy**: Tile variance gives <1% quality loss (O(n_tiles) vs O(n_tiles×4))

## Experiments Run (v4-v16, 13 PoC versions)

| Version | What was tested | Key finding |
|---------|----------------|-------------|
| v4 | 6 literature ideas (Hessian, lattice, low-rank, sparse, AQLM, Matryoshka) | Lloyd-Max best for residual; sparse works but expensive |
| v5 | ICQuant, dithering, entropy, half-bitwidth, D4 lattice | Dither no gain; entropy marginal; D4 worse than trellis |
| v6 | Per-expert sparse allocation (GLM-5.2) | Damage uniform (0.1% CV); allocation = uniform |
| v7 | Tile-level K3→K4 upgrade | Smooth 3-4 bpw, 54% gap at 3.5 bits |
| v8 | BitsMoE SVD decomposition | Quantization-neutral; experts too uniform |
| v9 | K4+tile K5 + 3-tier | Continuously variable 4-5.5 bpw |
| v10 | AlphaQ PL_Alpha_Hill | CV 0.11%; allocation HURTS 21-101% |
| v11 | DP-optimal + bitmap entropy | Greedy=optimal; H≤1.0 |
| v12 | Proxy (variance) tier assignment | <1% loss, O(n_tiles) |
| v13 | Cross-layer allocation | Layers 10 & 40 identical (ratio=1.0001) |
| v14 | Spatial structure analysis | Tile difficulty i.i.d. random (corr~0.001) |
| v15 | Fine-grained Pareto (0.1-bit steps) | Smooth 3.0-5.0 bpw |
| v16 | 4-tier K3/K4/K5/K6 | Extended to 6.0 bpw |

## Papers Reviewed (32 papers, 5 rounds)

| Round | Papers |
|-------|--------|
| 1 | RRQ, Drop-by-Drop, ResQ, R2Q, AQLM, AnyBCQ, MoPEQ, MatGPTQ |
| 2 | HyperQuant, ICQuant, Q-Palette, GLVQ, Radio, RateQuant |
| 3 | BitsMoE, MxMoE, TileQ, AlphaQ, MoPEQ, EAQuant |
| 4 | RRQ-Intel, MoBiQuant, FlexQuant, BCJR-QAT, RUQuant, PolarQuant |
| 5 | MXFP4, CodeQuant, GAMMA, WUSH, LLVQ, EinSort, EntroLLM |

## Key Insights

1. **Hadamard equalizes everything**: per-expert, per-layer, and spatial structure
   are all removed. Only tile-level variation remains (CV=10%).

2. **Tile-level is the correct granularity**: 10% CV in tile difficulty vs 0.1% CV
   in per-expert difficulty. 100× more variation to exploit.

3. **EXL3 trellis is near-optimal at 3-5 bpw**: Leech lattice (LLVQ) only wins
   at 2 bpw. At 3+ bpw, trellis + Lloyd-Max residual is competitive.

4. **Per-expert allocation is confirmed not viable** by 5 independent metrics:
   K3 MSE (0.03% CV), K4 MSE (0.05%), disambiguation (0.06%),
   PL_Alpha_Hill (0.11%), spectral energy (2.35%).

5. **The fungibility story is validated** by RRQ, MatQuant, MatGPTQ, MoBiQuant,
   Q-Palette, and Drop-by-Drop — single-checkpoint multi-precision is an active
   research direction. Our tile-level approach is the most flexible.

6. **RRQ's analysis confirms**: residual refinement works best with outliers.
   Hadamard removes outliers, making direct per-tier trellis better than
   global residual stages.

## Future Directions

1. **MXFP4 hardware path**: RTX 5090 native support for 16-element groups
2. **Token-aware precision** (MoBiQuant): per-token tier selection at runtime
3. **Data-aware per-tile rotation** (WUSH): create tile-level diversity
4. **Entropy-coded bitmap**: arithmetic coding for tier bitmap (H≤1.0)
5. **Expert merging + quantization**: combine pruning with fungible quantization
