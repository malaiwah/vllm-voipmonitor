# Fungible Quantization: Comprehensive Research Findings

## Summary

This document synthesizes all research and experimental findings from the
fungible quantization project (v4–v10), covering additive residual encoding,
tile-level mixed precision, per-expert allocation, and spectral decomposition
for EXL3 trellis quantization on GLM-5.2.

## Best Method Found: Tile-Level 3-Tier Mixed Precision (v9)

### Method
- Each 16×16 tile independently assigned to K3 (3bpw), K4 (4bpw), or K5 (5bpw)
- K5 = K4 + 2-bit Lloyd-Max on residual
- Tile assignment guided by per-tile quantization error (most-damaged tiles upgraded first)
- 2-bit bitmap per tile records tier (0.008 bpw overhead)
- Load-time parameter: no re-encoding needed to change tier assignment

### Quality Curve (real GLM-5.2 weights, 10 experts, layer 10)

| Bits | Method | MSE | Gap to K4 |
|------|--------|-----|-----------|
| 3.0 | K3 (all tiles) | 2.718e-02 | 0% |
| 3.3 | K3 + 25% K4 tiles | 2.154e-02 | 28.3% |
| 3.5 | K3 + 50% K4 tiles | 1.644e-02 | 54.0% |
| 4.0 | K4 (all tiles) | 7.286e-03 | 100% |
| 4.2 | K4 + 10% K5 tiles | 6.506e-03 | 103.9% |
| 4.5 | 3-tier: 50% K3 + 50% K5 | 3.926e-03 | 116.9% |
| 5.0 | K5 (K3+2lloyd uniform) | 1.066e-03 | 131.3% |

### Runtime Efficiency
- Storage: 2-bit tier bitmap per 16×16 tile (0.008 bpw overhead)
- Dequantization: Branch per tile (K3/K4/K5 lookup), similar to Q-Palette half-TCQ
- Kernel: Can use existing EXL3 trellis kernels per tile group
- No re-encoding: Tier assignment is a load-time parameter

## Methods Tested and Results

### Round 1: Literature Ideas (v4)

| Idea | Method | Result |
|------|--------|--------|
| #6 Hessian-weighted scale | Optimal scale under H-weighted MSE | No gain (synthetic Hessian uniform) |
| #2 Adaptive lattice | Grid-searched α₁, α₂ per group | No gain (Lloyd-Max already optimal) |
| #3 Low-rank subspace | SVD on residual + 1-bit | No gain (Hadamard makes residuals full-rank) |
| #5 Sparse residual | fp16 on top-k% + 1-bit | 102% gap at 5 bits (but expensive: 22 bits/entry) |
| #7 Multi-codebook (AQLM) | M additive codebooks | Fails (codebooks can't capture i.i.d. Gaussian) |
| #1 Matryoshka approx | Single-step alternating opt | No gain (correction term is zero) |

### Round 2: ICQuant + Dithering + Entropy (v5)

| Idea | Result |
|------|--------|
| ICQuant gap-index coding | Reduces sparse index cost from 24 to ~6 bits/entry |
| Subtractive dithering | Slightly worse than non-dithered (noise hurts) |
| Entropy estimation | 2-bit Lloyd codes have H=1.91 bits (vs 2.0 fixed) — marginal |
| Half-bitwidth mixing | Works but dominated by tile-level approach |
| D4 lattice quantization | Much worse than trellis (no Viterbi optimization) |

### Round 3: Per-Expert Sparse Allocation (v6)

| Method | Result |
|--------|--------|
| K3 + sparse fp16 (uniform) | 36.6% gap at 4 bits (vs 100% for K4) — inefficient |
| K3 + water-filling | Same as uniform (damage is uniform across experts) |
| K3 + disambiguation-weighted | Same as uniform (disambiguation is uniform) |
| K3 + multi-step sparse | Worse than single-step (diminishing returns) |
| K3 + 2-bit Lloyd-Max | 119.9% gap at 5 bits — best at 5 bits |

### Round 4: Tile-Level Mixed Precision (v7, v9)

| Method | Range | Key Result |
|--------|-------|------------|
| K3 + tile K4 upgrade (v7) | 3.0–4.0 bpw | Smooth curve, 54% gap at 3.5 bits |
| K4 + tile K5 upgrade (v9) | 4.0–6.0 bpw | 103.9% gap at 4.2 bits (beats K4!) |
| 3-tier K3+K4+K5 (v9) | 3.0–5.5 bpw | 116.9% gap at 4.5 bits |

### Round 5: BitsMoE Spectral Decomposition (v8)

| Finding |
|---------|
| SVD decomposition is quantization-neutral (0.01% difference) |
| Shared basis Φ can be stored unquantized (~0.006 bpw amortized) |
| Spectral energy CV across experts: 2.35% (too small for allocation) |

### Round 6: AlphaQ PL_Alpha_Hill (v10)

| Finding |
|---------|
| PL_Alpha_Hill CV across experts: 0.11% (even smaller) |
| AlphaQ allocation HURTS: 21-101% worse than uniform |
| GLM-5.2 experts are spectrally homogeneous |

## Key Insights

### 1. Hadamard Regularization Equalizes Everything
The EXL3 Hadamard transform makes all experts' residuals i.i.d. Gaussian,
destroying any per-expert structure. This means:
- Per-expert allocation provides NO benefit
- Per-tile allocation is the correct granularity
- Lloyd-Max 2-bit is near-optimal for the Gaussian residual

### 2. Tile-Level > Per-Expert for Quality Differentiation
Per-tile MSE varies by 10-100× within a single expert, while per-expert
MSE varies by <0.1%. The tile is the natural unit for mixed precision.

### 3. 3-Tier > 2-Tier for Bit Efficiency
The 3-tier approach (K3/K4/K5) is more bit-efficient than 2-tier (K3/K4 or K4/K5)
because it "saves" bits on less-damaged tiles (K3) and "spends" them on upgrading
more damaged tiles to K5.

### 4. K3+2lloyd is the Gold Standard at 5 Bits
At exactly 5.0 bits, K3+2-bit Lloyd-Max uniformly applied (MSE=1.066e-03)
beats all tile-level approaches by 3.6×. The K3 residual is large enough
that 2-bit Lloyd-Max captures most of the error.

### 5. Fungibility is the Real Win
A single encoded model with a tile-level bitmap supports continuously variable
3.0–5.5 bpw by controlling the tier assignment at load time. No re-encoding
needed — just change the bitmap.

## Literature Reviewed

| Paper | Key Idea | Applicable? |
|-------|----------|-------------|
| RRQ (2511.21736) | Sequential 1-bit residual refinement | Yes — validates additive approach |
| Drop-by-Drop (2606.12876) | Matryoshka additive codebooks | Yes — successive refinement theory |
| ResQ (2412.14363) | Low-rank residual subspace | No — Hadamard makes residuals full-rank |
| R2Q (2511.21736) | Adaptive quantization lattice | No — Lloyd-Max already optimal |
| AQLM | Multi-codebook additive | No — codebooks can't capture i.i.d. Gaussian |
| HyperQuant (2606.23406) | RHT + lattice + Rice coding + dither | Partially — dither doesn't help, Rice marginal |
| ICQuant (2505.00850) | Gap-index coding for outliers | Yes — reduces sparse index cost |
| Q-Palette (2509.20214) | Fractional-bit quantizers, half-TCQ | Yes — tile-level mixing is half-TCQ analog |
| BitsMoE (2606.00079) | SVD shared basis + spectral factors | Partially — decomposition is neutral |
| MxMoE (2505.05799) | Mixed-precision MoE co-design | Yes — tile-level approach |
| TileQ (2605.09281) | 2D tiling for MoE quantization | Yes — 2D tile clustering |
| AlphaQ (2606.04980) | PL_Alpha_Hill calibration-free allocation | No — variation too small for GLM-5.2 |
| MoPEQ | Per-expert mixed precision | No — experts too homogeneous |
| SpQR | Sparse-quantized representation | No — Hadamard removes outliers |
| PolarQuant (2603.29078) | Optimal Gaussian quantization | Yes — confirms Lloyd-Max optimality |

## Recommendations

1. **Implement tile-level 3-tier K3/K4/K5** as the production fungible quantization scheme
2. **Use tile-level K3→K4 upgrades** for 3-4 bpw range (smooth, efficient)
3. **Use K3+2lloyd uniform** for 5.0 bpw (best quality at this budget)
4. **Use 3-tier mixing** for intermediate 4-5 bpw (more efficient than K4+tile K5)
5. **Do NOT use per-expert allocation** for GLM-5.2 (experts are homogeneous)
6. **Store tier assignment as 2-bit bitmap** (0.008 bpw overhead, load-time parameter)
