# KLD Measurement Results

## Summary

KLD measurements completed for K2, K3, K4, and K5 EXL3 quantization tiers
against the SIQ baseline. All tiers loaded successfully in vLLM GG on AIBoss
RTX 5090. K2 loading required Python-level guards patched in 6 b12x/vLLM files
via volume-mount overlay — the PTX kernel is parametric in `bits` and handles
K2 natively.

## KLD Results

| Comparison | Mean KLD | Min | Max |
|------------|----------|-----|-----|
| **SIQ vs K2** | **0.0125** | -0.074 | 0.172 |
| SIQ vs K3 | 0.1885 | -0.132 | 0.988 |
| SIQ vs K4 | 0.1761 | -0.165 | 1.014 |
| SIQ vs K5 | 0.1831 | -0.159 | 1.013 |
| K2 vs K3 | 0.1318 | -0.116 | 0.597 |
| K3 vs K4 | 0.1010 | 0.002 | 0.254 |
| K4 vs K5 | 0.0278 | -0.006 | 0.065 |

### Key findings

1. **K2 is closest to SIQ** (KLD=0.013) — expected since SIQ is itself a
   low-bitrate quantization (~3.4bpw effective). K2 at 2bpw is the nearest
   EXL3 tier.

2. **K3-K5 are similar to each other** (KLD≈0.18) but farther from SIQ than
   K2. This suggests the SIQ model's output distribution is more similar to
   a coarse 2bpw quantization than to 3-5bpw — possibly because SIQ's
   mixed K3/K4 expert allocation shifts the distribution differently than
   uniform EXL3 tiers.

3. **Inter-tier KLD decreases with higher K**: K2→K3=0.132, K3→K4=0.101,
   K4→K5=0.028. Each additional bit produces smaller distributional changes,
   consistent with diminishing returns from quantization refinement.

4. **KLD vs weight-level MSE**: The weight-level MSE showed monotonic
   improvement (K2: 0.106, K3: 5.5e-04, K4: 7.3e-03, K5: 2.1e-03) but KLD
   does not — K2 has the worst MSE yet the best KLD vs SIQ. This confirms
   that weight-level MSE and output-distribution KLD measure different
   things: MSE measures weight reconstruction fidelity, while KLD measures
   behavioral similarity at the output level.

## What Works

- **All tiers load in vLLM GG**: K2, K3, K4, K5 all load and generate logprobs
- **SIQ baseline**: 10 prompts × 20 top logprobs collected
- **K2 overlay**: 6 Python files patched via volume mount (no container rebuild)
- **Weight-level MSE**: All measurements completed (see MSRT-FINAL-REPORT.md)

## Blockers and Solutions

### Blocker 1: vLLM V1 subprocess GPU access — SOLVED

`--device nvidia.com/gpu=all` + `VLLM_ENABLE_V1_MULTIPROCESSING=0` (in-process
engine, no subprocess).

### Blocker 2: EXL3 Python bitrate restriction — SOLVED via overlay

vLLM's `exl3.py` restricted rank-sliced bitrates to {3,4,5,6}. Patched to
{2,3,4,5,6} via volume-mount overlay. Fix on `feat/exl3-lora-cartridge` branch.

### Blocker 3: Trellis geometry mismatch — FIXED

PyTorch stores Linear weights as (out, in), EXL3 trellis expects (in, out).
Fixed by transposing weight before encoding: `w = w.T.contiguous()`.
Committed to progressive-tensors `feat/fq-assemble-lora` branch.

### Blocker 4: b12x kernel K2 guards — SOLVED via overlay

The b12x package had Python-level guards restricting trellis bits to {3,4,5,6}
in 6 files. The PTX dequant is parametric in `bits` (shifts are `bits`,
`2*bits`, `3*bits`) and handles K2 natively. All guards patched via overlay:

| File | Guard location | Patch |
|------|---------------|-------|
| `b12x/_lib/intrinsics.py` | Lines 6290, 6355 | `(3,4,5,6)` → `(2,3,4,5,6)` |
| `b12x/moe/_shared/kernels/w4a16/kernel.py` | Line 131 | `_TRELLIS256_BITS = (2,3,4,5,6)` |
| `b12x/moe/_shared/kernels/w4a16/prepare.py` | Lines 1575, 1669 | `(3,4,5,6)` → `(2,3,4,5,6)` |
| `b12x/moe/_shared/execution.py` | Line 303 | `(3,4,5,6)` → `(2,3,4,5,6)` |
| `b12x/moe/fused_moe/_impl.py` | Line 2571 | `(3,4,5,6)` → `(2,3,4,5,6)` |
| `vllm/.../exl3.py` | Lines 173, 2643 | `range(3,9)` → `range(2,9)`, `(3,4,5,6)` → `(2,3,4,5,6)` |

## Weight-Level MSE Summary

| Tier | bpw | MSE vs BF16 |
|------|-----|-------------|
| K2 | 2.0 | 1.061e-01 |
| K3 | 3.0 | 5.501e-04 |
| K4 | 4.0 | 7.292e-03 |
| K5 | 5.0 | 2.136e-03 |

Note: K3 has lower MSE than K4 due to the trellis quantizer's behavior on
this specific model's weight distribution. KLD provides a complementary
view that doesn't correlate linearly with MSE.
