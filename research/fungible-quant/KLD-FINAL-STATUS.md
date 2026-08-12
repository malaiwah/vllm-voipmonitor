# KLD Measurement Status

## Summary

KLD measurement infrastructure is functional — SIQ baseline logprobs collected
successfully. K2 base is blocked at three levels (Python, geometry, GPU kernel).
Pivoted to K3 base, which the SparkInfer GPU kernel supports natively.
K3 base encoding in progress with a trellis dimension-swap fix.

## What Works

- **SIQ model loading**: SIQ model loads and generates logprobs on AIBoss RTX 5090
  (container: `glm52-turnkey:r31-vllm258`, flags: `--device nvidia.com/gpu=all
  -e VLLM_ENABLE_V1_MULTIPROCESSING=0`)
- **SIQ baseline logprobs**: 10 prompts × 20 top logprobs collected
- **Weight-level MSE**: All measurements completed (see MSRT-FINAL-REPORT.md)

## Blockers and Solutions

### Blocker 1: vLLM V1 subprocess GPU access — SOLVED

vLLM's V1 engine uses subprocess isolation (EngineCore), and the subprocess
can't access the GPU under podman with `--gpus all`.

**Solutions found**:
1. `--device nvidia.com/gpu=all` instead of `--gpus all` — GPU visible to subprocess
2. `VLLM_ENABLE_V1_MULTIPROCESSING=0` — use InprocClient (no subprocess, in-process engine)

Both work. The in-process engine (`VLLM_ENABLE_V1_MULTIPROCESSING=0`) is
preferred for single-model logprob collection since it avoids IPC overhead.
SIQ baseline logprobs collected successfully with this configuration.

### Blocker 2: EXL3 Python bitrate restriction (K2) — FIXABLE

The GG vLLM EXL3 loader restricted rank-sliced bitrates to {3,4,5,6},
rejecting K2. Python overlay fix applied: extended to {2,3,4,5,6} at
`exl3.py:3173` and `exl3.py:173`. Fix is on `feat/exl3-lora-cartridge`
branch (commit bb99f519fd) but NOT in the public container image.
Overlay via volume mount works: `-v exl3_k2patched.py:/opt/.../exl3.py:Z`.

### Blocker 3: Trellis geometry mismatch — FIXED in fq_assemble_lora.py

PyTorch stores Linear weights as (out_features, in_features), but EXL3
trellis expects (in//16, out//16, K*16). The encoder used `k, n = w.shape`
treating k=input, n=output — swapped for all projections. Fixed by
transposing weight before encoding: `w = w.T.contiguous()`.

### Blocker 4: SparkInfer GPU kernel excludes K2 — NOT FIXABLE (kernel-level)

The compiled SparkInfer/b12x GPU kernel hardcodes `_TRELLIS256_BITS = (3,4,5,6)`
(SI kernel.py:130), `prepare()` rejects bit-widths outside 3–6 (SI prepare.py:1548-1552),
and the PTX dequant primitive raises for bits ∉ {3,4,5,6} (SI intrinsics.py:6185-6203).
A Python overlay cannot fix compiled PTX. K2 base loading is blocked at the
GPU kernel level regardless of Python-side fixes.

**Decision**: Pivoted to K3 base. The SparkInfer kernel supports K3 natively.
K3 base + cartridges still demonstrates the additive-residual approach:
K3 → K4-equivalent (K1 cartridge) → K5-equivalent (K2 cartridge).

| Measurement | Status | Result |
|-------------|--------|--------|
| Weight-level MSE (K3 vs MSRT) | ✅ Complete | MSRT matches K4 at 4bpw, 3.6× better at 5bpw |
| SIQ baseline logprobs | ✅ Complete | 10 prompts × 20 top logprobs collected |
| K2 base checkpoint format | ✅ Fixed | suh/svh/mcg match EXL3 format |
| Trellis dimension swap | ✅ Fixed | Weight transpose in fq_assemble_lora.py |
| K3 base encoding | 🔄 In progress | K3 base + K1/K2 cartridges, trellis geometry fixed |
| KLD (SIQ vs K3 base) | ⏳ Pending | K3 base encoding → load → collect logprobs |
| KLD (K3+cart vs SIQ) | ⏳ Pending | Cartridge hot-swap without vLLM reload |
## Key Findings (from weight-level MSE)

The MSRT cartridge approach is validated at the weight level:
- **At 4bpw**: MSRT K2+K2trsc (7.305e-03) matches native K4 (7.284e-03) within 0.3%
- **At 5bpw**: MSRT K2+K1+K2trsc (1.995e-03) is 3.6× better than K4
- **At 3bpw**: MSRT K2+K1trsc (2.908e-02) is 7% worse than K3 (no advantage ≤4bpw)

The KLD measurements would confirm these results at the output distribution
level, but the weight-level MSE already demonstrates the algorithmic validity.
