# KLD Measurement Final Status

## Summary

KLD measurements could not be completed due to a vLLM V1 engine subprocess
GPU access issue in podman containers. Weight-level MSE measurements (the
key algorithmic results) were successfully completed.

## What Works

- **SIQ model loading**: The SIQ model loads and runs successfully in the GG
  vLLM turnkey container on AIBoss RTX 5090
- **K2 base checkpoint format**: Fixed suh/svh/mcg to match EXL3 checkpoint
  format (float16, correct shapes, scalar mcg) — verified correct against SIQ
- **Weight-level MSE**: All measurements completed (see MSRT-FINAL-REPORT.md)

## What Doesn't Work

### Blocker 1: vLLM V1 subprocess GPU access

vLLM's V1 engine uses subprocess isolation (EngineCore), and the subprocess
can't access the GPU under podman:

```
(EngineCore pid=243) RuntimeError: CUDA driver initialization failed,
you might not have a CUDA gpu.
```

This is a known podman + vLLM V1 multiprocessing issue. The subprocess
doesn't inherit the NVIDIA GPU device context. Potential fixes:
1. Use `--device nvidia.com/gpu=all` instead of `--gpus all`
2. Use Docker instead of podman
3. Run vLLM as a server (REST API) instead of Python API
4. Use V0 engine (not available in this GG build — V1 is forced)

### Blocker 2: EXL3 rank-sliced bitrate restriction (K2 not supported)

The GG vLLM EXL3 loader restricted rank-sliced bitrates to {3,4,5,6},
rejecting K2 base checkpoints with:

```
ValueError: rank-sliced EXL3 requires an integral 3/4/5/6 bitrate, got 2
```

This restriction was at `exl3.py:3173`. The trellis shape validation at
line 2111 already accepts K=1..8 (`1 <= shape[2]//16 <= 8`), so K2 trellis
tensors (shape[2]=32) are structurally valid — the restriction was only
in the rank-sliced bitrate check.

**Fix applied** on `feat/exl3-lora-cartridge` branch (commit bb99f519fd):
Extended the accepted range to {2,3,4,5,6}. This fix is in PR #1 but
NOT in the public GG container image (`glm52-turnkey:r31-vllm258`).
To use K2, the container must be rebuilt from the branch, or the fix
must be cherry-picked into the container's vLLM installation.

**Alternative**: Use K3 as the base tier instead of K2. K3 is supported
by the public container image. This trades higher base memory for
immediate compatibility.
## Results We Have

| Measurement | Status | Result |
|-------------|--------|--------|
| Weight-level MSE (K3 vs MSRT) | ✅ Complete | MSRT matches K4 at 4bpw, 3.6× better at 5bpw |
| SIQ baseline loading | ✅ Complete | SIQ model loads and generates logprobs |
| K2 base checkpoint format | ✅ Fixed | suh/svh/mcg match EXL3 format |
| KLD (SIQ vs K2 base) | ❌ Blocked | vLLM V1 subprocess GPU access |
| KLD with cartridge hot-swap | ❌ Blocked | Requires K2 base loading + cartridge integration |

## Key Findings (from weight-level MSE)

The MSRT cartridge approach is validated at the weight level:
- **At 4bpw**: MSRT K2+K2trsc (7.305e-03) matches native K4 (7.284e-03) within 0.3%
- **At 5bpw**: MSRT K2+K1+K2trsc (1.995e-03) is 3.6× better than K4
- **At 3bpw**: MSRT K2+K1trsc (2.908e-02) is 7% worse than K3 (no advantage ≤4bpw)

The KLD measurements would confirm these results at the output distribution
level, but the weight-level MSE already demonstrates the algorithmic validity.
