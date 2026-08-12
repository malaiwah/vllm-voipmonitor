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
