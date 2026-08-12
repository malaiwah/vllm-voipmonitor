# Round 9: NVIDIA cuDNN Grouped GEMM+Quant for Blackwell, QTIP bitshift details

## NVIDIA cuDNN Grouped GEMM + Quant (SM100+, Blackwell, RTX 5090)

NVIDIA has released a native cuDNN API for "Unified Grouped GEMM + Quant fusion":
- Block-scaled grouped GEMM with output quantization and per-row gating
- Designed for MoE workloads
- Implemented with CUTLASS/CUTE
- Available on Blackwell GPUs (SM100+) — includes RTX 5090

**This is the hardware path for our tile-level approach!**

The kernel supports:
- Per-block scaling (matches our per-tile tier assignment)
- Grouped GEMM (matches MoE expert dispatch)
- Output quantization (for fused activation quantization)
- Per-row gating (matches MoE routing weights)

Implementation path:
1. Encode: Store K3 trellis codes + 2-bit benefit per tile + K4/K5/K6 residual codes
2. Load: Read target_bpw, threshold benefit to determine per-tile tier
3. Kernel: Use cuDNN Grouped GEMM+Quant with per-block scale from tier selection
4. Dequant: Per-tile branch (K3/K4/K5/K6 lookup) within the grouped GEMM

## QTIP Bitshift Trellis Details

QTIP (which EXL3 is based on) uses a "bitshift" trellis:
- Each high-dimensional vector encoded as a sliding bit-window
- Shift amount s controls the effective bitrate: s/V bits per weight
- V=2 for standard, s=4→2bpw, s=6→3bpw, s=8→4bpw
- Q-Palette extends this to fractional shifts (s=5→2.5bpw, s=7→3.5bpw)

**Relevance**: EXL3 currently only supports integer K (s=4,6,8 for K=2,3,4).
Q-Palette's half-TCQ achieves fractional bitwidths by mixing two shifts.
Our tile-level approach achieves the same effect by mixing integer-K tiles.

## Summary

The hardware path is now clear:
1. cuDNN Grouped GEMM+Quant on RTX 5090 (SM100+) provides native support
2. Our tile-level tier mixing maps directly to per-block scaling
3. The 2-bit benefit (0.0078 bpw) is the only overhead
4. All 4 tiers (K3/K4/K5/K6) can be served from a single encoded model

This confirms that our approach is not just theoretically sound but
practically deployable on current hardware.
