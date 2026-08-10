# 0c campaign pivot — stock convert_model cannot encode GLM-5.2 — 2026-08-10

## What failed

`exllamav3.conversion.convert_model` (image tree == a1-retile-sm120 @
`704aefd`, the r33 pin) asserts `Unknown architecture GlmMoeDsaForCausalLM`
— stock exllamav3 has no GLM-5.2/DSA model support. The Fruit-proxy K3/K4
conversions on GPUs 6/7 failed on this (retry loops killed).

## The real encoder (discovery)

The GLM-5.2 EXL3 checkpoints were produced by a **custom encoder bundle
published inside the K3 repo itself**:
`brandonmusic/GLM-5.2-EXL3-TR3-3.0bpw/calibration_encoder/`
(`encode_tr3_v31.py` 2805 lines, sha `e9a85a47…` — matches the pin in
rtx6kpro `glm5.2_exl3_shared_h_quantization.md`), with:

- `capture_b300.py` — pass A: run BF16 over `calibration/reap_recall_calib.jsonl`,
  capture per-expert Hessians;
- `encode_tr3_v31.py` — pass B: multi-GPU LDLQ/Trellis encode
  (`LDLQWalk`, `LayerCalib`), emits the **rank-sliced TP4 GG format
  directly** (the exact tensor_schema our segments repack); `--assemble`,
  `--upload`, `--smoke`, `--oracle`, `--bench` subcommands;
- model dims are constants annotated with their config.json sources
  (lines 145-164) — config-driven override is a small patch;
- kquant `glm52-shared-h` (local-inference-lab/kquant PR #1) layers the
  shared-H two-pass variant on this same bundle.

## Consequences

1. **The FQ lazy-encode executor's encoder IS this bundle** (or its
   quantize core) — the deterministic re-encode / `encode-of` provenance
   chain must pin `encode_tr3_v31.py`'s sha, its exllamav3 version
   (0.0.43 per the shared-H doc), and `hessian_id` = capture manifest.
2. **0c proxy plan**: adapt via a thin driver that imports
   `encode_tr3_v31` and overrides the dim constants from Fruit's
   config.json (same `GlmMoeDsaForCausalLM` arch, 13 layers / hidden 1024
   / moe_inter 512 / same 256 experts) → capture on Fruit BF16 → encode
   K3 + K4 → per-expert dKL measurement (custom, since measure_model has
   the same arch assertion) → variance → N_L solve.
3. **0c full-model**: same pipeline on GPUs 4-7 against local
   `zai-org/GLM-5.2`; the capture pass Hessians are KEPT (D3′: they seed
   the lazy-encode blend).
4. exllamav3 version note: bundle expects 0.0.43 / CUDA 12.9 (B300 era);
   image ships CUDA 13.2 + a1-retile-sm120. The bundle bootstraps its own
   ext (`bootstrap_ext_b300.py`) — reconcile against the image ext when
   adapting (SM120 kernels present in image build).

## Status

- Fruit conversions: stopped (arch assertion).
- Next: driver adaptation (`fruit_encode_driver.py`), capture pass on
  GPU 6, then K3/K4 encodes, then dKL measurement harness.
