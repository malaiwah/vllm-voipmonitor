# MSRT Cartridge Implementation Plan

## Overview

This plan implements MSRT (Multi-Stage Rescaled Trellis) additive cartridges
as LoRA-compatible adapters for GLM-5.2 EXL3 quantization, using the
progressive-tensors infrastructure and vLLM GG branch.

## Key Facts (from exploration)

### Fruit SIQ Proxy Model
- **Size**: hidden=1024, moe_intermediate=512, 256 experts, 13 layers, TP=1
- **MoE layers**: 3-13 (11 layers), 160 K3 + 96 K4 per layer (3.375 bpw)
- **Total size**: 3.0 GB on disk
- **BF16 version**: `malaiwah/GLM-5.2-SIQ-Fruit-Instruct-bf16` (10.1 GB on HF)
- **Expert shapes**: gate/up = (512, 1024), down = (1024, 512)

### Progressive-Tensors Infrastructure
- `fq_assemble`: takes recipe (1 K per expert) → bootable checkpoint
- `fq_prime`: extracts per-expert segments from source quants
- `fq_repack`: byte-verbatim repack into segment format
- Segment format: `layer-NNN.kK.safetensors` (per-layer, per-K)
- Recipe format: `fq-policy/2` with `bits_per_expert` dict
- Signing: ed25519, content-addressed

### vLLM GG Branch (lilab/codex/gg-exl3-r7-k345-20260810)
- EXL3 MoE: `Exl3MoEMethod` at exl3.py:2454
- `_apply_expert`: per-expert `exl3_gemm` call
- No LoRA support in EXL3 currently (only `NotImplementedError` for lm_head)
- `FusedMoEWithLoRA` exists at vllm/lora/layers/fused_moe.py
- `LoRAExpertsMixin` at vllm/model_executor/layers/fused_moe/experts/lora_experts_mixin.py
- LoRA wraps quantized base: `base_output = quant_method.apply(x)` + `lora_delta`

### MSRT Research (v35-v52)
- MSRT = K2 base + rescaled trellis residual stages
- K2+K1trsc (3bpw) = 7% worse than K3 (measured v50)
- K2+K2trsc (4bpw) = matches K4 within 0.3% (measured v50)
- K2+K3trsc (5bpw) = 3.8× better than K4 (measured v50)
- K3+K1trsc cartridge (3.42bpw) matches willfalco within 3.1% (v51)
- Dual-cartridge (K2+K1+K2+K3) creates 4-tier dynamic range (v52)
- Cartridge size for 108 experts: ~9-18 GiB (LoRA hot-swap feasible)

## Implementation Plan

### Phase 1: fq_assemble_lora (progressive-tensors repo)

**Goal**: New tool that encodes MSRT residual stages as LoRA-compatible adapter packages.

**Tool**: `tools/fq_assemble_lora.py`

**Workflow**:
1. Input: BF16 weights + base K + cartridge recipe (which experts get which residual stages)
2. Encode base K2 trellis (reuses EXL3 quantize_tiles)
3. Compute residual, rescale, encode residual trellis stages
4. Output: Two separate safetensors packages:
   - Base checkpoint (standard EXL3 format, loads normally)
   - Cartridge adapter (LoRA-format safetensors with trellis+suh+svh+scale per stage)

**Cartridge recipe format** (`fq-cartridge/1`):
```json
{
  "schema": "fq-cartridge/1",
  "base_k": 2,
  "stages": [
    {"k": 1, "label": "res1", "experts": "all"},
    {"k": 2, "label": "res2", "experts": [0,1,10,11,...]}
  ],
  "moe_layers": [3, 13]
}
```

**Cartridge adapter format**: Standard safetensors with tensors named:
```
model.layers.3.mlp.experts.0.gate_proj.rank0.trellis_res1
model.layers.3.mlp.experts.0.gate_proj.rank0.suh_res1
model.layers.3.mlp.experts.0.gate_proj.rank0.svh_res1
model.layers.3.mlp.experts.0.gate_proj.rank0.scale_res1
```

### Phase 2: Quantize Fruit Model

**Goal**: Create K2 base + cartridge segments from Fruit bf16.

**Steps**:
1. Download bf16 to AIBoss (already in progress)
2. Run `fq_assemble_lora encode` on AIBoss (using EXL3 in podman container)
3. Produce:
   - K2 base checkpoint (standard EXL3 format)
   - Cart A: K1trsc for all 256 experts (K3-equivalent)
   - Cart B: K2trsc for 96 hot experts (K4-equivalent, matching SIQ's tier allocation)
4. Measure weight-level MSE vs bf16 original

### Phase 3: EXL3 LoRA Support (vllm-voipmonitor repo)

**Goal**: Add LoRA cartridge support to EXL3 in vLLM GG.

**Changes**:
1. `Exl3Config.get_supported_lora_modules()` → return `["gate_proj", "up_proj", "down_proj"]`
2. New `Exl3LoRAMoEMethod` or extend `Exl3MoEMethod`:
   - `create_lora_weights()`: allocate per-stage trellis+suh+svh+scale storage
   - `set_lora()`: load cartridge tensors from LoRA adapter
   - `_apply_expert()`: if cartridge active, run additional GEMM passes and sum
3. The apply path:
   ```python
   output = _exl3_gemm(x, trellis_base, suh_base, svh_base, mcg, mul1)
   if cartridge_active:
       for stage in cartridge.stages:
           res = _exl3_gemm(x, stage.trellis, stage.suh, stage.svh, mcg, False)
           output += res * (1.0 / stage.scale)
   ```

### Phase 4: KLD Measurements

**Goal**: Measure KLD between different configs without reloading vLLM.

**Steps**:
1. Start vLLM with K2 base checkpoint
2. Send calibration prompts, collect logits (K2 baseline)
3. Load Cart A (K3-equivalent) via `add_lora`, collect logits
4. Unload Cart A, load Cart B (K3K4-mixed) via `add_lora`, collect logits
5. Also run SIQ model separately for reference logits
6. Compute KLD: `KL(P_bf16 || P_quant)` for each config
7. Generate comparison charts

### Constraints
- **Do NOT restart qwen36-27b service** (it's currently active on AIBoss)
- Fruit model is TP=1, small (3 GB) — can run alongside qwen36 on RTX 5090
- Use podman container with EXL3 for encoding
- GG vLLM branch for serving (may need to build or use existing container)

## References
- MSRT research: `research/fungible-quant/poc/V50-LOW-BITRATE-MSRT.md` through `V52-DUAL-CARTRIDGE-MSRT.md`
- Feasibility plan: `research/fungible-quant/MSRT-CARTRIDGE-FEASIBILITY-AND-PLAN.md`
- Cost analysis: `research/fungible-quant/GLM52-MSRT-COST-ANALYSIS.md`
- Progressive-tensors: `research/fungible-quant/MSRT-CARTRIDGE-PROGRESSIVE-TENSORS.md`
- vLLM LoRA analysis: `research/fungible-quant/MSRT-CARTRIDGE-PROGRESSIVE-TENSORS.md` (Section 2)
