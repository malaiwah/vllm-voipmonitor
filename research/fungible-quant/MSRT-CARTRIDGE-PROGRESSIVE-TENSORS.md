# MSRT Additive Cartridge via Progressive Tensors + vLLM LoRA Hot-Swap

## Executive Summary

**Yes — `fq_assemble` is the direct inspiration for a cartridge tool, and vLLM's
LoRA hot-swap mechanism can serve as the runtime delivery vehicle.** The
combination enables online, per-expert additive accuracy without reloading vLLM.

The architecture:

1. **Progressive Tensors** already publishes per-expert K2/K3/K4/K5 segments
   and assembles them into mixed-K checkpoints via `fq_assemble`
2. **MSRT cartridges** are additive trellis stages (K1, K2, K3) that correct
   a K2 base — stored as per-expert safetensors fragments, same as FQ segments
3. **vLLM LoRA hot-swap** provides `add_lora` / `remove_lora` APIs and
   `/v1/load_lora_adapter` REST endpoints for runtime adapter management
4. The **cartridge tool** (`fq_cartridge`) would extend the FQ pipeline to
   encode, publish, fetch, and assemble MSRT residual stages as LoRA-like
   adapters that vLLM can load/unload on the fly

---

## 1. Progressive Tensors: What Already Exists

### 1.1 FQ Segments Repository

[`malaiwah/GLM-5.2-EXL3-FQ-segments`](https://huggingface.co/malaiwah/GLM-5.2-EXL3-FQ-segments)
publishes per-expert EXL3 trellis segments at K2, K3, K4, K5:

| Tier | Segments | Layers | Per-expert size |
|------|----------|--------|-----------------|
| K3 (base) | 76 | 3-78 | 13.65 MiB |
| K2 (fast-load) | 75 | 3-77 | 9.15 MiB |
| K4 (promotion) | 56 | 3-58 | 18.15 MiB |
| K5 (hot) | 24 | 3-10, 35-50 | 22.65 MiB |

Each segment is pure safetensors, content-addressed, signed (ed25519), and
independently verifiable. Segments contain routed-expert tensors only
(trellis + suh + svh + mcg); non-expert tensors come from the source checkpoint.

### 1.2 fq_assemble

[`progressive-tensors`](https://github.com/malaiwah/progressive-tensors) provides:

| Tool | Role |
|------|------|
| `fq_fetch` | HTTP-Range-fetch only the expert spans a recipe needs |
| `fq_assemble` | Assemble segments + source → bootable checkpoint |
| `fq_verify` | Byte-identity verification against source |
| `fq_release` | Signed release verification |
| `fq_prime` | Encode new segments from a source quant |
| `fq_repack` | Byte-verbatim repack of existing quants into segment format |
| `fq_eps` | Per-expert encode error measurement |

**Key architecture:** The recipe (`fq-policy/2` schema) specifies one K per
expert per layer. `fq_assemble` reads segments, validates attestations, and
writes a bootable checkpoint with mixed-K metadata (`hybrid_tr3_tail.bits="mixed"`,
`tier_bitmap.json:bits_per_expert`).

### 1.3 The Recipe Format

```json
{
  "schema": "fq-policy/2",
  "bits_per_expert": {
    "3": [3, 3, 4, 3, 4, ...],   // layer 3: 256 K values
    "4": [3, 4, 3, 3, 4, ...],   // layer 4: 256 K values
    ...
  }
}
```

This is a **replacement** model: each expert gets exactly one K value. The
assembled checkpoint is a standard mixed-K EXL3 checkpoint that loads via
the GG vLLM EXL3 loader.

### 1.4 What fq_assemble Does NOT Do

- **No additive/residual stages:** Each expert has one trellis tensor, not
  multiple stages summed
- **No runtime hot-swap:** Assembly is offline; changing the recipe requires
  reassembling and reloading the model
- **No MSRT:** The segments are independent K-tier alternatives, not
  successive-refinement residual stages

---

## 2. vLLM LoRA Hot-Swap: What Already Exists

### 2.1 Runtime API

vLLM v1 engine provides full LoRA adapter hot-swap without model reload:

| API | Level | Method |
|-----|-------|--------|
| `add_lora(LoRARequest)` | Engine | Load adapter from disk, register, activate |
| `remove_lora(lora_id)` | Engine | Deactivate and unregister adapter |
| `pin_lora(lora_id)` | Engine | Pin adapter in GPU slots (no LRU eviction) |
| `list_loras()` | Engine | List all registered adapters |
| `POST /v1/load_lora_adapter` | REST | HTTP endpoint (requires `VLLM_ALLOW_RUNTIME_LORA_UPDATING=True`) |
| `POST /v1/unload_lora_adapter` | REST | HTTP endpoint |

### 2.2 Two-Tier Cache

| Tier | Location | Capacity | Purpose |
|------|----------|----------|---------|
| Registered | CPU | `max_cpu_loras` | Adapters loaded from disk, CPU-resident |
| Active | GPU | `max_loras` | Adapters copied into pre-allocated GPU slots |

Adapters move CPU→GPU on activation, GPU→CPU on deactivation (LRU eviction).
Both capacities are configurable at engine startup.

### 2.3 LoRA + Quantization Compatibility

**Critical finding:** LoRA is architecturally compatible with quantized weights.
The LoRA layer wraps the base quantized layer:

```python
# Simplified from vllm/lora/layers/base.py
class BaseLayerWithLoRA:
    def apply(self, x, ...):
        base_output = self.base_layer.quant_method.apply(x, ...)
        lora_delta = punica_kernels(x, self.lora_a, self.lora_b)
        return base_output + lora_delta * self.scaling
```

The base quant method is **never bypassed** — LoRA adds a delta on top.
This works with GPTQ, AWQ, bitsandbytes, FP8, compressed-tensors, and Marlin.

### 2.4 EXL3 + LoRA Status

EXL3 on the lilab GG branch has **no explicit LoRA integration**:
- No `get_supported_lora_modules` declaration
- Only mention: `NotImplementedError` for EXL3 lm_head + `--lora-extra-vocab-size`
- But the quant-agnostic wrapping pattern should work in principle

---

## 3. The MSRT Cartridge Concept

### 3.1 What an MSRT Cartridge Contains

Unlike LoRA (low-rank A·B decomposition), an MSRT cartridge contains **full-rank
additive trellis stages**:

```
Base checkpoint (K2):  model.layers.10.mlp.experts.0.gate_proj.rank0.trellis
                                                .rank0.suh
                                                .rank0.svh
                                                .rank0.mcg

Cartridge (K3 res):   model.layers.10.mlp.experts.0.gate_proj.rank0.trellis_res
                                                .rank0.suh_res
                                                .rank0.svh_res
                                                .rank0.scale_res   (float32 scalar)
```

The cartridge is applied at runtime as:
```
output = exl3_gemm(x, trellis_base, suh_base, svh_base)
       + (1/scale) * exl3_gemm(x, trellis_res, suh_res, svh_res)
```

### 3.2 How This Maps to LoRA Infrastructure

| LoRA concept | MSRT cartridge equivalent |
|-------------|--------------------------|
| `lora_a` (rank×input) | `trellis_res` (full-rank trellis) |
| `lora_b` (output×rank) | (none — trellis IS the weight) |
| `scaling = alpha/rank` | `1/scale_res` (float32 rescaling factor) |
| `punica_kernels(x, a, b)` | `exl3_gemm(x, trellis_res, suh_res, svh_res)` |
| Low-rank delta | Full-rank trellis residual |

The key difference: LoRA uses low-rank matrices (small A·B), while MSRT uses
full-rank trellis-quantized residuals (same shape as base, at K1-K3 bits).
But the **delivery mechanism is identical**: load additional weight tensors,
apply them as additive corrections at runtime.

### 3.3 Cartridge Selectivity

A cartridge can cover a subset of experts — the user's "recipe" specifies which
experts get additive correction:

```json
{
  "schema": "msrt-cartridge/1",
  "base_checkpoint": "malaiwah/GLM-5.2-EXL3-FQ-segments@K2",
  "stages": [
    {"k": 3, "label": "res1"}
  ],
  "cartridge_experts": {
    "3": [0, 1, 10, 11, 12, 100, 101, 102, 103, 104],
    "4": [0, 1, 10, 11, 12, 100, 101, 102, 103, 104],
    ...
  }
}
```

Experts not in the cartridge list stay at K2 base only. This is the memory
budget mechanism — the user trades accuracy for memory by choosing how many
experts get the cartridge.

---

## 4. Proposed Tool: `fq_cartridge`

### 4.1 Inspiration from fq_assemble

`fq_assemble` takes a recipe (one K per expert) and assembles a checkpoint by
selecting segments. `fq_cartridge` would take a cartridge recipe (additive
stages per expert) and:

1. **Encode** residual stages from BF16 weights (using MSRT pipeline)
2. **Publish** residual segments to HF (same safetensors + attestation format)
3. **Fetch** only the cartridge segments a recipe needs
4. **Assemble** into a LoRA-compatible adapter package
5. **Load** into vLLM via the LoRA hot-swap API

### 4.2 Tool Pipeline

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  BF16 weights │────▶│  fq_cartridge │────▶│  Cartridge   │
│  (GLM-5.2)   │     │  encode       │     │  segments    │
└──────────────┘     │  (MSRT)       │     │  (HF repo)   │
                     └──────────────┘     └──────┬───────┘
                                                 │
                     ┌──────────────┐            │
                     │  fq_cartridge │◀───────────┘
                     │  fetch        │
                     │  (recipe)     │
                     └──────┬───────┘
                            │
                     ┌──────▼───────┐
                     │  Cartridge    │
                     │  adapter      │
                     │  (safetensors)│
                     └──────┬───────┘
                            │
                     ┌──────▼───────┐
                     │  vLLM        │
                     │  add_lora()  │
                     │  remove_lora()│
                     └──────────────┘
```

### 4.3 Encode Step (`fq_cartridge encode`)

Input: BF16 weights + base K2 quantization
Output: Per-expert residual trellis segments

```python
for each MoE layer:
    for each expert:
        w_reg = regularize(w_bf16)           # Hadamard + scaling
        q_k2 = quantize_trellis(w_reg, K=2)  # Base tier
        residual = w_reg - q_k2              # K2 residual
        scale = cbs / rms(residual)           # Rescale factor
        q_k3 = quantize_trellis(residual * scale, K=3)  # Residual tier
        save_segment(trellis=q_k3, suh=..., svh=..., scale=scale)
```

This is exactly what PoC v50 does — just packaged as a segment encoder.

### 4.4 Fetch Step (`fq_cartridge fetch`)

Identical to `fq_fetch` — reads the cartridge recipe, HTTP-Range-fetches only
the residual segments needed, verifies attestations.

### 4.5 Assemble Step (`fq_cartridge assemble`)

Output: A LoRA-compatible adapter package

```
my-cartridge/
├── adapter_model.safetensors    # LoRA-format wrapper
├── adapter_config.json          # Cartridge metadata
└── tier_bitmap.json             # Which experts have cartridge
```

The `adapter_model.safetensors` would contain tensors like:
```
model.layers.10.mlp.experts.0.gate_proj.rank0.trellis_res
model.layers.10.mlp.experts.0.gate_proj.rank0.suh_res
model.layers.10.mlp.experts.0.gate_proj.rank0.svh_res
model.layers.10.mlp.experts.0.gate_proj.rank0.scale_res
```

### 4.6 Runtime Loading

```python
# vLLM engine startup
engine = LLM(
    model="assembled-k2-checkpoint",   # K2 base
    enable_lora=True,
    max_loras=4,                        # Up to 4 cartridge variants
    max_cpu_loras=16,                   # CPU cache for more variants
    max_lora_rank=0,                    # Not used (full-rank, not low-rank)
)

# Load a K2→K5 cartridge (adds K3 residual to all experts)
engine.add_lora(LoRARequest(
    lora_name="k2-to-k5-cartridge",
    lora_int_id=1,
    lora_path="./my-cartridge",
))

# Serve with K5-equivalent quality
engine.generate(...)

# Unload cartridge — back to K2 base quality
engine.remove_lora(1)

# Load a different cartridge (e.g., only top-96 experts)
engine.add_lora(LoRARequest(
    lora_name="k2-top96-cartridge",
    lora_int_id=2,
    lora_path="./top96-cartridge",
))
```

Or via REST:
```bash
curl -X POST http://localhost:8000/v1/load_lora_adapter \
  -d '{"lora_name": "k2-to-k5", "lora_path": "./my-cartridge"}'

curl -X POST http://localhost:8000/v1/unload_lora_adapter \
  -d '{"lora_name": "k2-to-k5"}'
```

---

## 5. Key Differences from Standard LoRA

| Aspect | Standard LoRA | MSRT Cartridge |
|--------|--------------|----------------|
| Weight format | Low-rank A·B (float16) | Full-rank trellis (int16) |
| Rank | Small (8-256) | Full (2048×6144) |
| Storage per adapter | MB-scale (rank × dimensions) | GB-scale (full trellis per expert) |
| Compute | Punica kernels (batched low-rank) | `exl3_gemm` (fused dequant+matmul) |
| Accuracy improvement | Fine-tuning delta | Quantization correction |
| Load time | Seconds (small) | Minutes (large, but cached on GPU) |
| Hot-swap | ✓ (vLLM supports) | ✓ (same mechanism) |

The main challenge: **cartridge size**. A standard LoRA adapter is MB-scale
(rank × dimensions). An MSRT cartridge is GB-scale (full trellis per expert).
At K3 residual on all 256 experts × 75 layers × 3 projections:
- 256 × 75 × 3 × 13.65 MiB = **~770 GB** total, ~193 GB per TP4 rank

This is the same size as the base checkpoint. The cartridge is not a small
delta — it's a second full checkpoint. The LoRA LRU cache mechanism still
works (CPU↔GPU swapping), but each adapter occupies a full checkpoint's
worth of GPU memory when active.

### 5.1 Memory-Efficient Cartridge Variants

To make cartridges practical for hot-swap:

| Variant | Cartridge size (TP4/rank) | Experts covered | Quality |
|---------|--------------------------|-----------------|---------|
| Full (all 256 experts, K3) | ~193 GiB | All | K5-equivalent |
| Top-96 (37.5%, K3) | ~72 GiB | 96/256 | Near K5 on hot experts |
| Top-32 (12.5%, K3) | ~24 GiB | 32/256 | K4 on hot experts |
| Top-32 (12.5%, K1) | ~8 GiB | 32/256 | K3 on hot experts |

The **top-32 K1 cartridge (~8 GiB)** is the most practical for hot-swap:
fits in GPU memory alongside the base K2 checkpoint, provides K3-quality
on the 32 most-routed experts, and can be loaded/unloaded in seconds.

### 5.2 Multi-Variant Serving

With `max_loras=4`, vLLM could hold 4 cartridge variants simultaneously:

| Slot | Cartridge | Use case |
|------|-----------|----------|
| 1 | Full K3 (all experts) | High-accuracy mode |
| 2 | Top-96 K3 | Balanced mode |
| 3 | Top-32 K1 | Fast mode |
| 4 | (reserved for dynamic load) | Experimental |

Different requests can use different cartridges via the LoRA mapping mechanism
(per-request `lora_int_id` in the sampling parameters).

---

## 6. Implementation Requirements

### 6.1 New Code

| Component | Effort | Description |
|-----------|--------|-------------|
| `fq_cartridge encode` | Medium | MSRT encoder (reuse PoC v50 code) |
| `fq_cartridge fetch` | Low | Reuse `fq_fetch` with cartridge recipe |
| `fq_cartridge assemble` | Medium | Package as LoRA-compatible adapter |
| EXL3 LoRA wrapper | Medium | `Exl3LoRALayer` that sums base GEMM + cartridge GEMM |
| EXL3 `get_supported_lora_modules` | Low | Declare MoE expert modules as LoRA targets |
| Cartridge recipe schema | Low | JSON schema for additive stages |

### 6.2 EXL3 LoRA Integration

The EXL3 quant method needs to declare LoRA support:

```python
class Exl3Config(QuantizationConfig):
    def get_supported_lora_modules(self):
        return ["gate_proj", "up_proj", "down_proj"]  # MoE expert modules
```

And a new `Exl3MsrtCartridgeLayer` that wraps the base EXL3 layer:

```python
class Exl3MsrtCartridgeLayer(BaseLayerWithLoRA):
    def apply(self, layer, x, bias=None):
        # Base EXL3 GEMM
        base_output = self.base_layer.apply(layer, x, bias)
        # Cartridge GEMMs (if active)
        if self.lora is not None:
            for stage in self.lora.stages:
                res_output = _exl3_gemm(
                    x, stage.trellis, stage.suh, stage.svh,
                    mcg=True, mul1=False
                )
                base_output += res_output * (1.0 / stage.scale)
        return base_output
```

### 6.3 What Doesn't Need to Change

- **CUDA kernels:** `exl3_gemm` and `b12x_trellis_linear` are reused as-is
- **Checkpoint format:** Cartridge segments are standard safetensors
- **Attestation/signing:** Reuse FQ's ed25519 attestation system
- **FQ fetch infrastructure:** `fq_fetch` HTTP-Range mechanism works as-is
- **vLLM LoRA manager:** LRU cache, CPU↔GPU swapping, per-request mapping

---

## 7. Feasibility Assessment

### 7.1 What Works Today

- ✅ FQ segments published at K2/K3/K4/K5 (per-expert, signed, verified)
- ✅ `fq_assemble` produces bootable mixed-K checkpoints
- ✅ vLLM LoRA hot-swap with `add_lora`/`remove_lora` APIs
- ✅ LoRA wrapping over quantized base layers (works with GPTQ, AWQ, FP8)
- ✅ MSRT encoding pipeline (PoC v35-v50, 50 experiments)
- ✅ K2 base checkpoint available on HF (`GLM-5.2-EXL3-FQ-segments`)

### 7.2 What Needs Building

- 🔨 `fq_cartridge` tool (encode + fetch + assemble)
- 🔨 EXL3 LoRA wrapper layer in vLLM
- 🔨 EXL3 `get_supported_lora_modules` declaration
- 🔨 Cartridge recipe schema
- 🔨 EXL3 cartridge GEMM summation in apply()

### 7.3 Risks

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Cartridge too large for GPU hot-swap | High | Use top-32 K1 cartridge (~8 GiB) |
| EXL3 LoRA wrapper has CUDA graph issues | Medium | `exl3_gemm` is already a custom_op; test under capture |
| Multiple cartridge variants exceed GPU memory | Medium | Limit `max_loras=2`, use CPU LRU for inactive |
| Encode time for new cartridges | Low | Encode is offline; PoC v50 ran in ~6 min for 10 experts |

---

## 8. Conclusion

The `fq_assemble` tool is the **direct architectural template** for an MSRT
cartridge system. The FQ pipeline already handles per-expert segment encoding,
publishing, fetching, verification, and assembly. Extending it to handle
additive (multi-stage) segments rather than replacement (single-K) segments is
the natural next step.

vLLM's LoRA hot-swap mechanism provides the runtime delivery: `add_lora` /
`remove_lora` APIs load and unload adapters without model reload, the two-tier
LRU cache manages CPU↔GPU memory, and per-request adapter mapping enables
multi-variant serving.

The combination enables **online quantization tuning**: a user loads a K2 base
checkpoint, then hot-swaps cartridges to match their accuracy/memory tradeoff
without restarting vLLM. Multiple variants can coexist (high-accuracy,
balanced, fast), and new cartridges can be encoded and loaded on demand.
