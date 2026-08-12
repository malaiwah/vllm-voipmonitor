# MSRT Additive Cartridge: Feasibility Analysis and Implementation Plan

## Executive Summary

MSRT (Multi-Stage Rescaled Trellis) is **directly implementable** as an additive
cartridge on top of the existing EXL3 runtime in vLLM's GG branch. The key
insight: MSRT is **separable in output space** — each trellis stage's
contribution to the output is an independent GEMM that can be summed. This maps
exactly to a LoRA-like pattern: base weight GEMM + one or more correction GEMMs.

No new CUDA kernels are required. The existing `exl3_gemm` / `b12x_trellis_linear`
kernels handle each stage independently. The cartridge is pure orchestration:
load N trellis tensors per weight, run N GEMMs, sum the outputs.

---

## 1. Feasibility Verification

### 1.1 MSRT is Output-Separable

MSRT reconstructs a weight $W$ as:

$$W \approx W_0 + \frac{1}{s_1} Q_1(s_1 \cdot R_1) + \frac{1}{s_2} Q_2(s_2 \cdot R_2) + \cdots$$

where $W_0 = Q_K(W_\text{reg})$ is the base trellis quantization, $R_i = W - W_{i-1}$
is the residual, $s_i = |c_{bs}| / \sigma(R_i)$ is the rescaling factor, and
$Q_i$ is trellis quantization at $K_i$ bits.

The forward pass computes $Y = XW$:

$$Y \approx X W_0 + \frac{1}{s_1} X Q_1(s_1 R_1) + \frac{1}{s_2} X Q_2(s_2 R_2) + \cdots$$

Each term is an independent GEMM: $X$ times a dequantized trellis weight.
This is the **fundamental separability** that makes the cartridge feasible.

### 1.2 Existing Infrastructure

The GG branch (`lilab/codex/gg-exl3-r7-k345-20260810`) already has:

| Component | File | Role |
|-----------|------|------|
| EXL3 quant config | `vllm/model_executor/layers/quantization/exl3.py:952` | `Exl3Config` — parses checkpoint metadata, supports R7 mixed-bitrate |
| Dense linear method | `exl3.py:1917` | `Exl3LinearMethod` — loads trellis+suh+svh, runs `exl3_gemm` |
| Online linear method | `exl3.py:1701` | `Exl3OnlineLinearMethod` — quantizes BF16→trellis at load time |
| MoE method | `exl3.py:2454` | `Exl3MoEMethod` — per-expert GEMM, R7 fused, rank-sliced |
| Fused dequant+GEMM | `exl3_gemm_inner.cuh` | CUDA kernel: loads trellis int16, dequants via codebook, matmuls |
| Dequant primitives | `exl3_dq.cuh` | `dq<half>`, `dq2<half2>`, `dq4`, `dq8` — codebook lookup from packed bits |
| B12X mixed trellis | `b12x.moe._shared.kernels.w4a16.mixed_trellis` | Multi-tier mixed-bitrate MoE kernel |
| Online cache | `exl3_online_cache.py` | Persistent cache for online-quantized weights |
| R7 tier system | `exl3.py:3662` | `_r7_projection_tiers` — splits experts into K3/K4/K5 tiers |

### 1.3 What MSRT Needs That Doesn't Exist Yet

| Need | Status | Complexity |
|------|--------|------------|
| Multiple trellis tensors per weight | Missing — current format stores 1 trellis per (expert, projection) | Medium: extend checkpoint format + weight loader |
| Sum of multiple GEMM outputs | Missing — current `apply()` runs 1 GEMM | Low: add correction GEMMs and accumulate |
| Rescaling factors per stage | Missing — current format has no per-stage scale | Low: store as float32 scalars in metadata |
| Cartridge loading (optional, per-expert) | Missing — no concept of "add-on" weights | Medium: new loading path, like LoRA |

### 1.4 Memory and Performance Analysis

**Memory (per expert, gate_proj 2048×6144):**

| Stage | K bits | Trellis size | + suh/svh | Total |
|-------|--------|-------------|-----------|-------|
| Base K2 | 2 | 2048×384×2B = 1.5 MB | 48 KB | 1.55 MB |
| Residual K1 | 1 | 2048×192×2B = 0.75 MB | 24 KB | 0.77 MB |
| Residual K3 | 3 | 2048×576×2B = 2.25 MB | 48 KB | 2.30 MB |

Total MSRT 6bpw (K2+K1+K3): **4.62 MB** vs single K6: **4.69 MB** — same footprint.

The cartridge (residual stages only) is **3.07 MB** per expert — 66% of total.
At 256 experts × 75 MoE layers × 3 projections: cartridge = ~168 GB. But this
**is the model** — there's no separate "base vs cartridge" memory cost. The
total is identical to any other 6bpw format.

**Runtime performance (per expert, per token):**

Each trellis GEMM is a fused dequant+matmul. MSRT at 6bpw requires 3 GEMM
passes instead of 1:

| Method | Passes | Kernel | GEMM volume |
|--------|--------|--------|-------------|
| Single K6 | 1 | `exl3_gemm` | M×K×N (full) |
| MSRT K2+K1+K3 | 3 | `exl3_gemm` × 3 | M×K×N (same, 3× overhead) |

The 3× GEMM count is the runtime cost. For decode (M=1-128), each GEMM is
memory-bound on the weight tensor, so 3 passes = 3× weight bandwidth. For a
6bpw model where weights dominate bandwidth, this means ~3× decode latency
vs single-tier K6.

**Mitigation:** The residual stages are smaller (K1 = 1 bit = half the bandwidth
of K2). Realistic overhead at 6bpw:
- K2 base: 2 bits × full bandwidth
- K1 residual: 1 bit × full bandwidth
- K3 residual: 3 bits × full bandwidth
- Total bandwidth: 6 bits (same as K6), but 3 kernel launches instead of 1

The overhead is **kernel launch latency × (N_stages − 1)**, not bandwidth.
At 3 stages, this is 2 extra launches per expert per token-batch. For decode
with ~8 experts active per token and batch=128, that's 16 extra microsecond-
scale launches — negligible vs the GEMM itself.

For **prefill** (large M), the GEMMs are compute-bound, and 3 passes at the same
total bitrate have the same total FLOPs. The overhead is again just launches.

### 1.5 Cartridge Concept (LoRA-like)

The user's vision: load base K2 or K3 weights flat, then load an additive
cartridge for additional accuracy.

**This is exactly how MSRT decomposes:**
- **Base:** K2 trellis (2 bpw) — always loaded, standard EXL3 checkpoint
- **Cartridge:** K1+K3 residual trellis stages (4 bpw) — optional add-on

The cartridge contains:
1. Residual trellis tensors (int16, same format as base)
2. Per-stage rescaling factors (float32 scalars, ~4 bytes per stage per expert)
3. suh/svh rotation vectors for each residual stage (same format as base)

The cartridge is **not LoRA** (low-rank) — it's **full-rank additive trellis**.
But the loading and application pattern is identical:
- Base weight: standard `Exl3MoEMethod.apply()` or `Exl3LinearMethod.apply()`
- Cartridge: additional trellis tensors, loaded separately, applied as
  correction GEMMs summed with the base output

### 1.6 Compatibility with R7 Mixed-Bitrate

The existing R7 system already supports per-expert, per-projection tier
assignments (K3/K4/K5). MSRT is orthogonal: R7 assigns different experts to
different single-tier bitrates; MSRT uses multiple tiers per expert. They can
compose:

- R7 tier assignment decides base bitrate per expert (e.g., important experts
  get K3 base, others get K2 base)
- MSRT cartridge adds residual stages on top of each expert's base
- The cartridge can be per-expert selective (only add residuals to experts that
  need them — though v48 showed CV=0.11%, so all experts benefit equally)

---

## 2. Codebase Orientation (Quick Reference)

### 2.1 Key Files

```
vllm/model_executor/layers/quantization/exl3.py          (4866 lines)
  ├── Exl3Config                     :952   — checkpoint metadata, R7 config
  ├── Exl3OnlineLinearMethod        :1701   — online BF16→trellis (dense)
  ├── Exl3LinearMethod              :1917   — pre-quantized trellis (dense)
  ├── Exl3MoEMethod                 :2454   — MoE path (per-expert, R7, rank-sliced)
  │   ├── create_weights            :2461   — allocate trellis/suh/svh per expert
  │   ├── process_weights_after_loading :2632 — validate, shard, prepare B12X
  │   ├── apply                      :4695   — dispatch: rank_sliced / R7 fused / R7 graph / monolithic
  │   ├── _apply_expert              :4777   — single expert: exl3_gemm(trellis, suh, svh)
  │   ├── _apply_r7_graph            :4211   — R7 CUDA-graph fused MoE
  │   └── _r7_projection_tiers       :3662   — split experts into K3/K4/K5 tiers
  └── warmup_exl3_mixed_trellis_route_pack :4813

vllm/model_executor/layers/quantization/exl3_online_cache.py  (382 lines)
  ├── Exl3OnlineCacheKey             :42    — cache identity (model, encoder, prefix, bits, seed, TP)
  └── load_or_quantize               :319   — load cached or encode fresh

exllamav3_ext/quant/
  ├── exl3_gemm.cu                          — Python bindings for exl3_gemm
  ├── exl3_gemm_inner.cuh                   — fused dequant+GEMM kernel template
  ├── exl3_dq.cuh                            — codebook dequant: dq, dq2, dq4, dq8
  └── codebook.cuh                           — codebook tables (mcg, mul1)

b12x/moe/_shared/kernels/w4a16/
  ├── mixed_trellis.py                      — build_tiered_maps, compile_mixed_trellis, run_mixed_trellis
  └── prepare.py                            — prepare_trellis256_moe_weights
```

### 2.2 Weight Storage Format

Each expert weight matrix is stored as:
- `trellis`: int16 tensor of shape `(K//16, N//16, 16*16)` — packed trellis indices
  - Actual shape: `(rows//16, cols//16, K*16)` where K*16 = bits per tile
  - For K2: shape[2] = 32; K3: 48; K4: 64; K5: 80; K6: 96
- `suh`: float16 tensor — row-side Hadamard scales (per 128-row block)
- `svh`: float16 tensor — column-side Hadamard scales (per 128-col block)
- `mcg` or `mul1`: int32 scalar sentinel — codebook selection

R7 checkpoint adds:
- `r7_routed_experts` in `quantization_config.json` with schema `r7-complete-v2-checkpoint-v1`
- Per-expert K values (3, 4, or 5) encoded in trellis tensor shape
- Tier maps: which experts are K3 vs K4 vs K5

### 2.3 Runtime GEMM Flow

```
Exl3MoEMethod.apply(layer, x, topk_weights, topk_ids)
  → dispatch based on layer attributes:
    → _apply_expert(layer, group, x, expert_id, shard_id)
        → _exl3_gemm(x, trellis, suh, svh, mcg, mul1)
            → ext.exl3_gemm(x, trellis, output, suh, x_had, svh, -1, mcg, mul1, 0)
                → CUDA: load trellis tiles, dequant via dq<bits,cb>, matmul, apply suh/svh
    → _apply_r7_graph(layer, x, topk_weights, topk_ids)
        → ext.exl3_moe_r7_fused(xh, out32, topk_ids, topk_weights, ..., gate_bits, up_bits, down_bits, ...)
    → _apply_rank_sliced(layer, x, weights, ids)
        → b12x fused_moe API
```

### 2.4 Key Constants

- `_MCG_SENTINEL = 0xCBAC1FED` — mcg codebook marker
- `_MUL1_SENTINEL = 0x83DCD12D` — mul1 codebook marker
- `_HADAMARD_BLOCK = 128` — Hadamard block size
- `codebook_scale ≈ 1.2437` — trellis codebook scale (from `m.codebook_scale`)
- Trellis tile: 16×16 = 256 elements, packed into K bits each
- K values supported: 1-8 (EXL3 extension), 3-5 (R7), 3-8 (online)

---

## 3. Implementation Plan

### Phase 1: Checkpoint Format Extension (PoC)

**Goal:** Define an MSRT checkpoint format that stores multiple trellis stages
per weight, with per-stage rescaling factors.

**Format: `msrt-v1`**

```json
{
  "quantization_config": {
    "bits": "msrt",
    "version": "msrt-v1",
    "codebook": "mcg",
    "msrt_config": {
      "schema": "msrt-v1-checkpoint",
      "base_k": 2,
      "stages": [
        {"k": 1, "label": "res1"},
        {"k": 3, "label": "res2"}
      ],
      "moe_layers": [3, 77]
    }
  }
}
```

Per-expert weight files:
```
model.layers.10.mlp.experts.0.gate_proj.rank0.trellis_base     # K2 base
model.layers.10.mlp.experts.0.gate_proj.rank0.trellis_res1     # K1 residual
model.layers.10.mlp.experts.0.gate_proj.rank0.trellis_res2     # K3 residual
model.layers.10.mlp.experts.0.gate_proj.rank0.suh_base
model.layers.10.mlp.experts.0.gate_proj.rank0.svh_base
model.layers.10.mlp.experts.0.gate_proj.rank0.suh_res1
model.layers.10.mlp.experts.0.gate_proj.rank0.svh_res1
model.layers.10.mlp.experts.0.gate_proj.rank0.suh_res2
model.layers.10.mlp.experts.0.gate_proj.rank0.svh_res2
model.layers.10.mlp.experts.0.gate_proj.rank0.scale_res1        # float32 scalar
model.layers.10.mlp.experts.0.gate_proj.rank0.scale_res2        # float32 scalar
```

**Files to modify:**
- `exl3.py:Exl3Config.from_config()` — parse `msrt_config` block
- `exl3.py:Exl3Config.__init__()` — store MSRT metadata
- New: `exl3.py:Exl3MsrtMethod` class (or extend `Exl3MoEMethod`)

### Phase 2: Weight Loading

**Goal:** Load multiple trellis stages per expert and store them as side
dictionaries (same pattern as existing `exl3_tensors`).

**Approach:** Extend `Exl3MoEMethod.create_weights()` to allocate:
- `layer.w13_trellis.exl3_tensors_base[(expert, shard)]` — base trellis
- `layer.w13_trellis.exl3_tensors_res[(stage, expert, shard)]` — residual trellis
- `layer.w13_scales.exl3_tensors[(stage, expert, shard)]` — rescaling factors

Or simpler: use a flat dict keyed by `(stage, expert, shard)` where stage 0
is the base.

**Files to modify:**
- `exl3.py:Exl3MoEMethod.create_weights()` — allocate multi-stage storage
- `exl3.py:Exl3MoEMethod.process_weights_after_loading()` — validate all stages
- `exl3.py:Exl3Parameter` — extend for multi-stage tensor dict
- `exl3.py:_exl3_weight_loader()` — route weight names to correct stage

### Phase 3: Runtime Apply (GEMM Summation)

**Goal:** Run N trellis GEMMs per expert and sum outputs with rescaling.

**Dense layers (`Exl3LinearMethod.apply()`):**
```python
def apply(self, layer, x, bias=None):
    x_2d = x.reshape(-1, x.shape[-1]).to(torch.float16).contiguous()
    # Base GEMM
    output = self._apply_one(layer, x_2d, shard_id, stage=0)
    # Residual GEMMs
    for stage in range(1, self.n_stages):
        res_output = self._apply_one(layer, x_2d, shard_id, stage=stage)
        scale = layer.scales.exl3_tensors[(stage, shard_id)]
        output = output + res_output * (1.0 / scale)
    if bias is not None:
        output = output + bias
    return output.reshape(*x.shape[:-1], -1)
```

**MoE layers (`Exl3MoEMethod._apply_expert()`):**
```python
@staticmethod
def _apply_expert(layer, group, x, expert_id, shard_id):
    key = (expert_id, shard_id)
    # Base GEMM
    trellis = getattr(layer, f"{group}_trellis").exl3_tensors[(0, *key)]
    output = _exl3_gemm(x, trellis, suh_base, svh_base, mcg, False)
    # Residual GEMMs
    for stage in range(1, layer.exl3_msrt_n_stages):
        trellis_res = getattr(layer, f"{group}_trellis").exl3_tensors[(stage, *key)]
        suh_res = getattr(layer, f"{group}_suh").exl3_tensors[(stage, *key)]
        svh_res = getattr(layer, f"{group}_svh").exl3_tensors[(stage, *key)]
        scale = getattr(layer, f"{group}_scales").exl3_tensors[(stage, *key)]
        res_output = _exl3_gemm(x, trellis_res, suh_res, svh_res, mcg, False)
        output = output + res_output * (1.0 / scale)
    return output[..., :logical_n]
```

**Key insight:** `_exl3_gemm` is already a `torch.library.custom_op` that
works under CUDA graph capture. Calling it multiple times and summing is
graph-safe — no new CUDA code needed.

**Files to modify:**
- `exl3.py:Exl3LinearMethod.apply()` — add residual GEMM loop
- `exl3.py:Exl3MoEMethod._apply_expert()` — add residual GEMM loop
- `exl3.py:Exl3MoEMethod._apply_r7_graph()` — extend R7 fused kernel call
  (this is harder — the R7 fused kernel does all experts in one launch;
   MSRT corrections would need either a second fused launch or a per-expert
   fallback. Start with per-expert path, optimize later.)

### Phase 4: Cartridge Loading (LoRA-like)

**Goal:** Allow loading MSRT residual stages as a separate "cartridge" on top
of a base EXL3 model.

**Pattern:** Similar to vLLM's LoRA loading:
1. Base model loads with standard EXL3 (K2 or K3, single trellis per weight)
2. Cartridge loads as a set of additional trellis tensors + scales
3. At runtime, base GEMM + cartridge GEMMs are summed

**Implementation:**
- New: `Exl3MsrtCartridge` class — holds residual trellis tensors and scales
- `Exl3MoEMethod` checks for attached cartridge in `apply()`
- Cartridge can be loaded/unloaded dynamically (like LoRA adapters)
- Cartridge can be per-expert selective (skip experts that don't need it)

**Files to create:**
- `vllm/model_executor/layers/quantization/exl3_msrt_cartridge.py`

### Phase 5: B12X Integration (Optimized Path)

**Goal:** Integrate MSRT with the B12X mixed-trellis kernel for fused
multi-stage execution.

**Current B12X mixed_trellis:** Runs multiple bitrate tiers (K3/K4/K5) for
different experts in one kernel launch. Each tier is a separate trellis tensor.

**MSRT extension:** Each expert has multiple trellis tensors (stages) at the
same effective bitrate. The B12X kernel would need to:
1. Load each stage's trellis tile
2. Dequant independently
3. Apply rescaling
4. Sum before the matmul (or sum outputs after)

The simplest integration: treat each MSRT stage as a separate "tier" in the
B12X mixed_trellis API. The kernel already handles multiple trellis tensors
per expert — we just need to sum their dequantized outputs with rescaling
before the GEMM accumulation.

**This is the performance-critical path.** For production, the B12X kernel
would need a modification to:
- Accept N trellis tensors per expert (not just 1 per tier)
- Apply per-stage rescaling factors
- Sum dequantized weights before matmul

**Estimated kernel modification:** ~200 lines of CUDA in `mixed_trellis.cu`,
adding a loop over stages in the tile dequant section.

### Phase 6: Encoding Pipeline

**Goal:** Create MSRT-encoded checkpoints from BF16 weights.

**Script:** `tools/msrt_encode.py`
1. Load BF16 weights
2. Apply Hadamard regularization (seed=0, block=128)
3. Quantize base tier (K2) with EXL3 trellis
4. Compute residual, rescale, quantize (K1)
5. Compute residual, rescale, quantize (K3)
6. Save all stages + scales in `msrt-v1` format

This is exactly what the PoC scripts already do — just needs packaging as
a checkpoint converter.

---

## 4. Implementation Priority

| Phase | Effort | Impact | Priority |
|-------|--------|--------|----------|
| 1: Checkpoint format | Low | Enables everything | P0 |
| 2: Weight loading | Medium | Loads MSRT checkpoints | P0 |
| 3: Runtime apply (per-expert) | Low | Correctness proof | P0 |
| 4: Cartridge loading | Medium | User's LoRA-like vision | P1 |
| 5: B12X integration | High | Production performance | P2 |
| 6: Encoding pipeline | Low | Usability | P1 |

**Minimum viable path:** Phases 1-3 give a working MSRT runtime using the
existing per-expert GEMM path. This is correct but not optimally fast (3×
kernel launches vs 1). Phase 5 (B12X) brings it to production speed.

---

## 5. Risk Assessment

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| 3× GEMM latency too slow for decode | Medium | B12X fused kernel (Phase 5) reduces to ~1× |
| CUDA graph capture with multi-stage GEMMs | Low | `_exl3_gemm` is already a custom_op; summing outputs is graph-safe |
| Checkpoint size larger than single-tier | None | Same total bits per weight (K2+K1+K3 = 6 bits = K6) |
| Rescaling factor precision | Low | float32 scalars are sufficient (PoC confirmed) |
| R7 tier system conflict | Low | MSRT is orthogonal to R7 — can compose |
| B12X API doesn't support multi-stage | Medium | Fall back to per-expert GEMM path (Phase 3) |

---

## 6. Conclusion

MSRT is **directly implementable** on the GG branch with zero new CUDA code
for the initial path. The existing `exl3_gemm` kernel handles each trellis
stage independently — MSRT just calls it N times and sums. The cartridge
concept maps naturally: base trellis = standard checkpoint, residual trellis
stages = add-on cartridge.

The production path (B12X fused multi-stage) requires ~200 lines of CUDA
modification to the mixed_trellis kernel, fusing the N dequant passes into
one kernel launch. This eliminates the 3× launch overhead and makes MSRT
runtime-equivalent to single-tier EXL3.
