# MIXED-K Progressive Tensors first boot — Fruit proxy (M0 gate)

Date: 2026-08-10. GPUs 0-3 (RTX PRO 6000 Blackwell), GG v20-r33 rootfs env, TP4.

**Outcome: PASS.** The first mixed K3/K4 checkpoint assembled from Progressive
Tensors segments + the solve-derived policy boots under the GG vLLM fork,
loads layer-dependent mixed expert partitions exactly per policy, and
generates coherent text at the same throughput as the pure-K3 reference.

## 1. What was built

- `fq_assemble.py` extended with a **mixed-size reindex path** (auto-engaged
  when any expert tensor's segment size differs from its source slot; trellis
  is `[in/16, out/16, 16*K]` i16 so K4 rows are 4/3 the bytes of K3):
  source header TENSOR ORDER kept, non-expert tensors byte-exact from the
  source shard, header rebuilt with dtype/shape from the SEGMENT header for
  expert tensors and fresh contiguous data_offsets. The uniform-K
  byte-identity template path is unchanged (all previous tests still green;
  17 tests pass).
- Assembled checkpoint: `/home/mbelleau/fq-0c/fruit-mixed-042` (3.7 GB;
  between pure K3 3.5 GB and pure K4 3.9 GB) from
  segments `/home/mbelleau/fq-0c/fruit-segments`,
  policy `/home/mbelleau/fq-0c/policy-fruit-mixed-042.json` (fq-policy/2,
  n_k4_per_layer 42..152 over layers 3-12),
  source `/home/mbelleau/fq-0c/fruit-k3`.
  Byte-level spot checks: expert tensors identical to their policy-K segment
  bytes, non-expert tensors identical to the source shard.

## 2. What metadata a mixed checkpoint needs (GG loader contract)

Derived from `exl3.py` (`_configure_rank_sliced`, `_load_rank_sliced_bitrates`,
gg-v20-r33) — the pure K3/K4 configs differ only in `hybrid_tr3_tail.bits`
(3.0 vs 4.0); mixed needs three things:

1. **`config.json` → `hybrid_tr3_tail`**:
   - `"bits": "mixed"` (string),
   - `"k_values": [3, 4]` (must be within 3..6),
   - `"bits_per_expert": "tier_bitmap.json:bits_per_expert"` — a
     `"file.json:field"` reference (string), NOT inline data.
2. **The referenced JSON** (we use `tier_bitmap.json`): maps
   `str(layer)` → dict whose `bits_per_expert` field is a 256-int list
   (one bitrate per expert, values ⊆ k_values), for every layer in
   `moe_layers=[3,12]`. MTP layer 13 stores experts as plain bf16
   `.weight` tensors outside the rank-sliced range — no entry needed.
   (Loader quirk: an entry with a 256-long `tail_tr3` field and no
   bitrate field defaults to all-K3 — the big-model MTP convention.)
3. **`config.json` → `quantization_config` stub**
   (`{"quant_method": "exl3", "bits": "mixed", "codebook": "mcg", ...}`):
   vLLM's `weight_utils.get_quant_config` resolves the quant class from
   `hf_config.quantization_config` BEFORE `Exl3Config.maybe_update_config`
   ever reads `hybrid_tr3_tail`; without it (and without a
   `quantization_config.json` file, which the rank-sliced format does not
   ship) boot dies with "Cannot find the config file for exl3". The pure
   fruit-k3/k4 checkpoints have the same gap — worked around at serve time
   via `--hf-overrides '{"quantization_config": {...}}'`.

Plus regenerated `model.safetensors.index.json` (new shard sizes/offsets) and
`MANIFEST.sha256`. All of this is emitted by `fq_assemble.py` automatically
when the policy uses >1 K value.

The loader accepted the metadata first try and planned the exact
runtime-dynamic partitions from the policy, e.g.:

```
EXL3 mixed Trellis model.layers.12.mlp.experts: tiers=((3, 115), (4, 141))
EXL3 mixed Trellis prefill block policy: ... shape=1024x128 tiers=((3, 214), (4, 42)) topk=8
EXL3 mixed Trellis runtime planned: tiers=((3, 185), (4, 71)) one-grid decode=32 one-grid prefill=8192/8192 block_m=64
```

Per-layer K4 counts observed at load == policy exactly:
{3:71, 4:42, 5:79, 6:103, 7:125, 8:113, 9:106, 10:143, 11:152, 12:141}.

## 3. Serve configuration (the one non-obvious bug)

Command (via `serve-fruit.sh`, wraps `gg-run.sh`):

```
CUDA_VISIBLE_DEVICES=0,1,2,3 gg-run.sh python -m vllm.entrypoints.openai.api_server \
  --model /home/mbelleau/fq-0c/fruit-mixed-042 --served-model-name fruit-mixed \
  --host 127.0.0.1 --port 8801 --trust-remote-code --tensor-parallel-size 4 \
  --quantization exl3 --attention-backend B12X_MLA_SPARSE --moe-backend b12x \
  --max-model-len 4096 --max-num-seqs 4 --gpu-memory-utilization 0.30 \
  --kv-cache-dtype fp8_ds_mla --hf-overrides '{"use_index_cache":true}'
```

(env: VLLM_WORKER_MULTIPROC_METHOD=spawn, VLLM_USE_B12X_MOE=1,
VLLM_USE_B12X_SPARSE_INDEXER=1, PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True.
CUDA graphs on — FULL_AND_PIECEWISE captured fine; --enforce-eager never needed.
No index_topk_pattern override needed: Fruit's config already carries the
13-layer all-"F" pattern.)

**Critical finding — `--kv-cache-dtype fp8_ds_mla` is required.** With the
default (auto → bf16, "B12X_NON_COMPRESSED_INDEXER" cache layout), BOTH the
mixed and the pure-K3 checkpoints boot cleanly and then emit prompt-INDEPENDENT
degenerate text (identical continuation regardless of prompt — attention
contributes nothing). The big GLM-5.2 serve always ran B12X_MLA_SPARSE with a
ds_mla compressed cache (`nvfp4_ds_mla`); the sparse-MLA kernel stack is only
correct with a ds_mla KV layout. This was diagnosed by the planned pure-K3
sanity boot: pure K3 garbled identically → serve config, not the mixing.
With `fp8_ds_mla` both serve coherently. Model load: 1.09 GiB/GPU, ~3.6 s.

## 4. Coherence + throughput (greedy, temperature 0)

| Prompt | fruit-mixed-042 | fruit-k3 (reference) |
|---|---|---|
| "Once upon a time, there was a small robot who" | " lived in a small house. He was very lonely, so he decided to go for a walk. He walked and walked until he found a big tree. He sat down under the tree and look…" | identical text |
| "The quick brown fox" | " jumps over the lazy dog. The fox is very scared and runs away. The fox is very fast and very clever…" | " jumps over the lazy dog. The dog is very scared and runs away. The fox is very happy and runs away…" |
| "To make a cup of tea, first you" | " need to boil the water. You can do this by adding a little salt and stirring it…" | " need to add the tea leaves. You can use a spoon or a spoon to stir the tea leaves…" |
| "1, 2, 3, 4, 5," | " 6, 7, 8, 9, 10, 11, 12, …" (exact) | identical |

Quality is proxy-level (5.04B assistant-masked SFT; simple, repetitive prose)
but fully grammatical, prompt-dependent, and on-topic for both. Mixed vs K3
divergence is small and consistent with slightly different expert precision.

Throughput (single request, 512-token greedy decode, batch 1):
- fruit-mixed-042: **501.6 tok/s** (short 80-100 tok runs: 400-424 tok/s)
- fruit-k3: **503.1 tok/s** (short runs: 401-416 tok/s)

→ Mixed-tier execution costs ~0% decode throughput on this proxy.

## 5. Files / commits

- Assembler + tests: `research/fungible-quant/tools/fq_assemble.py`,
  `test_fq_assemble.py`, `test_fq_repack.py` — commit `a6ff61d80`
  ("fq_assemble: mixed-K reindex path + GG loader mixed metadata (M0 gate)").
- Launcher: `research/fungible-quant/runs/serve-baseline/serve-fruit.sh`.
- Logs: `runs/serve-baseline/fruit-mixed.log`, `fruit-k3.log`,
  `fruit-bf16.log` (aborted BF16 reference: unquantized path fell into a
  ~20-min flashinfer fused-MoE autotune; not needed once K3-vs-mixed
  isolated the KV-cache-dtype bug).
- Checkpoint: `/home/mbelleau/fq-0c/fruit-mixed-042` (left serving on port
  8801, tmux `fq:serve`; the big GLM-5.2 serve on GPUs 0-3 was stopped per
  authorization and can be restarted from warm caches).

## 6. Open notes

- The `quantization_config` stub belongs in the encode/convert pipeline too
  (pure fruit-k3/k4 need the hf-override workaround until regenerated).
- `bits_per_expert` values here are full-precision members {3,4}; the loader
  validates k_values ⊆ 3..6, so K5/K6 tiers need no new metadata shape.
- BF16 reference boot (quality ceiling) still pending if ever needed —
  requires sitting through flashinfer MoE autotune or a different
  moe-backend; not required for the M0 gate.
