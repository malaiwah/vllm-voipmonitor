# capture_stream: layer-streaming BF16 activation capture — design + validation report

Date: 2026-08-10 · Author: Claude (fungible-quant 0c campaign)
Tool: `tools/fruit-encoder/capture_stream.py` · Driver: `runs/0c-campaign/run-capture-glm52.sh`

## What this replaces

The planned serving-engine (vLLM) capture of full GLM-5.2 is pre-empted by a
layer-major streaming pipeline: the 78-layer model never resides in memory at
once, so the full BF16 checkpoint captures on **one 96GB GPU per shard** with a
~35GB peak GPU footprint (one layer's weights + one sample's activations), and
— decisively — the output is **bit-identical** to the transformers reference
forward, which a serving engine's continuous batching can never be (see
"Why exact mode", below).

## Design

Layer-major streaming over the sealed calibration plan (same
sample-selection/order as all prior captures):

1. **Embed stage** — tokenizer + `embed_tokens` weights only; all planned
   samples are embedded in exact plan order into a bf16 *boundary* file
   (`boundary_000.bin` = input to layer 0).
2. **Per-layer sweep** — for layer L: materialize ONLY layer L's weights on
   GPU, stream boundary L through it one sample at a time, append boundary
   L+1. The input boundary is deleted once the output is sealed (peak: 2
   boundaries on disk). At MoE layers in the capture window, a
   `forward_pre_hook` on `mlp.experts` records `x.bin`/`ids.bin` with
   capture_hf.py's exact fp32 gate math, cross-checked against the model's own
   top-k on every token.
3. **DSA cross-layer top-k sharing** — GLM-5.2's `indexer_types` mixes
   `full`/`shared` layers; shared layers reuse the previous full layer's
   indexer top-k. When layer L is full and L+1 shared, the per-sample
   `topk_indices` are persisted to a *topk store* and fed as
   `prev_topk_indices` to the dependent shared layers. Stores whose consumers
   extend past the window's stop layer are preserved for the next window
   (layer 10's store feeds layers 11–13).
4. **Sharding** — 2 contiguous corpus shards (GPU 6 / GPU 7), split at sample
   boundaries by a cost model (`tokens + 2000/sample`, because each per-sample
   MoE call pays a full ~19.3GB expert-weight HBM read). Shards `pwrite`
   disjoint token ranges of the shared `x.bin.partial`/`ids.bin.partial`; a
   `--seal` step merges shard markers, verifies token/routed-count audits,
   sha256s the payloads, and writes the ABI manifests.
5. **Resume** — every artifact (boundary, topk store, per-layer capture) has a
   sealed/consumed marker; rerunning any shard skips completed work and
   restarts an interrupted layer idempotently (region rewrites at absolute
   offsets). `state.json` at the capture root is refreshed ≤2-minutely with
   per-shard stage/layer/pack progress and the preserved-boundary contract.

### Forward fidelity (why the reference implementation is used verbatim)

- Model skeleton instantiated on the **meta device** from the checkpoint
  config (`AutoModelForCausalLM.from_config`, `attn=sdpa`,
  `experts=grouped_mm` — the sealed reference's exact configuration); one
  `GlmMoeDsaDecoderLayer` at a time is materialized with
  `load_state_dict(assign=True)` from the safetensors shards. accelerate is
  not present in the gg env, so `torch.device("meta")` is used directly —
  functionally identical to `init_empty_weights`.
- Per-expert `gate/up/down` checkpoint weights are fused into
  `gate_up_proj`/`down_proj` exactly as transformers' qwen2_moe weight
  converter does (stack dim 0, concat gate|up dim 1).
- dtype rules replicate `from_pretrained(dtype=bf16)`:
  `e_score_correction_bias` → fp32 (`_keep_in_fp32_modules_strict`);
  everything else bf16 (`_keep_in_fp32_modules` = fp16-only in transformers
  5.14, so `indexer.weights_proj` stays bf16).
- The decoder layer forward is decomposed into the module's own submodule
  calls with the residual adds mirrored verbatim from
  `GlmMoeDsaDecoderLayer.forward`; attention masks come from
  `create_causal_mask` exactly as `GlmMoeDsaModel.forward` builds them
  (`None` for sdpa batch=1 — same path as the reference).
- **Rotary gotcha (found by the gate):** `inv_freq` must be computed on CPU
  and moved to GPU, as `from_pretrained` does. CUDA `pow()` differs from CPU
  by 1 ulp on 3/32 exponents at rope_theta=5e5; that alone perturbed cos/sin
  enough to flip ~1.4% of routings per layer and compound into an 88% id
  match by layer 12. With CPU-init rotary the pipeline is bit-exact.

### Why exact mode (batch=1 per sample, one sample per MoE call)

Two batching optimizations were built and measured, and both **fail the
gate** — not because the math is wrong but because this GPU's kernels are not
row-stable across batch shapes:

- Packed multi-sample batches (8192-token budget, block-diagonal masks,
  per-sample position ids): semantically correct (selftest passes at
  tolerance), but sdpa/GEMM reduction-order noise (~1 ulp) flips ~2% of
  near-boundary routings per layer; flipped routing = different experts =
  large hidden-state changes that compound: ids match 97.8% (layer 3) → 81.6%
  (layer 12) vs the sealed reference.
- Grouped MoE across samples: directly measured NOT row-stable —
  `grouped_mm` and even plain cublas GEMMs (incl. the fp32 router) return
  different bits for the same row when the batch composition changes.

Exact mode reproduces the sealed reference's shapes exactly and is fast
anyway (see timings): the feared per-sample expert-weight read is HBM-side
(~11ms/sample on GLM-5.2), not disk-side.

## Validation

### Selftest (`--selftest`, small random GlmMoeDsa with full+shared indexer layers)

- Mode A (per-sample): streaming pipeline output — all 8 boundaries, all
  captured x and ids — **bitwise equal** to a `from_pretrained` full-model
  forward. This covers materialization, expert fusion, dtype rules, boundary
  IO, topk-store write/read across shared layers, and the capture hook.
- Mode B (packed+grouped): within tolerance only (documented smoke bound).

### VALIDATION GATE — Fruit snapshot vs SEALED reference (`/home/mbelleau/fq-0c/capture`)

Full pipeline (2 shards on GPUs 6+7, exact mode, seal, compare) on the
sealed plan (fingerprint `c338b547…`, 1,050,468 tokens, 4,497 samples):

| layer | ids row match | x rows allclose (rtol .02, 4096 rows) | x rows bitwise | sha256(x.bin) == reference |
|-------|---------------|----------------------------------------|----------------|-----------------------------|
| 3–12 (all 10) | **100.0000%** | **100.0000%** | **100.00%** | **yes, every layer** |

- `routed_counts` identical on every layer; `sha256(ids.bin)`-level equality
  implied by 100% row match over all 1,050,468 rows per layer.
- Routing verify (fp32 recompute vs model top-k at every captured token):
  10,504,680 token-checks, **0 mismatches**.
- Seal spot-check: 64/64 exact top-8 per layer, 0 near-ties.
- The streaming capture is **byte-identical** to the whole-model reference
  capture — the gate (≥99.9% ids, x allclose) is passed with margin zero-loss.
- Wall clock: 121s (shard 0) / 102s (shard 1) for layers 0–12 including
  10 captured layers; seal+verify ≈ 45s.
- Report: `/home/mbelleau/fq-0c/stream-val/capture/compare_report.json`.

## GLM-5.2 run (window 1: layers 3–10)

- **Scope revision honored**: streaming-ring campaign on fixed 3TB disk —
  this pass captures ONE 8-layer window (3–10, ~103GB payload), and preserves
  the post-layer-10 boundary (`work/shard{0,1}/boundary_011.bin`, 12.9GB
  total) plus layer 10's DSA topk store as the next window's (11–18) input.
  The ring is: capture window → encode K2..K5 → publish → delete window's
  `layer_*` payloads → rerun driver with `LAYERS=11-18 STOP=18`.
- Plan: **regenerated** for GLM-5.2 (`/home/mbelleau/fq-0c/capture_plan_glm52.json`,
  fingerprint `c442aa4c…`) with capture_fruit's exact selection logic pointed
  at the GLM-5.2 snapshot: **identical sample selection and order** to the
  prior plan (verified line-by-line), source/tokenizer SHAs re-bound to
  `zai-org/GLM-5.2@b4734de4` (tokenizer.json is hash-identical to Fruit's).
- Capture dir `/home/mbelleau/glm52-capture` (disk, not shm); `state.json`
  refreshed ≤2-minutely; layer manifests schema `glm52-b300-layer-capture-v1`
  with `hidden: 6144`; run manifest documents the window/resume contract in
  its `layer_stream` section.
- Launched 2026-08-10 18:13 UTC in tmux window `fq:3 (capture)` via
  `run-capture-glm52.sh`, foreground piped to
  `runs/0c-campaign/capture-glm52.log`; shard 0 → GPU 6, shard 1 → GPU 7,
  seal runs automatically after both shards.
- **ETA (measured)**: dense layers 6–16s/shard; captured MoE layers
  **49–86s/shard** (compute ~30–50s + 19.3GB weight load, slower when both
  shards hit the disk simultaneously). Layers 0–4 completed in the first
  3.5 min. Projected window wall-clock **~25–40 min** total (launch 18:13
  UTC → sealed ≈ 18:40–18:55 UTC), including seal (8 × 12.9GB sha256 + gate
  spot-checks). The driver is idempotent — rerun it to resume after any
  interruption.

## Files

- Tool: `research/fungible-quant/tools/fruit-encoder/capture_stream.py`
- Driver: `research/fungible-quant/runs/0c-campaign/run-capture-glm52.sh`
- GLM-5.2 plan: `/home/mbelleau/fq-0c/capture_plan_glm52.json`
- Validation artifacts: `/home/mbelleau/fq-0c/stream-val/` (capture +
  compare_report.json + logs)
- GLM-5.2 window: `/home/mbelleau/glm52-capture/` (layer_003..layer_010,
  capture_run_manifest.json, state.json, work/shard{0,1})
