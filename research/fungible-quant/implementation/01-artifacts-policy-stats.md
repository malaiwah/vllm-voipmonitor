# 01 — Artifacts, policy store, stats collector, policy engine

## 1. On-disk artifacts

### 1.1 Weight artifact pair (produced offline, once per model)

Two complete routed-expert encodings from the same BF16 checkpoint + the same
calibration Hessians, produced by the existing exllamav3 convert pipeline:

```
<model>-EXL3-FQ/
  base-k3/            # every routed expert, K3  (~260 GB for GLM-5.2 class)
  overlay-k4/         # every routed expert, K4  (~347 GB)
  dense-bf16/         # dense/attention as today (online K6 at startup)
  fq-manifest.json    # schema below
```

Layout requirement: per-expert tensors must be individually addressable —
one file (or one safetensors shard entry) per
`(layer, expert, proj ∈ {w1,w2,w3})` at each K, so the swap engine can read
exactly one expert's 3.375/4.5 MiB without touching the rest. The existing
rank-sliced `tensor_schema` (`model.layers.{L}.mlp.experts.{E}.{proj}.rank{r}.*`)
already has this granularity; store **unsharded** (slice at load — trellis
tiles are self-contained at 16-column granularity) to keep D4 topology
neutrality.

`fq-manifest.json`:
```json
{
  "schema": "fq-manifest/1",
  "base_model": "zai-org/GLM-5.2",
  "revision": "<hf-rev>",
  "k_variants": [3, 4],
  "hessian_id": "<hash of calibration set + measure run>",
  "moe_layers": [3, 77],
  "num_experts": 256,
  "tensor_index": "index-k3.json / index-k4.json"
}
```

### 1.2 Policy artifact (the fungible part)

Extends the existing per-expert bitrate JSON the GG loader
(`_load_rank_sliced_bitrates`) already consumes — same shape, added envelope:

```json
{
  "schema": "fq-policy/1",
  "manifest": "<hash of fq-manifest.json>",
  "budget": { "mode": "fixed_cardinality", "n_k4_per_layer": {"3": 108, "...": 96} },
  "bits_per_expert": { "3": [3,4,3, ...256 ints], "...": [...] },
  "pinned": { "3": [17, 201], "78": "all" },
  "provenance": {
    "created": "...", "windows_observed": 0,
    "parent": "<hash of previous policy or 'generic-v1'>",
    "workload_stats_hash": "..."
  }
}
```

Rules: keyed by **logical expert id**; NO rank/world_size/tp/device fields
(D4). `pinned` entries are excluded from downgrade by the policy engine.
`n_k4_per_layer` is fixed at startup (D1) — a policy whose counts differ from
the running cardinality is applied by permutation only after projecting onto
the running counts (take its top-N_L per layer by its own ordering).

### 1.3 Caches (`VLLM_CACHE_ROOT/fq/`)

| Cache | Key | Contents |
|---|---|---|
| `policy/` | manifest hash | `current.json` + rolling `history/NNN.json` (last 16, for rollback/inspection) |
| `slabs/` | manifest hash + policy hash + **topology** (tp, dcp, layout tag) | pre-packed per-rank tier slabs for fast boot (optional; regenerable) |
| `stats/` | manifest hash | persisted window summaries (for provenance + cold-start reweighting) |

Policy cache is authoritative and tiny; slab cache is a boot-time optimization
(skip re-slicing 74 GiB/rank from NVMe artifacts) and always safe to delete.

## 2. Stats collector (`exl3_fungible/stats.py`)

Hook: `BaseRouter.set_capture_fn` — ungated, per-layer, fires on **logical**
ids after `_compute_routing`, before EPLB remap (GG branch:
`base_router.py:185`, `:296-300`; bound per-MoERunner at
`gpu_model_runner.py:7906-7919` — see `gg-integration-surface.md`).
**The slot is single-occupancy: the FQ collector must chain any previously
bound capture fn (call it after recording), not overwrite it** — the
mtp78-collector plugin family uses the same hook. GLM-5.2 routes through
`GroupedTopKRouter` (`use_grouped_topk=True`), which inherits the hook.

**Graph-safety contract (hard rules, tested in 03):** the callback runs INSIDE
captured CUDA graphs (MoE ops are not splitting ops). Therefore it must be:
pure tensor ops; no `.item()`, no host branches, no Python-object mutation, no
allocation. Persistent pre-allocated buffers only.

Per layer L, two persistent device buffers:
```
count_buf[L]  : int32  [E]   — scatter_add_(ones)      (token routings)
mass_buf[L]   : fp32   [E]   — scatter_add_(topk_weights.flatten())  (gate mass)
```
Both indexed by **logical** ids (capture fn sees pre-EPLB-remap ids; EP=1 so
logical==physical anyway). Cost per layer per step: two scatter_adds over
`T×8` elements — noise next to the MoE GEMM.

Windowing (host side, in `FungibleQuantState.step()`, called from the model
runner once per engine step exactly like EPLB's `step()`):
- every `window_stride` steps: copy accumulators to a circular window
  `W: [window_len, num_moe_layers, E]` (device→device), zero accumulators.
- decayed view for the policy: `w_e = Σ_i λ^i · W[-i]` with λ from Phase 0a.
- dummy/capture steps: zero accumulators, don't record (EPLB semantics).

Persistence: on every policy write, dump the decayed summary (few MB) to
`stats/` with the policy hash.

## 3. Policy engine (`exl3_fungible/policy.py`)

Pure host code, NumPy, runs OFF the hot path (background thread or the
rebalance tick). No GPU work.

### 3.1 Inputs

- `eps[L][e][k]` — static per-expert error at K3/K4 from the offline measure
  campaign (dKL deltas from `measure_model.py`), shipped in the manifest.
- `count`, `mass` — decayed window sums from the collector.
- Current membership `tier_of[L][e]`, dwell counters, pin set, budget.

### 3.2 Decision (per layer, O(E log E))

```
score[e] = (eps[L][e][3] - eps[L][e][4]) * mass[e]**beta * count[e]**alpha
desired  = top N_L experts by score (pinned-K4 forced in, pinned-K3 forced out)
proposal = desired  XOR  current      # symmetric difference → swap pairs
```

Guards applied to `proposal`, in order:
1. **Dwell**: drop any expert resident in its tier < `dwell_steps`.
2. **Hysteresis**: an entering expert must outscore the leaving expert it
   displaces by factor `h` (default 1.25 until 0a says otherwise).
3. **Cap**: keep at most `max_swaps_per_layer_per_interval` (default 2) pairs,
   highest score-gap first; at most `max_total_swaps_per_interval` model-wide.
4. **Emit** ordered swap list [(L, e_out, e_in), ...].

### 3.3 Verification loop (post-apply)

After a batch of swaps commits: run the held-out KL probe (fixed ~32-prompt
set, teacher-forced, greedy) and compare vs pre-swap probe score kept in
state. If ΔKL > `probe_regression_limit` → revert the batch (swap lists are
their own inverse) and mark the offending experts frozen for
`freeze_steps`. Probe runs on the serving engine between requests (it is just
a low-priority request batch); budget ~1-2 s.

**Router-shift guard** (cheaper, every interval): Jaccard overlap of top-8 ids
vs the previous window's on a fixed shadow prompt set; < `jaccard_floor`
(default 0.95, refine via 0e) → no swaps this interval (the workload itself is
shifting; wait for it to settle).

### 3.4 Cadence

Reuse EPLB's trigger pattern: `interval_steps` (default 3000, refine via 0a),
first decision at 25% of interval, counter advances on dummy steps for rank
lockstep. All ranks compute the SAME decision deterministically: inputs are
logical-domain and identical across ranks (EP=1) → no collective needed for
agreement; assert via a cheap hash all-reduce in debug mode only.

## 4. New env/config knobs (registered GG-style in envs.py)

| Knob | Default | Meaning |
|---|---|---|
| `VLLM_FQ_ENABLE` | 0 | master switch |
| `VLLM_FQ_ARTIFACT_DIR` | — | path to `<model>-EXL3-FQ/` |
| `VLLM_FQ_INTERVAL_STEPS` | 3000 | decision cadence |
| `VLLM_FQ_WINDOW` | 1000×stride | stats window |
| `VLLM_FQ_MAX_SWAPS_LAYER` | 2 | per-layer per-interval cap |
| `VLLM_FQ_MAX_SWAPS_TOTAL` | 64 | model-wide cap |
| `VLLM_FQ_DWELL_STEPS` | 2×interval | min tier residency |
| `VLLM_FQ_HYSTERESIS` | 1.25 | entry/exit score ratio |
| `VLLM_FQ_JACCARD_FLOOR` | 0.95 | router-shift guard |
| `VLLM_FQ_PROBE` | on | held-out KL probe + rollback |
| `VLLM_FQ_APPLY_MODE` | `atomic` | `atomic` (row swap) / `reload` (brutal path) / `dryrun` (decide+log only) |

`dryrun` is not a debug afterthought — it is **milestone M2's shipping mode**
(observe + decide + log, apply nothing) and permanently useful as a shadow
evaluator.

## 5. Multi-model note (MTP-78)

The draft layer is one more MoE layer to the collector and policy engine
(its capture fn binds the same way). Its objective differs (acceptance
length, not KL) — v1 pins MTP-78 out of the swap set (`pinned: {"78": "all"}`)
and only *records* its stats; a later milestone gives it its own probe
(MAL bench from the collector repo) and its own budget.

## 6. Knobs ← Phase 0 mapping

| Phase 0 result | Sets |
|---|---|
| 0a Kendall τ curve vs window gap | `INTERVAL_STEPS`, `DWELL_STEPS`, decay λ |
| 0c dKL variance per layer | `n_k4_per_layer` via global budget solve |
| 0d generic vs blended gap | whether ε gets a workload-blend refresh cycle |
| 0e Jaccard distribution | `JACCARD_FLOOR` |
| 0f(ii) encode benchmark | documentation only (D3 removed re-encode) |
