# Gate mass: why `mass` was a copy of `count`, and what now binds it

**Date:** 2026-08-11
**Repos:** `gg-vllm @ fq/m1-stats-collector`, `vllm-voipmonitor`
**Status:** real gate mass is recorded, **opt-in** via `VLLM_FQ_GATE_MASS=1`

---

## 1. The symptom

The live GLM-5.2 serve dump (`results/k3-fq/stats.jsonl`, 13 records,
75 layers x 256 experts) has `count` and `mass` byte-identical in every
record:

```
rec 1  count (75, 256)  mass (75, 256)  identical=True  max|c-m| = 0.0
rec 2  ... identical=True
rec 3  ... identical=True
```

The swap policy is specified to score on routing **mass** (Σ gate
weights), which ranks a confident route above a marginal one. It was
ranking on raw hit frequency.

## 2. Why the getter was never bound

Two independent reasons, and the second is the real one.

**(a) `bind_router` never passed a getter.**
`stats.py:bind_router` called
`self.make_capture_fn(layer_id, prev_fn=prev)` — the `topk_weights_getter`
parameter existed on `make_capture_fn` but had no path from `bind_router`,
and `integration.py:maybe_init_fq_collector` (the only live binder) calls
`bind_router`. So the `topk_weights_getter is None` branch was taken for
all 75 layers, `_mass_is_count` was populated for all of them, and
`decayed()` returned `c, c.clone()`.

**(b) Nothing held the top-k weights at capture time — they were not
reachable from where we bind.**
In this GG build the capture hook fires inside
`vllm/model_executor/layers/fused_moe/router/base_router.py`:

```
base_router.py:290   topk_weights, topk_ids = self._compute_routing(...)
base_router.py:296   if self.capture_fn is not None:
base_router.py:297       self.capture_fn(topk_ids)          # ids ONLY
```

`topk_weights` is a **local** in `BaseRouter._select_experts`. It is not
stored on the router, not on the `MoERunner`, and not on
`FusedMoERouter` (`fused_moe_router.py` keeps only `_routing_replay_out`,
which holds *ids* cast to int16). Every concrete router
(`GroupedTopKRouter`, `FusedTopKBiasRouter`, `ZeroExpertRouter`, …)
subclasses `BaseRouter` and implements `_compute_routing`; none of them
overrides `_select_experts`, and none caches the weights. So no getter
written against the router object could have returned this step's gate
weights. Fixing (a) alone would have produced zeros or, worse, a stale
tensor.

A stash-on-the-router scheme (wrap `_compute_routing`, write
`router._fq_last_weights`, read it in the capture fn) was rejected: under
dynamo tracing / CUDA-graph capture, a deferred attribute store can make
the capture fn read the *previous* step's tensor, which is silently wrong
data rather than a visible failure.

## 3. What was bound

**Router side (`base_router.py`, +26/-6).** A capture fn may opt in to
receiving the gate weights by carrying `wants_topk_weights = True`.
`set_capture_fn` resolves that once into `router.capture_fn_wants_weights`
(a plain bool, so the hot path does not `getattr` a callable), and
`_select_experts` calls `capture_fn(topk_ids, topk_weights)` for tagged
fns, `capture_fn(topk_ids)` otherwise. The existing one-argument
routed-experts capturer in `gpu_model_runner._bind_routed_experts_capturer`
is unaffected.

**Collector side (`stats.py`).**
`FqStatsCollector(..., record_mass=…)`, `bind_router(..., record_mass=…)`
and `make_capture_fn(..., record_mass=…)`. In mass mode the capture fn is
tagged `wants_topk_weights` and accumulates
`mass.scatter_add_(0, idx, topk_weights.flatten())`.

**Bind-time capability probe.** After `set_capture_fn`, `bind_router`
checks `router.capture_fn_wants_weights`. If the runtime's `BaseRouter`
predates the contract (e.g. the rootfs copy was not redeployed), it
**downgrades that layer to count-only and logs a warning** rather than
claiming mass it will never receive. This is what stops the fix from
re-becoming a silent alias.

**Wiring (`integration.py`).** `VLLM_FQ_GATE_MASS=1` sets
`record_mass=True`, and the hook logs the resolved mode at bind time:

```
FQ stats: bound 75 MoE routers, 256 experts, gate mass RECORDED (VLLM_FQ_GATE_MASS=1)
```

**Deploy (`deploy-fq.sh`).** `base_router.py` added to `HOOKS`. The
serve runs the rootfs copy, and that file was byte-identical to the
source tree before this change, so copying it is safe. **Without this,
gate mass cannot work at all** — and the probe would report it.

## 4. Explicit, detectable aliasing

`mass` and `count` both remain available in every artifact. The aliasing
is now stated, not inferred:

| surface | field |
|---|---|
| `FqStatsCollector.mass_is_real(layer_id=None)` | per layer, or model-wide (True only if *every* bound layer records real mass) |
| `collector.summary()` | `mass_is_real` at top level and per layer |
| `VLLM_FQ_DUMP_STATS` JSONL (`loop.py:_dump_stats`) | one new field: `"mass_is_real": true/false` |

This matters because equal arrays are not proof of aliasing: a uniform
router legitimately produces `mass ≈ count`. Reading the flag is the only
reliable test, and `loop.py`'s change is a single field (that file is
being edited concurrently).

## 5. Sentinel and out-of-range handling

The already-fixed `histc` subtlety is preserved and mirrored on the new
path. `torch.histc`'s last bin is **closed** at `max`, so binning `[0,E)`
as `bins=E/max=E` does not drop `id == E` — it folds the padding sentinel
into the last real expert. The count path bins `E+1/max=E+1` and slices.

The mass path cannot use `histc` (no weighted variant with a static
output shape), so it keeps the explicit **overflow slot** at index
`num_experts`, sized into the accumulators at bind time:

```python
ids = flat.to(torch.int64)
idx = torch.where(ids.lt(0), oor_slot, ids)   # negatives -> slot
idx.clamp_(max=num_experts)                   # E, E+1, garbage -> slot
mass.scatter_add_(0, idx, topk_weights.flatten().to(torch.float32))
```

Negatives are redirected *first*, then everything is clamped down — one
kernel cheaper than `where(ids.ge(0) & ids.lt(E), ids, E)`. A bare
`clamp(0, E)` would be wrong: it maps negatives onto real expert 0.
`idx ∈ [0, E]` unconditionally, so `scatter_add_` can never address past
the buffer (an OOR `scatter_add_` was the illegal memory access that
killed a live engine on the first M2 dryrun boot). Reads slice `[:E]`, so
the sentinel contributes to neither count nor mass — identical semantics
on both paths.

`oor_slot` is a persistent 0-dim tensor allocated at construction, so the
capture fn still allocates nothing and stays CUDA-graph-safe.

Counts use the **same `histc` in both modes**, so enabling mass cannot
perturb the count signal — an A/B across runs stays comparable.

## 6. Measured hot-path cost

`E=256`, `topk=8`, CPU, 1 thread, float32 weights, int32 ids,
20k iterations x 7 reps, best-of reported. GPUs were serving and were not
touched.

| batch | tokens | count-only | + gate mass | delta | ratio | aten ops off/on |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 8 | 7 582 ns | 14 529 ns | +6 947 ns | 1.92x | 6 / 12 |
| 8 | 64 | 7 725 ns | 14 693 ns | +6 969 ns | 1.90x | 6 / 12 |
| 64 | 512 | 9 310 ns | 17 160 ns | +7 850 ns | 1.84x | 6 / 12 |
| 256 | 2048 | 14 031 ns | 23 906 ns | +9 875 ns | 1.70x | 6 / 12 |

Views and slices are free on device; the **device kernel count goes 3 -> 8
per MoE layer per forward**:

```
count-only : _to_copy, histc, add_
+ mass     : _to_copy, histc, add_, _to_copy, lt, where, clamp_, scatter_add_
```

At 75 MoE layers that is 375 extra kernel launches per forward.

**This is not cheap enough to default on.** The in-repo prior measurement
is that a ~10-kernel guarded scatter chain cost 4-5% decode overhead on
the Fruit proxy against a PERFORMANCE.md gate of <0.5%; 8 kernels is the
same regime. I could not measure GPU decode overhead directly (GPUs
0-3 serving, 4-7 encoding), so I am not going to claim it clears the
gate.

**Therefore: opt-in.** Default `VLLM_FQ_GATE_MASS=0` keeps the 3-kernel
count-only fast path bit-for-bit as before. The M5 convergence run should
set `VLLM_FQ_GATE_MASS=1` and record PP/TG against a
`VLLM_FQ_ENABLE=0` baseline on the same rig, per PERFORMANCE.md.

## 7. Tests

`tests/exl3_fungible/test_gate_mass_cpu.py` (20 tests) and
`tests/exl3_fungible/test_base_router_gate_mass_cpu.py` (5 tests), CPU-only.

* real mass differs from count for non-uniform weights — and the argmax
  the policy reads flips from the frequently-hit expert to the confident one
* mass equals count when all gate weights are 1.0
* the padding sentinel `id == E` contributes to neither, and is not folded
  into expert `E-1`
* OOR ids (`-1`, `-1000`, `E`, `E+1`, `E+2`, `300`, `2^20`, `±2^31`) all land
  in the overflow slot; accumulators stay `E+1` long; no weight is lost
* count is byte-identical with and without mass recording
* `mass_is_real()` correct per layer, model-wide, for unbound layers, for
  mixed models, and for a fresh collector
* a router that cannot pass weights downgrades to count-only instead of
  reporting real mass it is not getting
* `summary()` and the `VLLM_FQ_DUMP_STATS` record both carry the flag
* the router half is exercised against the **real** `base_router.py`,
  loaded by path with its five vllm imports stubbed (the source tree's
  vllm is not built, which is why this directory runs `--noconftest`)

`test_base_router_gate_mass_cpu.py` fails 5/5 without the `base_router.py`
change (verified by reverting the file and re-running).

Suite: **206 passed, 11 skipped** (was 181 passed, 11 skipped).

## 8. Reasons `mass` could still alias

Ordered by likelihood. All are visible in the dump's `mass_is_real` field
and in the engine log line.

1. **`VLLM_FQ_GATE_MASS` not set to `1`.** Default is count-only by
   design. `mass_is_real: false`.
2. **`base_router.py` not deployed to the rootfs.** The serve loads
   `/home/mbelleau/rootfs/gg-v20-r33/opt/venv/.../vllm`, not the source
   tree. `deploy-fq.sh` now carries the file; if that is lost (rootfs
   re-extract, HOOKS edit), the bind-time probe downgrades every layer,
   logs a warning per layer, and sets `mass_is_real: false`.
3. **A router that is not a `BaseRouter`, or one overriding
   `_select_experts`.** None exist in-tree today. Same downgrade path.
4. **Legitimately uniform gate weights** would make `mass ≈ count`
   numerically. That is not aliasing — check the flag, not the arrays.

One interpretation note for the convergence analysis: GLM-4/5 MoE routes
through `GroupedTopKRouter` with `scoring_func="sigmoid"`,
`renormalize=config.norm_topk_prob` and `routed_scaling_factor`
(`glm4_moe.py:198-206`). With renormalization the per-token weights sum to
a constant, so per-layer total mass tracks token count and the signal
lives entirely in **how confidence distributes across experts** — which is
exactly the intended discriminator, but it means mass will not diverge
from count in total magnitude, only in shape.

## 9. Files changed

`gg-vllm @ fq/m1-stats-collector`
* `vllm/model_executor/layers/fused_moe/router/base_router.py` — weight-passing capture contract
* `vllm/model_executor/layers/quantization/exl3_fungible/stats.py` — `record_mass`, guarded mass scatter, bind-time probe, `mass_is_real()`
* `vllm/model_executor/layers/quantization/exl3_fungible/integration.py` — `VLLM_FQ_GATE_MASS`, resolved-mode log
* `vllm/model_executor/layers/quantization/exl3_fungible/loop.py` — one field: `mass_is_real` in `_dump_stats`
* `tests/exl3_fungible/test_gate_mass_cpu.py` (new)
* `tests/exl3_fungible/test_base_router_gate_mass_cpu.py` (new)
* `tests/exl3_fungible/test_stats_cpu.py` — two expectations updated for the new weighted-path semantics

`vllm-voipmonitor`
* `research/fungible-quant/runs/gg-env/deploy-fq.sh` — deploy `base_router.py`
* `research/fungible-quant/runs/m5-serve/gate-mass-binding.md` — this report
