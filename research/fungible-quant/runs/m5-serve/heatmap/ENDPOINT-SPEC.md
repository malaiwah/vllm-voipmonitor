# Activation-matrix endpoint — implementation spec

Status: **design, ready to implement**. No code written. Author: design pass
2026-08-11. Target tree: `gg-vllm` branch `fq/m1-stats-collector`.
Companion surface: `../admin-api-spec.md` (shipped) — this spec deliberately
reuses its gating, worker-plumbing and cross-rank conventions so an operator
sees one `/fq` API, not two.
Companion visual design: `./DESIGN.md` — this spec is the data contract behind
it. The two agree everywhere except the Reset control; §7.7 reconciles that.

The ask: a read-only endpoint the operator's heatmap page can poll to see
**which experts are actually being routed to, right now**, per layer, with the
current K tier overlaid, and a way to "reset the counters".

Short version of the answer, derived below from the real dumps and the real
server:

* The signal is a **75 x 256 = 19,200-cell** matrix. Naively serialised the way
  `loop._dump_stats` does it, one sample is **~800 KB** and a 2 s poll is
  **~400 KB/s**. That is not acceptable, and it is not necessary.
* Base64 bf16 + a gzipped response gets the default sample to **32.8 KB
  (gzip -1) / 31.4 KB (gzip -6)** — a **25x** reduction — at 0.39 % worst-case
  relative error, which is far below what a colour ramp can show.
* On the live serve the decayed matrix is **piecewise constant between window
  rolls** — it only changes once every `window_stride` engine steps, i.e. about
  **every 15 s at the observed ~2.1 steps/s**. A weak `ETag` therefore turns a
  2 s poll into ~2 KB/s average, because six polls out of seven are `304`.
* Reset: the page uses a **client-side baseline** by default. A real
  server-side reset is also specified, but it resets the *endpoint's own*
  cumulative accumulator, not the collector — zeroing the collector is a
  third, separately gated, documented-as-harmful option.

---

## 1. What the data actually is

### 1.1 The archived dumps

Four `VLLM_FQ_DUMP_STATS` files in
`research/fungible-quant/runs/m5-serve/results/k3-fq/`:

| File | Records | Bytes | Bytes/record |
|---|---:|---:|---:|
| `stats.jsonl` | 17 | 13,718,704 | 806,982 |
| `stats-code-axis.jsonl` | 13 | 10,424,264 | 801,866 |
| `stats-synthetic.jsonl` | 117 | 93,036,294 | 795,182 |
| `stats-INVALID-truncated-corpus.jsonl` | 51 | 40,921,654 | 802,385 |

`stats-INVALID-truncated-corpus.jsonl` came from a broken replay — its
*contents* are meaningless, but its *shape* is identical to the others and it is
a valid fixture for wire-format tests.

Per-line size is tightly clustered: **795,876 – 810,024 B**, mean **806,982 B**
for `stats.jsonl`. The one outlier (292,420 B, first record of
`stats-synthetic.jsonl`) is the very first interval, where 2,017 of 19,200 cells
are still exactly `0.0` and serialise as three bytes instead of eighteen.

**This is the number that drives the whole design: one sample is ~800 KB of
JSON, and it is 800 KB because 38,400 float64 values are being printed at full
`repr` precision.**

### 1.2 Record shape (identical in all four files)

```
keys: step, interval, layers, tier_of, count, mass
```

* `step` — engine steps since boot. `stats.jsonl` runs 18,200 → 19,800 in
  100-step increments; `interval` runs 182 → 198. So that run had
  `VLLM_FQ_INTERVAL_STEPS=100` (the demo default is 200, `run-demo1.sh:222`).
* `layers` — **75 entries, `[3, 4, …, 77]`**, contiguous, no gaps. Layer 78
  (MTP) is not instrumented and never appears. This is the row axis and it must
  be shipped, not assumed: a client that hardcodes `layer = row + 3` will break
  on any model with a different MoE span.
* `tier_of` — `[75][256]` of Python `int`. In every archived record every cell
  is `3` (these are the K3-uniform runs; `fq_tier_occupancy` on the live serve
  agrees: `{layer="3",tier="k3"} 256.0`, `k4 0.0`). The value domain is
  `occupancy_table.TIERS = (2, 3, 4, 5)` — a `uint8` covers it with room to
  spare.
* `count`, `mass` — `[75][256]` of `float`.
* `mass_is_real` — **absent from all four files.** `loop._dump_stats` writes it
  (`loop.py:933-934`) but that line landed in commit `6e08f683d`
  ("record REAL gate mass, not a copy of the hit count"), *after* these dumps
  were taken. A fixture loader must therefore default it to `False` and must
  **not** try to infer it by comparing `count` to `mass` — `stats.py:302-318`
  is explicit that equal arrays are ambiguous (a uniform router produces them
  legitimately; the alias produces them by construction). In all four files
  `np.array_equal(count, mass)` is `True`, which is consistent with
  `VLLM_FQ_GATE_MASS` having been off — but the arrays are not the evidence,
  the flag is.

### 1.3 What the numbers mean (this changes the endpoint design)

`count` is **not** a hit counter. It is
`decayed()` (`stats.py:320-341`):

```
w_e = Σ_{i<n} λ^i · W[-1-i]        n = min(windows_rolled, window_len)
```

a **λ-decayed sum over a bounded ring of window slots**, returned as float64.
That is why the values are non-integral (`stats.jsonl` record 0: min
`57.90269175357707`, max `98024.43661932467`) and why every per-layer row sums
to exactly the same value to 15 digits (`1610587.985011254…` — top-k routing
puts the same total mass through every layer).

Defaults, unchanged by `serve-glm52.sh`: `window_len=64`, `window_stride=32`,
`decay=0.95` (`stats.py:62-64`).

Three consequences the endpoint must respect:

1. **The matrix has a horizon, and it self-forgets.** Effective horizon is
   `window_stride / (1 - λ)` = `32 / 0.05` = **640 engine steps**; the ring
   spans `64 × 32` = **2,048 steps**. Traffic older than that is already gone.
   "Reset the counters" is therefore *much* less necessary than it sounds
   (§7).
2. **It is piecewise constant.** `_count_win`, `_win_pos` and `_windows_rolled`
   are only written inside `step()` at a roll (`stats.py:287-298`), and
   `decayed()` reads nothing else. Between rolls the returned array is
   **byte-identical**. On the live serve, intervals 195→198 (300 steps) took
   02:59:29 → 03:01:55 = 146 s, i.e. **~2.05 steps/s**; intervals 186→189 give
   ~2.13 steps/s. A roll every 32 steps is therefore **a roll every ~15 s**.
   Polling every 2 s oversamples the decayed view by ~7.5x.
3. **`count / step` is not a rate.** The decayed sum is bounded, so dividing by
   a monotonically growing step counter is wrong. Two correct rate paths are
   specified in §5.4.

The live, sub-roll signal does exist: `collector.count_buf[lid][:E]` is a float32
accumulator that the capture fn adds to on **every** forward
(`stats.py:224-227`), zeroed at each roll. It is exact (histc output, integral
below 2^24) and it is what makes a 2 s refresh actually show something moving.
The endpoint exposes it as `live_count`.

### 1.4 Value ranges (for choosing a dtype)

From `stats.jsonl` and `stats-synthetic.jsonl`:

| Quantity | Observed |
|---|---|
| `count` min (nonzero) | `0.0437…` (synthetic, early interval) |
| `count` max | `337,651.33` |
| `count` per-layer row sum | up to `6,749,860.85` |
| zero cells | 0 – 10.5 % depending on how warm the run is |
| dynamic range within one sample | ~4.5 decades |

Four and a half decades rules out any fixed-point/linear-scaled integer
encoding that is scaled to the maximum: a `uint16` scaled to the per-layer max
gives a quantum of `rowmax/65535` ≈ 5 counts at `rowmax=337k`, which erases the
distinction between "cold" and "never routed" — precisely the question an
operator opens a heatmap to answer. The encoding must be **floating point** or
log-domain. See §5.

---

## 2. How a request reaches the numbers

### 2.1 The dev-mode gate

```
vllm/entrypoints/openai/api_server.py:244-247
    if envs.VLLM_SERVER_DEV_MODE:
        from vllm.entrypoints.serve import register_vllm_dev_api_routers
        register_vllm_dev_api_routers(app)
```

`register_vllm_dev_api_routers` (`vllm/entrypoints/serve/__init__.py:37-77`)
logs `SECURITY WARNING: Development endpoints are enabled!` (lines 38-41), then
attaches `cache` (43-45), `rlhf` (47-49), `rpc` (51-53), `server_info` (55-59),
`sleep` (61-63), and — already — the FQ admin router behind its own env gate
(65-77). Our block goes immediately after, same shape.

`serve-glm52.sh:94` sets `VLLM_SERVER_DEV_MODE=1`, so the live serve is already
inside this branch.

The reference router shape is five lines
(`vllm/entrypoints/serve/dev/rpc/api_router.py:16,19-20,23,57-58`): module-level
`router = APIRouter()`, a local `engine_client(request)` returning
`request.app.state.engine_client`, decorated handlers,
`def attach_router(app): app.include_router(router)`.

### 2.2 The collector is in the WORKERS, not the API server

`gpu_worker.py:807-811`:

```python
from vllm.model_executor.layers.quantization.exl3_fungible.integration import (
    maybe_init_fq_state,
)
self.model_runner.fq_collector = maybe_init_fq_state(self.model_runner)
```

One collector per **worker process**. With `--tensor-parallel-size 4`
(`serve-glm52.sh:111`) that is four processes, none of which is the API server
process, and none of which is the EngineCore process. The endpoint must
round-trip.

### 2.3 The round trip, end to end

```
FastAPI handler   (API server process)
  └─ await engine_client.collective_rpc(method, timeout, args, kwargs)
       vllm/engine/protocol.py:227-235          (protocol)
       vllm/v1/engine/async_llm.py:964-976      (AsyncLLM)
       vllm/v1/engine/core_client.py:1188-1197  (collective_rpc_async)
       vllm/v1/engine/core_client.py:1104-1116  (_call_utility_async — msgpack
                                                 over a socket, NO client-side
                                                 timeout: it just awaits a future)
  ── process boundary ──
EngineCore process
       vllm/v1/engine/core.py:1387-1394   run_busy_loop:
                                            1) _process_input_queue()
                                            2) _process_engine_step()
       vllm/v1/engine/core.py:1514-1527   UTILITY dispatch
       vllm/v1/engine/core.py:1562-1579   _invoke_utility_method — catches every
                                            exception and returns it to the
                                            client as `failure_message`
       → executor.collective_rpc(...)
       vllm/v1/executor/multiproc_executor.py:358-421
            :376  deadline = now + timeout   (one deadline for all ranks)
            :392  rpc_broadcast_mq.enqueue((method, args, kwargs, output_rank))
            :398-414 get_response(): dequeue one response per rank
  ── process boundary x4 ──
Worker process (TP rank 0..3)
       Worker.<method>(...)  — resolved BY NAME on the worker object
```

**The utility call runs between engine steps and blocks the step loop.**
`run_busy_loop` drains the input queue *then* steps
(`core.py:1391-1393`), and `collective_rpc` is synchronous (`non_block` defaults
to `False`, `multiproc_executor.py:364,421`). Every millisecond the endpoint
spends inside the workers is a millisecond of stalled decode. This is the single
strongest constraint on the design and it dictates §4.4 (single-flight + TTL
cache) and §5.5 (encode in the API server, not the worker).

`collective_rpc` returns **a list, one entry per worker**
(`multiproc_executor.py:398-415`; aggregation reference at
`dev/rpc/api_router.py:46-54`). The FQ admin API already normalises that list —
`_rank_results` (`admin.py:2039-2047`) decodes bytes/str/dict uniformly,
`_first_error` (`:2050-2054`) surfaces the first `ok:false`, `_agree`
(`:2057-2059`) checks cross-rank equality of a named key. **Reuse all three by
import**; do not re-implement them in `heatmap.py`.

### 2.4 Are the four ranks' counters the same?

Yes, on this serve, and the endpoint must not silently assume it.

The collector binds to `BaseRouter` (`integration.py:149-162`), i.e. to the
**gate**, which under tensor parallelism is replicated: every rank computes the
same `topk_ids` from the same hidden states, so every rank's histogram is the
same. `serve-glm52.sh` passes no `--enable-expert-parallel`, so EP=1 and
`num_experts` is the global 256 on every rank
(`integration.py:157`, `routers[0][1].global_num_experts`).

Under **data parallelism** this breaks: each DP replica sees different requests,
so the counters differ and would have to be summed — and worse,
`DPLBAsyncMPClient.call_utility_async`
(`core_client.py:1449-1458`) runs the utility on every engine and
**returns only `[0]`**, so the endpoint would silently report one replica's
traffic as if it were the whole serve. The admin API answers this with
`501 dp_not_supported`; we do the same (§8).

**Merge rule (v1):** rank 0 is canonical; other ranks return a digest only, and
the digests must match. See §4.3.

---

## 3. The endpoint

### 3.1 Routes

New module `vllm/model_executor/layers/quantization/exl3_fungible/heatmap.py`,
exporting `attach_router(app, *, environ=None) -> bool` and
`build_router(*, environ=None)`, following `admin.py:2062-2081,2084-2102`
exactly (router lives in the FQ package, not under `vllm/entrypoints/`, so the
CPU suite can drive it with FastAPI `TestClient` — see `admin-api-spec.md`
§I.1 for why importing anything under `vllm/entrypoints/` is not testable in
this environment).

`APIRouter(prefix="/fq", tags=["fungible-quant heatmap"])`. Two `APIRouter`s
may share a prefix; FastAPI merges them, and `/fq/heatmap` cannot be shadowed by
the admin router's `/fq/layer/{layer}` because the literal segment differs.

| Method | Route | Mutating | Purpose |
|---|---|---|---|
| `GET` | `/fq/heatmap` | no | one sample of the activation matrix |
| `GET` | `/fq/heatmap/meta` | no | shape + gates only, ~400 B, no worker RPC after the first call |
| `POST` | `/fq/heatmap/reset` | see §7 | reset semantics, three scopes |

`/fq/heatmap/meta` exists so the page can lay out its 75 x 256 grid, learn the
layer ids and the encoding, and render its axes *before* it pays for a sample —
and so a dashboard can health-check the surface without stalling the engine.
Its answer is cached in the API server for the process lifetime except
`fq_active`.

### 3.2 Query parameters (`GET /fq/heatmap`)

| Param | Type | Default | Meaning |
|---|---|---|---|
| `include` | csv of `live`,`cum`,`mass` | `mass` | extra arrays; `mass` is honoured only when `mass_is_real` |
| `precision` | `bf16` \| `f32` | `bf16` | wire dtype for the float arrays |
| `reduce` | `rank0` \| `all` | `rank0` | `all` returns every rank's arrays (debug; 4x the bytes) |
| `max_age_ms` | int | `1000` | serve the cached sample if it is younger than this; `0` forces a fresh RPC |
| `layers` | csv or `a-b` | all | row subset, e.g. `layers=20-40`; shrinks the payload proportionally |

`fields`/`include` is additive and validated: an unknown token is
`400 unknown_include`, matching the admin API's `unknown_field` discipline.

### 3.3 Gating

Same ladder as the admin API (`admin-api-spec.md` §2, `admin.py:2073-2075`,
`:2112-2126`), evaluated at attach time **and** re-checked per request:

1. `VLLM_SERVER_DEV_MODE=1` — the router is not registered otherwise.
2. `VLLM_FQ_HEATMAP=1` (alias accepted: `VLLM_FQ_HEATMAP_API`) — the second,
   explicit opt-in. Off → the routes do not exist; if reached by another path,
   `404 {"error": {"code": "fq_heatmap_disabled"}}`.
3. `VLLM_FQ_HEATMAP_TOKEN` — if set, require header `X-FQ-Heatmap-Token`,
   compared with `hmac.compare_digest`; else `403 fq_heatmap_forbidden`.
   *(Deliberately a separate token from `VLLM_FQ_ADMIN_TOKEN`: read-only
   telemetry should be shareable with a dashboard without handing out the
   credential that can mutate live weights.)*
4. FQ must be running: `worker.model_runner.fq_collector` must exist. Reuse
   `admin._loop_state` (`admin.py:1804-1812`) when the loop is up; fall back to
   the bare collector when it is not — `maybe_init_fq_state` degrades to a
   collector-only object on loop-boot failure (`integration.py:119-125`), and
   the heatmap is still perfectly meaningful in that mode (no `tier`, no
   `interval`). Neither present → `404 fq_not_active`.

Why a second gate at all, for a read-only route: the matrix is a fingerprint of
what the model is being asked to do. 19,200 routing intensities sampled over
time are enough to distinguish workloads and, with effort, to attack routing.
It is not `/health`.

`attach_router` logs at INFO (not WARNING — nothing here mutates):
`FQ heatmap API enabled (VLLM_FQ_HEATMAP=1): GET /fq/heatmap`.

### 3.4 Response schema (`200`)

```jsonc
{
  "schema": "fq-heatmap/1",

  // identity and ordering ------------------------------------------------
  "sample_id": 137,                 // monotonically increasing, +1 per sample
                                    // actually taken from the workers; a cache
                                    // hit repeats the previous sample_id
  "server_boot_id": "b1f0c8a2",     // uuid4[:8], fixed for the API server
                                    // process — sample_id resets across it
  "sampled_at_unix_ms": 1786455301123,
  "cached": false,                  // true => served from the TTL cache
  "cache_age_ms": 0,

  // engine clocks --------------------------------------------------------
  "step": 19100,                    // collector._step, every step incl. dummy
  "real_steps": 19094,              // loop._real_steps, excludes dummy; null
                                    // in collector-only mode
  "interval": 191,                  // loop._intervals_run; null if no loop
  "window": {"len": 64, "stride": 32, "decay": 0.95,
             "rolled": 596,         // collector._windows_rolled
             "n_effective": 64,     // min(rolled, len) — the n in decayed()
             "steps_since_roll": 12,// step % stride: the age of live_count
             "horizon_steps": 640}, // stride/(1-decay), the decayed horizon

  // shape ----------------------------------------------------------------
  "layers": [3, 4, "…", 77],        // 75 ids, ascending, no gaps here but do
                                    // not assume contiguity
  "num_layers": 75,
  "num_experts": 256,
  "cells": 19200,                   // == num_layers * num_experts; validate

  // provenance -----------------------------------------------------------
  "mass_is_real": false,            // collector.mass_is_real(); NEVER inferred
  "ranks": {"count": 4, "canonical": 0, "agree": true, "reduce": "rank0"},
  "policy_sha": "9c1f…",            // null in collector-only mode
  "apply_mode": "atomic",

  // encoding -------------------------------------------------------------
  "encoding": {
    "layout": "layer-major",        // index = row * num_experts + expert
    "byte_order": "little",
    "count":      "bf16",
    "mass":       "bf16",
    "live_count": "bf16",
    "cum_count":  "u32",
    "tier":       "u8",
    "transfer":   "base64"
  },

  // the arrays -----------------------------------------------------------
  "count":      "AABIQ…",           // 19200 x bf16, base64  — decayed window
  "mass":       null,               // omitted/null when mass_is_real == false
  "tier":       "AwMDAw…",          // 19200 x u8,   base64  — 2|3|4|5
  "live_count": "AAAgQ…",           // only when include=live
  "cum_count":  null,               // only when include=cum
  "cum_since_step": null,           // step at which cum_count was last rebased
  "cum_lossy": false,               // true => a gap was dropped, see §7.4

  "warnings": []
}
```

Rules a client can rely on:

* Every array decodes to exactly `cells` elements. A client **must** check this
  and refuse the sample otherwise; a truncated base64 blob must never be
  silently rendered as a shifted heatmap.
* `mass` is `null` **iff** `mass_is_real` is `false`. In that case the client
  aliases mass to count itself — which is exactly what `decayed()` does
  server-side (`stats.py:338-339`) — and the UI must label it "mass = count
  (gate mass not recorded; set `VLLM_FQ_GATE_MASS=1`)". Shipping a duplicate
  76 KB array to say "same as the other one" is the single easiest 50 % saving
  on the wire, and it is free.
* `tier` is present whenever the loop state exists; `null` in collector-only
  mode.
* `sample_id` is strictly increasing within a `server_boot_id`. A client that
  sees it go backwards must discard its baseline.

### 3.5 Response headers

```
Content-Type:     application/json
Content-Encoding: gzip
ETag:             W/"fqhm-1-3a91c7d2e4f6b088"
Cache-Control:    no-cache, must-revalidate
X-FQ-Sample-Id:   137
X-FQ-Step:        19100
```

`Cache-Control: no-cache` (revalidate every time) rather than `no-store`, so the
`ETag` path in §6 works.

---

## 4. Worker plumbing

### 4.1 The worker method

One thin lazy delegate on `Worker`, mirroring `gpu_worker.py:829-848` exactly
(same laziness contract, same JSON-string-in/JSON-string-out convention as
`/collective_rpc`, `dev/rpc/api_router.py:38-41`):

```python
# vllm/v1/worker/gpu_worker.py — +6 lines, immediately after fq_admin_apply
def fq_heatmap_sample(self, request_json: str = "{}") -> str:
    from vllm.model_executor.layers.quantization.exl3_fungible.heatmap import (
        worker_sample,
    )
    return worker_sample(self, request_json)
```

`heatmap.py` also exports an `FqHeatmapWorker` mixin for
`--worker-extension-cls` deployments, with the same warning `admin.py:2015-2024`
carries: vLLM injects exactly **one** extension class into `Worker.__bases__`
after asserting no attribute collides (`worker_base.py:261-286`), so use the
mixin **or** the core method, never both. On *this* serve the extension slot is
already taken by `--worker-extension-cls fq_reload.FqReloadWorker`
(`serve-glm52.sh:122`), so the core method is the only usable route here.

### 4.2 What the worker does per sample

```
worker_sample(worker, request_json) -> json str
  state     = admin._loop_state(worker)      # or the bare collector
  collector = state.collector if hasattr(state, "collector") else state
  layers    = state.layers    if loop else sorted(collector.count_buf)

  1. read window bookkeeping: _windows_rolled, _win_pos, _step
  2. count/mass: [collector.decayed(cid) for cid in mapped(layers)]
       -> exactly what loop._read_stats does (loop.py:661-667)
  3. live_count (only if asked): torch.stack([count_buf[cid][:E] ...]).cpu()
  4. cum_count  (only if asked): integrate the ring, §7.4
  5. cast to the requested wire dtype, return raw little-endian bytes
     base64'd inside the JSON string
```

Costs, per sample, per rank:

* Step 2 is **the same call the loop already makes every interval**
  (`loop.py:709`, `_read_stats` at `loop.py:661-667`): 75 x (gather + pow + mul
  + sum) on device plus 150 x 2 KB device→host. It runs in production today at
  one interval per 100–200 steps, i.e. roughly every 50–100 s at the observed
  ~2.1 steps/s. A 2 s poll would run it ~30x more often, which is why the
  cache in §4.4 is not optional.
* Step 3 is one stacked D2H of `75 x 256 x 4 B` = **76,800 B**.
* Step 4 is one stacked D2H of `n_new x 75 x 256 x 8 B`; at a 2 s poll
  `n_new` is 0 or 1, so ≤ **153,600 B**.

No tearing is possible: the worker's RPC handler and `execute_model` (which
calls `collector.step()`, and therefore does the roll) are dispatched from the
same worker loop over `rpc_broadcast_mq`, so they are serialised. The endpoint
must still read `_windows_rolled` **before** and **after** the array reads and
retry once if it moved — cheap insurance, and it makes the CPU test for the
race trivial to write.

### 4.3 Cross-rank fan-out and merge

`collective_rpc` executes on **all** ranks and returns all their results; the
async `EngineClient` path exposes no `unique_reply_rank`
(`protocol.py:227-235` / `async_llm.py:964-976` take only
`method, timeout, args, kwargs`; the parameter exists only on
`MultiprocExecutor.collective_rpc`, `multiproc_executor.py:365`, which we do
not reach). So all four ranks will do work regardless.

Therefore: **rank 0 returns the arrays; every other rank returns a digest.**

```jsonc
// rank 0
{"ok": true, "rank": 0, "canonical": true,
 "digest": "3a91c7d2…",          // sha256 over count|tier|step|rolled, 16 hex
 "step": 19100, "rolled": 596, "mass_is_real": false,
 "count": "…b64…", "tier": "…b64…"}

// ranks 1..3   (~180 bytes each)
{"ok": true, "rank": 2, "canonical": false,
 "digest": "3a91c7d2…", "step": 19100, "rolled": 596, "mass_is_real": false}
```

The router then:

1. `results = admin._rank_results(raw)` (`admin.py:2039-2047`).
2. `err = admin._first_error(results)` → re-raise as the caller's error
   (`admin.py:2050-2054`).
3. `agree = admin._agree(results, "digest")` (`admin.py:2057-2059`).
4. Serve rank 0's arrays, with `"ranks": {"agree": agree, …}`.

**Divergence is reported, not fatal.** Unlike the admin API — where divergence
after a weight mutation means the ranks' weights disagree and it is a `500`
(`admin-api-spec.md` §7.3) — here it means the routing histograms differ, which
under TP-replication is a real anomaly worth surfacing but is not a reason to
deny the operator their picture. So: `200` with `ranks.agree = false`, a
`warnings` entry naming the disagreeing ranks and their digests, and a
`logger.warning` (rate-limited to once per 60 s so a persistent divergence does
not flood the engine log). `?reduce=all` then returns every rank's arrays so the
operator can diff them.

`ranks.agree` deliberately compares the **digest**, not the arrays, so the
check costs 32 bytes per rank rather than 76 KB.

### 4.4 Single-flight, TTL cache, and what "slow" means

Because every sample stalls the engine step loop (§2.3), the endpoint must
decouple *client* poll rate from *engine* RPC rate:

* Module-level `asyncio.Lock` in the router (same pattern as `admin.py:2103`).
  Concurrent requests do **not** queue up N RPCs; the first takes the lock, the
  rest await it and are then served from the cache.
* TTL cache of the last full envelope, keyed by
  `(include-set, precision, reduce, layers-subset)`. Default TTL
  `max_age_ms=1000`, ceiling `VLLM_FQ_HEATMAP_MIN_PERIOD_MS` (default `500`) —
  `max_age_ms` below the floor is clamped, not honoured, and the clamp is
  reported in `warnings`. Three browser tabs polling at 2 s therefore cost
  **one** RPC per second at worst, not three.
* `?max_age_ms=0` bypasses the TTL but not the floor.

**Timeout.** Pass `timeout=VLLM_FQ_HEATMAP_TIMEOUT_S` (default `5.0`) through
to `collective_rpc`. The deadline is shared across ranks
(`multiproc_executor.py:376`): rank 0 doing the real work eats most of it and
the digest-only ranks are near-instant, so 5 s is generous by two orders of
magnitude for a ~1 ms sample.

**What happens if a rank is slow or fails** — this is the part that needs
stating plainly, because the failure is not clean:

* *A rank raises.* `WorkerProc` returns a non-`SUCCESS` status, the executor
  raises `RuntimeError("Worker failed with error '…'")`
  (`multiproc_executor.py:409-413`), `_invoke_utility_method` catches it and
  returns it as `failure_message` (`core.py:1576-1578`), and the awaited future
  raises in the API server. → `503 fq_heatmap_worker_error`, message forwarded
  verbatim, nothing cached. The next poll retries. Harmless.
* *A rank is slow past the deadline.* `mq.dequeue` raises and the executor
  raises `TimeoutError("RPC call to fq_heatmap_sample timed out.")`
  (`multiproc_executor.py:404-407`). **The broadcast has already been enqueued
  (`:392`) and the slow worker will still execute it and still write its
  response.** Nothing drains that orphaned response, so the *next* collective
  RPC on that queue can read the stale one — the RPC channel is desynchronised
  and every later admin/heatmap call is suspect.

  Therefore a timeout is treated as **poisoning, not a transient**:
  `504 fq_heatmap_timeout`; set a module-level `_POISONED` flag; every
  subsequent `GET /fq/heatmap` returns `503 fq_heatmap_poisoned` with
  `"guidance": "the collective-RPC channel may be desynchronised after a
  timeout; heatmap sampling is disabled until the serve restarts"`, and does
  **not** issue another RPC. Log once at ERROR. A telemetry endpoint must never
  be the thing that corrupts the channel the admin API needs.

  `POST /fq/heatmap/reset {"scope":"poison"}` clears the flag for an operator
  who has decided the risk is acceptable — behind the same token as the other
  writes. It issues no RPC of its own.

* *Engine unavailable* (sleeping / shutting down): `_reject_utility_in_shutdown`
  (`core.py:1546-1560`) answers `"Server shutting down"` → `503
  engine_unavailable`.

### 4.5 Data parallelism

If `vllm_config.parallel_config.data_parallel_size > 1`, return
`501 dp_not_supported` at attach time and per request, matching
`admin-api-spec.md` §7.2. The reason is concrete and worth putting in the error
body: `DPLBAsyncMPClient.call_utility_async` (`core_client.py:1449-1458`)
gathers from every engine and returns `[0]`, so the endpoint would report one
replica's routing as the whole serve's. Silently wrong beats loudly missing
here, so we choose loudly missing.

---

## 5. Payload size and encoding

### 5.1 The problem, in bytes

19,200 cells per array; up to four arrays (`count`, `mass`, `tier`,
`live_count`). Measured on `stats.jsonl` record index 9 (`step=19100`,
`interval=191`), gzip via Python `zlib`:

| Encoding of ONE 19,200-cell count array | raw | gzip -6 |
|---|---:|---:|
| JSON, float64 `repr` (what `_dump_stats` writes) | 375,459 | 180,184 |
| JSON, rounded to integers | 133,012 | 52,148 |
| JSON, 4 significant digits | 171,412 | 45,016 |
| base64 float64 | 204,800 | 153,641 |
| base64 float32 | 102,400 | 75,577 |
| base64 uint32 (rounded integer counts) | 102,400 | 55,470 |
| base64 uint16 (per-layer scaled) | 51,200 | 38,678 |
| **base64 bfloat16** | **51,200** | **30,629** |
| base64 uint8 (per-layer scaled) | 25,600 | 18,899 |
| base64 uint8 — the `tier` array | 25,600 | **66** |

`tier` is the easy one: 19,200 base64'd bytes of `3`s and `4`s gzip to
**66 bytes**. Never encode it as JSON integers (57,750 B raw).

### 5.2 Why bfloat16

`bf16` is float32 with the low 16 mantissa bits dropped: same exponent range
(so `337,651.33` and `0.0437` both survive), 8 mantissa bits, **measured
worst-case relative error 0.38898 %** across all 19,200 cells of a real record
(`f32` on the same record: 5.93e-8). A colour ramp on a 4.5-decade log scale
resolves nothing close to 0.4 %.

It also decodes in four lines of JavaScript with no library, because the bit
pattern *is* the top half of a float32:

```js
function bf16(b64, n) {                    // -> Float32Array(n)
  const bin = atob(b64), u16 = new Uint16Array(n), out = new Float32Array(n);
  for (let i = 0; i < n; i++) u16[i] = bin.charCodeAt(2*i) | (bin.charCodeAt(2*i+1) << 8);
  const u32 = new Uint32Array(out.buffer);
  for (let i = 0; i < n; i++) u32[i] = u16[i] << 16;
  return out;
}
```

`uint16` scaled per layer is the same raw size but gzips *worse* (38,678 vs
30,629 — the scaled integers have high-entropy low bytes where bf16 has a
repeating exponent plane), and it destroys the low end of a 4.5-decade range
(§1.4). `bf16` wins on both axes.

`?precision=f32` exists for one reason: a client doing exact arithmetic on
differences of nearly-equal samples (§7.2) will lose small deltas to bf16's
0.39 % quantum. It costs 2.4x the bytes (table below) and the page does not use
it.

### 5.3 The chosen wire, measured end to end

**JSON envelope, arrays as base64 of little-endian raw bytes, whole response
gzipped by the handler.** Full envelope including all metadata fields of §3.4,
measured on the same record (index 9, `step=19100`):

| `include` | `precision` | JSON bytes | gzip -1 | gzip -6 | @2 s poll (gzip -1) |
|---|---|---:|---:|---:|---:|
| — (count+tier, mass aliased) | bf16 | 77,435 | **32,785** | 31,353 | 16.0 KB/s |
| `mass` (real gate mass) | bf16 | 128,644 | 64,658 | 61,872 | 31.6 KB/s |
| `live` | bf16 | 128,651 | 64,896 | 62,342 | 31.7 KB/s |
| `live,mass` | bf16 | 179,860 | 96,740 | 92,880 | 47.2 KB/s |
| — | f32 | 128,609 | 76,210 | — | 37.2 KB/s |
| `live,mass` | f32 | 333,474 | 217,825 | — | 106.4 KB/s |
| *reference: `_dump_stats` JSON* | f64 | 808,999 | 359,878 | — | **175.7 KB/s** |

**Default path: 32.8 KB per sample, 16.0 KB/s at a 2 s poll — 25x smaller than
the naive JSON, and §6 takes the *average* to ~2 KB/s.**

`gzip` level: **1**, not 6. Measured encode+compress cost in the API server
process for the largest variant (`live,mass`, bf16): **2.20 ms at level 1 vs
5.23 ms at level 6**, for 96,740 vs 92,880 bytes — 2.4x the CPU for 4 % of the
bytes. Level 1 it is. (For `f32` the same comparison is 6.16 ms vs 11.87 ms.)
Even 2.2 ms is worth keeping off the event loop: run the compress in
`starlette.concurrency.run_in_threadpool`.

vLLM's `build_app` installs **no** `GZipMiddleware`
(`api_server.py:203-356` — CORS, auth, request-id, scaling, and the
user-supplied `--middleware` list, nothing else), so the handler compresses its
own body and sets `Content-Encoding: gzip` itself. Do not assume a proxy will
do it. Honour `Accept-Encoding`: if the client did not offer gzip, send the
plain 77 KB and note it in `warnings`.

### 5.4 Rates, without a cumulative counter

Two exact answers, no extra bytes:

**(a) From the decayed window alone.** If the per-step hit rate `r` were
constant, `count = r · stride · (1 - λⁿ)/(1 - λ)`, so

```
hits_per_step ≈ count · (1 - decay) / (window.stride · (1 - decay ** window.n_effective))
```

At the shipped defaults with a full ring (`n=64`, `λ=0.95`, `λⁿ=0.03752`) the
denominator is `32 · 19.2496 = 615.99`, so `hits_per_step ≈ count / 616`.
Everything the client needs (`decay`, `stride`, `n_effective`) is in the
envelope. This is a smoothed rate over the ~640-step horizon (`horizon_steps`
is the `n → ∞` limit, `stride/(1-λ) = 640`; the finite-ring denominator is
615.99).

**(b) Instantaneous, from `live_count`.** `live_count` is the un-rolled
accumulator; `window.steps_since_roll` is exactly how many steps it covers.
`hits_per_step = live_count / steps_since_roll` — exact integer counts over an
exact step count. Guard `steps_since_roll == 0` (a sample taken immediately
after a roll): the endpoint then reports `steps_since_roll: 0` and the client
falls back to (a) for that frame.

Neither needs a monotonic counter, which is why `cum_count` is opt-in (§7.4)
rather than default: it is the single most expensive array on the wire
(u32 base64 gzip -1 = 67 KB even at modest magnitudes, more as it grows).

### 5.5 Where the encoding happens

Cast, base64 and gzip in the **API server process**, not the worker. The worker
returns raw little-endian bytes already base64'd (the `/collective_rpc`
convention is JSON strings both ways, `dev/rpc/api_router.py:38-41`, and the
admin API follows it, `admin.py:2131-2134`) but does **no** compression: every
CPU millisecond in the worker is a stalled engine step (§2.3), whereas the API
server is a separate process with an idle event loop.

### 5.6 Levers deliberately left for v1.1

Measured, real, and not worth the complexity yet:

* **Byte-plane transposition** before gzip (ship all low bytes, then all high
  bytes): bf16 goes 31,955 → **25,561** at gzip -1 (**-20 %**); the opt-in u32
  `cum_count` goes 67,030 → **50,767** (**-24 %**). One `.view(u8).reshape(-1,
  itemsize).T` server-side and a de-interleave loop client-side. Add it behind
  `encoding.transform: "byteplanes"` when someone actually needs the 20 %.
* **Raw binary body** (`Accept: application/octet-stream`, JSON header in a
  response header) drops base64's 33 % inflation: 172,800 raw vs 230,400 b64
  for count+mass+tier. After gzip the gap narrows to ~11 %, which does not pay
  for a second content type.
* **Deflate-then-base64** (compress the array, then base64 it, inside the JSON)
  reaches 36,105 B for the default sample vs 32,785 B for plain-base64 +
  `Content-Encoding: gzip`. The transport-level answer is both smaller and
  simpler for the client. Rejected.

---

## 6. Freshness: the ETag makes the poll nearly free

From §1.3(2): `decayed()` is byte-identical between window rolls, and on this
serve a roll happens every ~15 s. `tier` changes only when the policy commits.
So the *entire default response body*, minus `sample_id` and the timestamp, is
a pure function of:

```
etag_basis = (schema_version, windows_rolled, win_pos,
              sha256(tier_bytes)[:8], mass_is_real,
              layers_digest, num_experts,
              include_set, precision, reduce)
```

`ETag: W/"fqhm-1-<sha256(etag_basis)[:16]>"`, **weak** — the body is
semantically equivalent, but `sample_id` / `sampled_at_unix_ms` / `cache_age_ms`
differ, so a strong ETag would be a lie.

On `If-None-Match` match → **`304 Not Modified`**, no body, headers only
(`ETag`, `X-FQ-Sample-Id`, `X-FQ-Step`). The client keeps the arrays it already
decoded and just advances its clock.

Effect at the observed cadence: a 2 s poll against a 15 s roll yields roughly
**one 32.8 KB `200` and six ~200 B `304`s per roll** — about **2.2 KB/s
average**, versus 175.7 KB/s for the naive JSON. That is a **~80x** reduction,
and most of it comes from having read the collector rather than from clever
compression.

The `304` still costs an RPC unless the TTL cache covers it — so the
`max_age_ms` default of 1000 ms means at most one RPC per second regardless of
how many tabs are open, and the `304` is decided in the API server from the
cached basis.

**`include=live` defeats the ETag by design**: `live_count` changes every step,
so the basis must include `step` and no two samples match. Document it in
`/fq/heatmap/meta`. Guidance for the page: poll `include=live` at 2 s only while
the operator has the "live" toggle on (47.2 KB/s worst case with `mass`);
otherwise the ETag path at ~2 KB/s.

---

## 7. Reset semantics

The operator asked to "reset the counters". There are three different things
that could mean, they have very different blast radii, and the endpoint should
be explicit about which one it is doing.

### 7.1 The three scopes

`POST /fq/heatmap/reset`

```jsonc
{"scope": "client" | "heatmap" | "collector" | "poison",
 "reason": "operator marked a baseline before the coder workload"}
```

| `scope` | Server state touched | Gate | Affects the policy? |
|---|---|---|---|
| `client` *(default)* | none | read gate | no |
| `heatmap` | `heatmap.py`'s own `cum_count` accumulator | write token | no |
| `collector` | `collector._count_win` / `_mass_win` / `_windows_rolled` | `VLLM_FQ_HEATMAP_ALLOW_COLLECTOR_ZERO=1` **and** token | **yes — see §7.5** |
| `poison` | clears the §4.4 poison flag | write token | no |

### 7.2 `scope: "client"` — what the page uses, and why

The server does nothing but hand back a marker:

```json
{"scope": "client", "baseline_sample_id": 137, "baseline_step": 19100,
 "applied": false,
 "note": "no server state was changed; snapshot and subtract client-side"}
```

The page keeps the decoded `count` array from sample 137 and renders
`count_now - count_baseline` (or `count_now / count_baseline` on a diverging
ramp).

**This is the default the page uses.** Reasons, in order of weight:

1. **It cannot destroy data another consumer is reading.** The same counters
   feed the policy loop every interval (`loop.py:709`), the `fq_jaccard` gauge,
   and any other dashboard pointed at the same serve. A destructive reset by
   one viewer silently rewrites what every other viewer and the policy sees.
   Read-only telemetry that can corrupt the system it observes is a bad trade
   for a UI convenience.
2. **Compare mode comes free.** Hold several baselines (start of the math
   workload, start of the code workload) and diff any pair, in the browser,
   with no round trip. A server-side zero gives exactly one comparison and
   destroys the ability to make another.
3. **It is instant and per-viewer.** No RPC, no engine stall, no coordination
   between tabs.
4. **It is honest about the horizon.** See the caveat below — a client-side
   baseline makes the "difference of two decayed windows" nature visible in the
   API instead of hiding it behind a button labelled "reset".

**The caveat, stated in the UI, not just here:** because `count` is a
λ-decayed *window* and not a cumulative total, `count_now - count_baseline` is a
**difference of two decayed windows**, not "traffic since the mark". It can be
negative (an expert that cooled off), and once more than ~640 steps have passed
the baseline has fully decayed out of the current window, so the difference
converges back to the plain current value. That is a perfectly good
*"what changed since I marked"* signal — label the view **"change since mark"**,
not "since reset". For a true "hits since the mark" total, use `include=cum`
with `scope: "heatmap"`.

Precision note: bf16's 0.39 % quantum sets the floor on a detectable change.
A client doing tight difference arithmetic should request `precision=f32`
(2.4x the bytes, 6e-8 relative error).

### 7.3 `scope: "heatmap"` — the real, harmless server-side reset

This is the answer to "the operator explicitly asked for a reset". It rebases
`heatmap.py`'s **own** cumulative accumulator (§7.4) on every rank, via
`collective_rpc("fq_heatmap_sample", {"op": "reset_cum"})`, and touches nothing
the collector or the policy owns:

```json
{"scope": "heatmap", "applied": true, "ranks": 4, "agree": true,
 "cum_since_step": 19104, "previous_cum_since_step": 12000,
 "note": "cumulative accumulator rebased; the collector window and the policy
          are untouched"}
```

After this, `include=cum` returns **exact hits since the reset**, shared by
every viewer, and the policy loop has no idea it happened. This is what the
operator's "reset the counters" button should call when they want a reset that
is visible to everyone, and it is the only server-side reset the page ever
issues.

### 7.4 `cum_count` — how a cumulative exists at all

There is no cumulative counter in the collector: `count_buf` is zeroed at every
roll (`stats.py:295-296`) and the window ring is decayed on read. But the ring
itself is **exact, undecayed `int64`** (`stats.py:131-133`), so
`heatmap.py` can maintain a true cumulative with **zero hot-path cost** by
integrating the slots that have rolled since its last sample:

```python
rolled  = collector._windows_rolled
win_pos = collector._win_pos
n_new   = rolled - acc.rolled              # first sample: n_new = 0, cum = 0
if n_new > collector.window_len:           # we were away too long
    n_new, acc.lossy = collector.window_len, True
slots = [(win_pos - 1 - i) % collector.window_len for i in range(n_new)]
acc.cum += stack([_count_win[cid][slots].sum(0) for cid in cids]).cpu().numpy()
acc.rolled = rolled
```

* **Exact** while `n_new <= window_len` — i.e. while the poll interval is under
  `64 * 32 = 2,048` steps ≈ **17 minutes** at the observed cadence. A 2 s poll
  is three orders of magnitude inside that.
* **Detectably lossy** otherwise: `cum_lossy: true` and a `warnings` entry
  naming how many slots were dropped. Never silently wrong.
* Lags the true total by at most `window.steps_since_roll` steps (the un-rolled
  partial), which the envelope reports; add `live_count` if the client wants
  that residue.
* Wire dtype `u32`, holding hits **since the last rebase**, not since boot —
  which keeps the magnitudes small (better compression) *and* is exactly the
  semantics of "since reset". If any cell approaches `2**31`, the endpoint
  auto-rebases, sets `cum_overflow_rebased: true` and reports the new
  `cum_since_step`. At the observed peak per-cell rate (~530 hits/step)
  that is ~4 x 10⁶ steps ≈ 23 days of continuous decode.

`cum_count` is opt-in (`?include=cum`) because it is the heaviest array on the
wire: base64 u32 at gzip -1 measures 67,030 B at ~10⁶ magnitude and 78,512 B at
~10⁹, versus 32,785 B for the whole default response.

### 7.5 `scope: "collector"` — why it is gated and what it breaks

Zeroing `_count_win`, `_mass_win`, `_win_pos` and `_windows_rolled` on every
rank makes `decayed()` return all zeros (`stats.py:326`: `n = min(rolled,
window_len)` → `n = 0` → the zero fast-path at `:328-330`). **The policy loop
reads exactly those arrays** — `_read_stats` (`loop.py:661-667`) is called at
the top of every `run_interval` (`loop.py:709`) — so:

* **Swaps: mostly held, but not guaranteed.** `score()` (`policy.py:69-78`) is
  `(eps_k3 - eps_k4) · mass^β · count^α`, which is 0 everywhere, and the
  hysteresis guard `s_in > 1.25 · s_out` (`policy.py:141-142`, default
  `hysteresis=1.25` at `loop.py:120`) is `0 > 0` → `False`. Unpinned swaps are
  therefore suppressed. **But pin-forced pairs bypass hysteresis**
  (`policy.py:142` — the guard is skipped when `forced_in or forced_out`; the
  gap is then set to `inf` at `policy.py:144`). On a serve where the admin API
  has set pins — exactly the surface being built in parallel — a zero can emit
  real, `inf`-priority swaps whose *partner* was chosen by index order rather
  than by traffic, because with all-zero scores the desired set collapses to
  `lexsort((arange(E), -0))` = experts `0..n_k4-1` (`policy.py:118-126`).
* **The router-shift guard is poisoned for two intervals.** `_desired_sets`
  (`loop.py:678-686`) uses the same lexsort, so the desired set jumps to the
  lowest-numbered experts, Jaccard against the previous set collapses far below
  the `0.95` floor (`loop.py:123`, `loop.py:717-730`), and *all* proposals are
  held. The next interval compares against that garbage set, so it is held
  again. Two intervals — 200 steps here, 6,000 at the shipped default — during
  which the policy is frozen and `fq_jaccard` reads near zero in Prometheus.
* **The window needs ~640 steps to be trustworthy again** (and 2,048 to be
  full). Until then the policy scores on a partially-refilled ring: shape is
  roughly preserved, magnitudes are biased low.
* `fq_tier_occupancy` is computed from `tier_of`, not from the counters
  (`loop.py:945-950`), so the gauge does not move *directly* — it moves
  *because* the policy made a decision on zeroed input. The chain is
  counters → `score` → `decide` → `tier_of` → `fq_tier_occupancy`, and a
  destructive reset injects a discontinuity at the head of it.

So: third gate `VLLM_FQ_HEATMAP_ALLOW_COLLECTOR_ZERO=1`, token required, a
`logger.warning` naming the actor, `409 collector_zero_not_allowed` when the
gate is off, and the response body carries the warning verbatim:

```json
{"scope": "collector", "applied": true, "ranks": 4,
 "warnings": ["the policy loop reads these counters (loop.py:661-667); the next
   1-2 intervals will score on a zeroed window, the Jaccard router-shift guard
   will hold all proposals, and pin-forced pairs bypass hysteresis and may swap
   against index order rather than traffic"]}
```

It must be a `collective_rpc` to **all** ranks — zeroing rank 0 only would make
the ranks' policy inputs disagree, and T6 cross-rank decision agreement
(`loop.py:694-698`) would start failing.

### 7.6 What the page does, in one sentence

**The page uses `scope: "client"` by default** — it snapshots the decoded
`count` array, labels the view "change since mark", and never issues a
mutating call; the "reset for everyone" button calls `scope: "heatmap"` and the
`collector` scope is not wired to any UI control at all.

### 7.7 Divergence from `DESIGN.md` §6.2 — reconcile before implementing

The sibling visual design (`./DESIGN.md`, commit `585289afd`) specifies the
live view's Reset control as "**zeroes the collector accumulators**… it is
destructive and affects any concurrent consumer of the stats", with a
confirmation dialog. That is this spec's `scope: "collector"`, which §7.5
argues against and puts behind a third env gate.

The two documents agree on the *hazard* — `DESIGN.md` names the concurrent
consumer explicitly — and differ only on the default. Everything else in
`DESIGN.md` §6.2 is served better by `scope: "client"` / `scope: "heatmap"`
than by a collector zero:

* "stamp the figure with *window since `<timestamp>` / `<n>` intervals*" — the
  envelope already carries `sampled_at_unix_ms`, `interval`,
  `window.rolled` and (with `scope: "heatmap"`) `cum_since_step`, so the stamp
  is exact without destroying anything.
* "the picture immediately after reset is pale and converges as traffic
  accumulates, which correctly communicates low confidence" — this is
  *precisely* what `include=cum` after `scope: "heatmap"` gives: the cumulative
  starts at literal zero and fills in. A collector zero gives the same
  visual while also blinding the policy for two intervals (§7.5).
* The compare-mode metric `256·share_B − 256·share_A` (`DESIGN.md` §5.4) is a
  difference of two samples, i.e. the client-side baseline of §7.2 — it needs
  no reset at all, and a destructive reset actively removes the ability to
  hold more than one baseline.

**Recommendation:** `DESIGN.md` §6.2's Reset should bind to `scope: "heatmap"`
(with `include=cum`), keeping its confirmation dialog and its
"do not rescale the colour domain" rule, both of which stand unchanged. If the
operator still wants the collector zero, it stays available behind
`VLLM_FQ_HEATMAP_ALLOW_COLLECTOR_ZERO=1` and is not a UI control.

---

## 8. Polling vs SSE vs WebSocket

**Recommendation for v1: plain HTTP polling, `GET` every 2 s, with `ETag`.**

Reasons, specific to this system rather than general:

1. **The data is piecewise constant on a 15 s cadence** (§1.3). A push
   transport's advantage is delivering change the instant it happens; here
   "the instant it happens" is once every ~32 engine steps, and a 2 s poll
   already resolves that 7.5x over. SSE would spend its life sending
   keep-alives.
2. **Every sample stalls the engine** (§2.3). A push stream tempts you into
   sampling on a server-side timer whether or not anyone is looking; polling
   makes the cost strictly proportional to demand, and the TTL cache caps it.
   With SSE the natural implementation is a background task that samples
   forever after the first subscriber connects — the exact failure mode we do
   not want on a serve that is also running an encode campaign.
3. **`ETag` + `304` already gives push-like efficiency** (~2.2 KB/s) with none
   of the state. There is no reconnect logic, no last-event-id replay, no
   heartbeat, no proxy buffering problem, and `curl` works.
4. **The client is a heatmap that redraws whole frames.** There is no
   incremental-update structure to exploit; a partial diff protocol would have
   to ship most of 19,200 cells anyway.
5. **It matches every other dev endpoint in the tree.** `/fq/state`,
   `/server_info`, `/is_sleeping` and `/metrics` are all pull. Prometheus
   already scrapes this serve on a pull cadence; a heatmap that is also pull is
   one less transport to operate.

When to revisit: if someone wants sub-roll fidelity at high engine throughput
(a serve running at 40 steps/s rolls every 0.8 s, and a 2 s poll would start
*undersampling* the decayed view), SSE at `GET /fq/heatmap/stream` becomes
worth it. Ship v1 pull; the response schema is unchanged if it ever becomes an
SSE `data:` frame, which is the point of specifying the envelope rather than
the transport.

---

## 9. Worked example

### 9.1 The curl

```bash
# 1. shape and gates first — ~400 B, no arrays
curl -sS --compressed http://127.0.0.1:8000/fq/heatmap/meta | jq .

# 2. one sample, default encoding (count + tier; mass aliased so omitted)
curl -sS --compressed -D /tmp/h.txt http://127.0.0.1:8000/fq/heatmap -o /tmp/s.json
grep -i '^etag' /tmp/h.txt
wc -c /tmp/s.json

# 3. the polling loop the page runs: revalidate with the ETag
ETAG=$(grep -i '^etag:' /tmp/h.txt | cut -d' ' -f2- | tr -d '\r')
curl -sS --compressed -w '%{http_code} %{size_download}\n' \
     -H "If-None-Match: $ETAG" \
     http://127.0.0.1:8000/fq/heatmap -o /dev/null
# -> 304 0        (six times out of seven at a 2 s poll)

# 4. live view + exact cumulative, with a token set
curl -sS --compressed \
     -H 'X-FQ-Heatmap-Token: '"$FQ_HEATMAP_TOKEN" \
     'http://127.0.0.1:8000/fq/heatmap?include=live,cum&precision=bf16' | jq 'del(.count,.mass,.tier,.live_count,.cum_count)'

# 5. mark a baseline (non-destructive, the default)
curl -sS -X POST http://127.0.0.1:8000/fq/heatmap/reset \
     -H 'Content-Type: application/json' \
     -d '{"scope":"client","reason":"before the coder workload"}' | jq .

# 6. the shared, still-harmless server-side reset
curl -sS -X POST http://127.0.0.1:8000/fq/heatmap/reset \
     -H 'Content-Type: application/json' \
     -H 'X-FQ-Heatmap-Token: '"$FQ_HEATMAP_TOKEN" \
     -d '{"scope":"heatmap","reason":"operator reset button"}' | jq .
```

Environment on the serve (`serve-glm52.sh` already sets the first):

```bash
export VLLM_SERVER_DEV_MODE=1
export VLLM_FQ_HEATMAP=1
export VLLM_FQ_HEATMAP_TOKEN=$(openssl rand -hex 16)   # optional
```

### 9.2 A real response (arrays truncated)

Field values below are the real ones from `stats.jsonl` record index 9
(`step=19100`, `interval=191`) and the live serve's `/metrics`
(`fq_tier_occupancy{layer="3",tier="k3"} 256.0`):

```jsonc
{
  "schema": "fq-heatmap/1",
  "sample_id": 137,
  "server_boot_id": "b1f0c8a2",
  "sampled_at_unix_ms": 1786455301123,
  "cached": false,
  "cache_age_ms": 0,

  "step": 19100,
  "real_steps": 19094,
  "interval": 191,
  "window": {"len": 64, "stride": 32, "decay": 0.95, "rolled": 596,
             "n_effective": 64, "steps_since_roll": 12, "horizon_steps": 640},

  "layers": [3, 4, 5, 6, 7, 8, 9, 10, "…62 more…", 76, 77],
  "num_layers": 75,
  "num_experts": 256,
  "cells": 19200,

  "mass_is_real": false,
  "ranks": {"count": 4, "canonical": 0, "agree": true, "reduce": "rank0"},
  "policy_sha": "9c1f4d0b7a2e5c31",
  "apply_mode": "atomic",

  "encoding": {"layout": "layer-major", "byte_order": "little",
               "count": "bf16", "mass": "bf16", "live_count": "bf16",
               "cum_count": "u32", "tier": "u8", "transfer": "base64"},

  // 19200 x bf16 = 38,400 B = 51,200 base64 chars (elided below).
  // The first 6 cells of layer 3 in this record are
  //   27956.67, 14229.64, 10076.52, 35942.18, 20928.84, 19015.03
  // and decode from bf16 as
  //   27904.0,  14208.0,  10048.0,  35840.0,  20992.0,  19072.0
  // i.e. the 0.39 % quantum, invisible on a log colour ramp.
  "count": "…51,200 base64 chars…",

  "mass": null,

  // 19200 x u8, every cell 3 on this K3-uniform serve.
  // 25,600 base64 chars that gzip to 66 bytes.
  "tier": "…25,600 base64 chars…",

  "live_count": null,
  "cum_count": null,
  "cum_since_step": null,
  "cum_lossy": false,

  "warnings": [
    "mass is aliased to count (collector.mass_is_real() == false); set VLLM_FQ_GATE_MASS=1 at boot to record real gate mass"
  ]
}
```

Wire reality for that exact body: **77,435 B of JSON, 32,785 B on the wire**
after `Content-Encoding: gzip` (level 1). The revalidation that follows it:

```
HTTP/1.1 304 Not Modified
ETag: W/"fqhm-1-3a91c7d2e4f6b088"
X-FQ-Sample-Id: 137
X-FQ-Step: 19104
```

### 9.3 Client decode, in full

```js
const CELLS = r.cells;                       // 19200 — validate, do not assume
const count = bf16(r.count, CELLS);          // Float32Array, see §5.2
const tier  = new Uint8Array([...atob(r.tier)].map(c => c.charCodeAt(0)));
if (count.length !== CELLS || tier.length !== CELLS) throw new Error("bad sample");

const at = (row, e) => count[row * r.num_experts + e];   // encoding.layout
const layerId = row => r.layers[row];                    // never row + 3

// smoothed hits/step over the decayed horizon (§5.4a)
const denom = r.window.stride * (1 - r.window.decay ** r.window.n_effective)
            / (1 - r.window.decay);          // 615.99 at the defaults
const rate = i => count[i] / denom;

const mass = r.mass_is_real ? bf16(r.mass, CELLS) : count;   // documented alias
```

---

## 10. Errors

Structured bodies via `JSONResponse`, same envelope as the admin API
(`admin-api-spec.md` §1.4, §7.2):

```json
{"error": {"code": "fq_heatmap_timeout", "message": "...", "details": {}}}
```

| HTTP | `code` | When |
|---|---|---|
| 400 | `unknown_include` | unrecognised `include` token |
| 400 | `bad_layers` | malformed `layers=` selector, or an id not in `state.layers` |
| 400 | `bad_precision` | `precision` not in `{bf16, f32}` |
| 400 | `bad_scope` | reset `scope` not in the four |
| 403 | `fq_heatmap_forbidden` | `X-FQ-Heatmap-Token` missing/wrong |
| 404 | `fq_heatmap_disabled` | `VLLM_FQ_HEATMAP != 1` |
| 404 | `fq_not_active` | no `fq_collector` on the worker |
| 409 | `collector_zero_not_allowed` | `scope: "collector"` without the third gate |
| 501 | `dp_not_supported` | `data_parallel_size > 1` (§4.5) |
| 503 | `engine_unavailable` | engine sleeping / shutting down |
| 503 | `fq_heatmap_worker_error` | a rank raised; message forwarded |
| 503 | `fq_heatmap_poisoned` | a previous sample timed out (§4.4) |
| 504 | `fq_heatmap_timeout` | `collective_rpc` exceeded the deadline |

---

## 11. Environment knobs

| Variable | Default | Meaning |
|---|---|---|
| `VLLM_SERVER_DEV_MODE` | unset | vLLM's gate; the router does not exist without it |
| `VLLM_FQ_HEATMAP` (alias `VLLM_FQ_HEATMAP_API`) | `0` | second gate, specific to this surface |
| `VLLM_FQ_HEATMAP_TOKEN` | unset | if set, require `X-FQ-Heatmap-Token` |
| `VLLM_FQ_HEATMAP_MIN_PERIOD_MS` | `500` | floor on the TTL cache; caps engine-stall rate |
| `VLLM_FQ_HEATMAP_TIMEOUT_S` | `5.0` | `collective_rpc` deadline |
| `VLLM_FQ_HEATMAP_ALLOW_COLLECTOR_ZERO` | `0` | enables the destructive reset scope |
| *read, not set here:* `VLLM_FQ_GATE_MASS` | `0` | whether `mass` is real (`integration.py:90-92`) |
| *read, not set here:* `VLLM_FQ_WINDOW_LEN` / `_STRIDE` / `VLLM_FQ_DECAY` | `64` / `32` / `0.95` | reported in `window`, not overridden |

---

## 12. Files touched

| File | Change |
|---|---|
| `.../exl3_fungible/heatmap.py` | **new** — `worker_sample`, the cumulative accumulator, `build_router`/`attach_router`, `FqHeatmapWorker` mixin, encoders |
| `vllm/entrypoints/serve/__init__.py` | +8 lines in `register_vllm_dev_api_routers`, immediately after the FQ admin block (lines 65-77), same double-gate shape |
| `vllm/v1/worker/gpu_worker.py` | +6 lines: one lazy delegate `fq_heatmap_sample`, after `fq_admin_apply` (`:843-848`) |
| `tests/exl3_fungible/test_heatmap_cpu.py` | **new** |

**Not touched:** `admin.py`, `policy.py`, `loop.py`, `occupancy_table.py`,
`stats.py`, `swap.py`, `store.py`, `fragments.py`. The endpoint reads the
collector through its existing public surface (`decayed`, `mass_is_real`,
`count_buf`) plus the documented window bookkeeping, and reuses the admin API's
rank helpers by import. If an implementation finds itself editing `stats.py`,
that is the signal it has drifted from the design.

---

## 13. Tests that must land with the code

CPU-only, `tests/exl3_fungible/test_heatmap_cpu.py`, no built `vllm._C`, driven
with FastAPI `TestClient` against `build_router()` (the pattern
`test_admin_cpu.py` established, `admin-api-spec.md` §I.7).

**Fixtures come from the real dumps** —
`results/k3-fq/stats-INVALID-truncated-corpus.jsonl` is explicitly fair game for
shape (its contents are from a broken replay and are never asserted on), and
`stats.jsonl` record 9 for values. Ship a small fixture loader that reads one
line and builds a fake collector whose `decayed()` returns those rows.

| Requirement | Test |
|---|---|
| shape | 75 layers `[3..77]`, 256 experts, `cells == 19200`; every array decodes to exactly `cells`; a doctored 19,199-cell array is rejected by the client-side validator test |
| layer ids are data | a fixture with layers `[5, 9, 40]` renders correctly; nothing assumes `row + 3` |
| `mass_is_real` | flag comes from `collector.mass_is_real()`, **not** from `array_equal(count, mass)`; a fixture where a *uniform router* makes count == mass while `mass_is_real() is True` still ships `mass`; the archived dumps (field absent) load as `False` |
| aliased mass omitted | `mass_is_real == false` → `mass is None` and a warning is present; response is ~half the bytes of the `mass_is_real == true` case |
| bf16 round-trip | encode/decode of the real record 9 → max relative error < 0.5 % (asserted against the measured 0.389 %); `f32` path < 1e-6 |
| byte budget | the default envelope for record 9 gzips to < 40,000 B; the `live,mass` envelope to < 110,000 B — regression guards on the numbers in §5.3 |
| tier encoding | 19,200 `u8` of `3` gzips to < 200 B; values outside `occupancy_table.TIERS` raise |
| ETag | two samples with the same `windows_rolled`/`win_pos`/tier produce the same weak ETag; bumping `windows_rolled` changes it; `If-None-Match` → `304` with an empty body; `include=live` makes every ETag distinct |
| rate maths | `hits_per_step` from the §5.4a formula matches a synthetic constant-rate collector to 1 %; the `steps_since_roll == 0` guard |
| single-flight | 10 concurrent `GET`s with `max_age_ms=1000` issue exactly **one** `collective_rpc`; `max_age_ms=0` still respects `MIN_PERIOD_MS` and reports the clamp |
| rank merge | 4 fake ranks, rank 0 canonical + 3 digests → `agree: true`; a mismatched digest → `200`, `agree: false`, warning naming the rank, arrays still served; `reduce=all` returns 4 array sets |
| rank failure | a rank returning `ok: false` → `503 fq_heatmap_worker_error` with the message forwarded; nothing cached |
| timeout poisons | a `collective_rpc` raising `TimeoutError` → `504`; the **next** request → `503 fq_heatmap_poisoned` with **no** RPC issued; `scope: "poison"` clears it |
| gates | truth table over `VLLM_SERVER_DEV_MODE` x `VLLM_FQ_HEATMAP` x token; routes absent for every off-state; wrong token → `403`; `_loop_state` absent → `404 fq_not_active`; DP > 1 → `501` |
| collector-only mode | a bare `FqStatsCollector` (no loop) still serves `count`, with `tier`/`interval`/`policy_sha` null and no crash |
| reset: client | changes **nothing** server-side — assert `collector._count_win`, `_windows_rolled`, `tier_of` and `policy_sha` are byte-identical before and after, and that no RPC was issued |
| reset: heatmap | rebases `cum_count` on all ranks; the collector window is untouched; the next `include=cum` sample starts from 0 at the new `cum_since_step` |
| reset: collector | refused `409` without the third gate; with it, all ranks zeroed; **and a regression test that drives `policy.decide` on the zeroed stats and asserts (a) no unpinned swaps, (b) a pin-forced pair DOES emit — the §7.5 hazard, pinned in a test so nobody "fixes" the warning away** |
| cumulative exactness | synthetic ring: integrating over `n_new <= window_len` reproduces the exact int64 sum; `n_new > window_len` sets `cum_lossy` and warns; the `2**31` auto-rebase path |
| no tearing | a fake collector that increments `_windows_rolled` between the pre-read and post-read forces exactly one retry, and the returned sample is internally consistent |
| fixture shapes | all four archived `.jsonl` files parse into a valid sample (including the INVALID one — shape only, no value assertions) |

---

## 14. Open questions for the implementer

1. **`layers=` subsetting and the ETag basis.** Specified as part of the basis,
   so two different subsets never collide. If subsetting turns out unused,
   drop the parameter rather than carrying an untested code path.
2. **Should `/fq/heatmap/meta` cost an RPC?** Specified as: one on first call
   (to learn `layers`, `num_experts`, `mass_is_real` from a worker), cached for
   the process lifetime thereafter except `fq_active`. If the loop can change
   `layers` at runtime — it cannot today, `state.layers` is fixed at
   construction (`loop.py:491`) — this needs revisiting.
3. **`ranks.agree == false` under TP.** Never observed; the design says surface
   and continue. If it turns out to happen routinely for a benign reason (a
   dummy-step race on one rank, say), downgrade the warning before it trains
   operators to ignore it.
4. **Bit-exactness of the digest across ranks.** The digest is over the bf16
   bytes, so two ranks whose float64 accumulations differ in the last mantissa
   bit still agree. That is intentional. If a stricter check is ever wanted,
   digest the f32 bytes instead and accept the false positives.
