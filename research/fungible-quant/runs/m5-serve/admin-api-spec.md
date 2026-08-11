# Admin API for forced expert re-tiering — implementation spec

Status: **design, ready to implement**. No code written. Author: design pass
2026-08-11. Target tree: `gg-vllm` branch `fq/m1-stats-collector`.

The operator's ask, verbatim:

> can we expose (optionally) an admin api that would be exposed by vLLM (same
> principle as sleep/pause/resume?) that would allow to force upgrade/downgrade
> of specific experts K -- for example layer=23,expert=250,adjust_k=-1
> (relative, or absolute like adjust_k=3) and as a batch also doing
> layer=23,expert=1,adjust_k=+1 .. this being guarded by maximum memory usage,
> obviously.

Answer: yes, and every piece it needs already exists. `SwapEngine` does the
atomic trade, `PolicyStore` does the persistence, `FungibleQuantState` does the
loop bookkeeping, `occupancy_table` does the operator-facing view. The endpoint
is glue plus guards. What it must **not** do is invent a second way to move an
expert between tiers.

---

## 1. The vLLM precedent

### 1.1 The dev-mode gate

`build_app` registers the always-on routers unconditionally, then:

```
vllm/entrypoints/openai/api_server.py:244-247
    if envs.VLLM_SERVER_DEV_MODE:
        from vllm.entrypoints.serve import register_vllm_dev_api_routers
        register_vllm_dev_api_routers(app)
```

`register_vllm_dev_api_routers` (`vllm/entrypoints/serve/__init__.py:35-61`)
logs a `SECURITY WARNING: Development endpoints are enabled! This should NOT be
used in production!` at lines 36-39 and then attaches five sub-routers, each a
`vllm/entrypoints/serve/dev/<name>/api_router.py` exporting
`attach_router(app: FastAPI)`:

| Router | File | Routes |
|---|---|---|
| cache | `serve/dev/cache/api_router.py:20,47,58` | `/reset_prefix_cache`, `/reset_mm_cache`, `/reset_encoder_cache` |
| rlhf | `serve/dev/rlhf/api_router.py:29,74,94,139,157,175,187,205,211` | `/pause`, `/resume`, `/abort_requests`, `/is_paused`, weight-transfer set |
| rpc | `serve/dev/rpc/api_router.py:23` | `/collective_rpc` |
| server_info | `serve/dev/server_info/api_router.py:43` | `/server_info` |
| sleep | `serve/dev/sleep/api_router.py:21,32,45` | `/sleep`, `/wake_up`, `/is_sleeping` |

Every one of them is the same five-line shape: module-level
`router = APIRouter()`, a local `def engine_client(request) -> EngineClient:
return request.app.state.engine_client`, decorated handlers, and
`def attach_router(app): app.include_router(router)`. Our router copies that
shape exactly.

### 1.2 How a request reaches the workers

Two mechanisms are in the tree, and they are different things:

**(a) Purpose-built `EngineClient` methods.** `/sleep` calls
`engine_client(raw_request).sleep(int(level), mode)`
(`serve/dev/sleep/api_router.py:26`); `/pause` calls
`engine.pause_generation(mode=..., clear_cache=..., ...)`
(`serve/dev/rlhf/api_router.py:51-55`). These require adding a method to the
`EngineClient` protocol (`vllm/engine/protocol.py`) and to `AsyncLLM`.

**(b) `collective_rpc` — the generic path.** `EngineClient.collective_rpc`
(`vllm/engine/protocol.py:227-235`) →
`AsyncLLM.collective_rpc` (`vllm/v1/engine/async_llm.py:964-976`) →
`self.engine_core.collective_rpc_async(method, timeout, args, kwargs)`. The
method name is looked up **on the worker object**, so any method on `Worker`
(or on a class injected via `--worker-extension-cls`) is callable.

Worker-extension injection: `parallel_config.worker_extension_cls`
(`vllm/config/parallel.py:262`, CLI `--worker-extension-cls` at
`vllm/engine/arg_utils.py:1153`) is resolved and *dynamically added to
`Worker.__bases__`* at `vllm/v1/worker/worker_base.py:261-286`, after asserting
no attribute collides with the worker class.

**We use (b).** Re-tiering is FQ-specific and must not widen the `EngineClient`
protocol. See §6 for exactly where the worker methods live.

### 1.3 Response aggregation

`collective_rpc` returns a **list, one entry per worker**. The reference
aggregation is minimal (`serve/dev/rpc/api_router.py:46-54`): `None` → bare
`Response(200)`; otherwise each result is passed through if it is `dict`/`list`,
else `str()`-ed, and wrapped as `{"results": [...]}`. Note the security comment
at lines 38-41 — the generic route only forwards *serialized string* args, and
the callee owns deserialization. We keep that convention (JSON strings in and
out) even though our router calls `collective_rpc` directly rather than through
the HTTP route.

Our endpoint does **more** than pass through: it must verify all ranks agree
(§7.3). Divergence between TP ranks after a weight mutation is a serve-down
condition, not something to hand back as an opaque list.

### 1.4 Error conventions

Both styles are present and both are accepted upstream:

* `raise HTTPException(status_code=..., detail="...")` —
  `serve/dev/rpc/api_router.py:28-37`, `serve/elastic_ep/api_router.py:46-85`.
* `return JSONResponse(content={"error": str(err)}, status_code=...)` —
  `serve/dev/rlhf/api_router.py:61-71, 86-91, 131-136` (the `/pause` and
  `/resume` the operator named).

`scale_elastic_ep` is the closest analogue to what we are building — a guarded,
draining, mutating admin action — and it is worth copying wholesale:

```
vllm/entrypoints/serve/elastic_ep/api_router.py:32-41
@router.post("/scale_elastic_ep",
    dependencies=[Depends(validate_json_request)],
    responses={HTTPStatus.OK.value: {"model": dict},
               HTTPStatus.BAD_REQUEST.value: {"model": ErrorResponse},
               HTTPStatus.REQUEST_TIMEOUT.value: {"model": ErrorResponse},
               HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": ErrorResponse}})
```

It also demonstrates the in-flight flag pattern (`set_scaling_elastic_ep(True)`
… `finally: set_scaling_elastic_ep(False)`, lines 68/87) and a 408 on drain
timeout (lines 77-82). `validate_json_request` is
`vllm/entrypoints/serve/utils/api_utils.py:348-354` (rejects non-`application/json`).

**Convention for this API:** structured error bodies via `JSONResponse` (the
`/pause` style), because an admin API is machine-driven and a bare `detail`
string is not parseable:

```json
{"error": {"code": "cardinality_unbalanced",
           "message": "layer 23: 1 promotion, 0 demotions",
           "details": {"layer": 23, "promotions": [1], "demotions": []}}}
```

---

## 2. Route surface

New router `vllm/entrypoints/serve/dev/fq/api_router.py`, attached from
`register_vllm_dev_api_routers` (`serve/__init__.py`) after the sleep router.
`APIRouter(prefix="/fq")` — the FQ family is large enough to earn a namespace,
unlike the flat single-purpose dev routes.

| Method | Route | Mutating | Purpose |
|---|---|---|---|
| `GET` | `/fq/state` | no | policy sha, per-layer budget, occupancy, memory accounting, pins, in-flight flag |
| `GET` | `/fq/layer/{layer}` | no | per-expert current K, live score rank, dwell, pin — the table you read before choosing a demotion partner |
| `POST` | `/fq/retier` | **yes** | the operator's endpoint |
| `POST` | `/fq/pins` | yes (policy only) | set/clear pins without moving weights |

Gating, in order, all evaluated before the body is read:

1. `envs.VLLM_SERVER_DEV_MODE` — the router does not exist otherwise.
2. `VLLM_FQ_ADMIN_ENABLE=1` — a **second** gate. Dev mode is enabled for lots of
   reasons; forced weight mutation should require saying so. When unset the
   routes return `404 {"error": {"code": "fq_admin_disabled"}}`.
3. Optional `VLLM_FQ_ADMIN_TOKEN` — if set, require header
   `X-FQ-Admin-Token` to match (constant-time compare); else `403
   fq_admin_forbidden`.
4. FQ must actually be running: `VLLM_FQ_ENABLE=1` and the workers must report a
   live `FungibleQuantState` with a bound `SwapEngine`. Otherwise `404
   fq_not_active`.

---

## 3. `POST /fq/retier` — request schema

```jsonc
{
  "items": [                                   // required, 1..VLLM_FQ_ADMIN_MAX_ITEMS
    {"layer": 23, "expert": 250, "adjust_k": -1},
    {"layer": 23, "expert": 1,   "adjust_k": "+1"}
  ],
  "mode": "strict_pair",                       // strict_pair (default) | auto_balance | grow_budget
  "dry_run": false,                            // default false
  "pin": "hold",                               // hold (default) | none | release
  "drain_mode": "wait",                        // wait (default) | abort | keep
  "reset_prefix_cache": false,                 // default false
  "expect_policy_sha": null,                   // optional optimistic-concurrency guard
  "timeout_s": 120,                            // default VLLM_FQ_ADMIN_DRAIN_TIMEOUT_S
  "actor": "michel",                           // optional; also accepted as X-FQ-Actor header
  "reason": "expert 1 is hot on the coder axis"// optional but strongly encouraged
}
```

Unknown top-level or item keys → `400 unknown_field`. Empty `items` with
`pin != "none"` is allowed (pure pin change, no swap); empty `items` with
`pin == "none"` → `400 empty_request`.

### 3.1 Single-item query-string shorthand

Because the operator wrote query-parameter syntax and `/sleep` sets that
precedent (`serve/dev/sleep/api_router.py:24-25`), accept:

```
POST /fq/retier?layer=23&expert=250&adjust_k=-1
```

with **no body**. It desugars to `{"items": [{"layer": 23, "expert": 250,
"adjust_k": "-1"}], "mode": "strict_pair", ...defaults}`. Query params and a
body together → `400 mixed_input`. Note this form will almost always fail with
`cardinality_unbalanced` (§4) — that is correct and instructive; a lone
promotion or demotion is not a legal operation.

### 3.2 `adjust_k`: relative vs absolute

This is the one genuinely ambiguous part of the ask, and JSON makes it worse:
`+1` is **not a legal JSON number**, so `{"adjust_k": +1}` cannot be sent.
Disambiguation rule, chosen so the operator's literal examples both work:

| Sent | Type | Interpretation | Why unambiguous |
|---|---|---|---|
| `-1` | number, negative | **relative** `delta_k = -1` | no negative K exists |
| `3` | number, positive | **absolute** `k = 3` | matches "absolute like adjust_k=3" |
| `"+1"` | string with `+` | **relative** `delta_k = +1` | explicit sign |
| `"-1"` | string with `-` | **relative** `delta_k = -1` | explicit sign |
| `"3"` | string, bare digits | **absolute** `k = 3` | no sign |
| `1`, `0` | number, positive, not in ladder | **`400 ambiguous_adjust_k`** | K1/K0 do not exist; the operator probably meant `"+1"` |
| `0.5`, `"x"` | anything else | `400 bad_adjust_k` | — |

The `400 ambiguous_adjust_k` message must say, literally: *"adjust_k=1 is not a
valid absolute tier (ladder is K3/K4). For a relative +1, send the string
\"+1\", or use the explicit field delta_k: 1."*

Machine clients should skip `adjust_k` entirely and use the explicit fields:

* `"k": 4` — absolute target.
* `"delta_k": -1` — relative.

Exactly one of `adjust_k` / `k` / `delta_k` per item; zero or two+ →
`400 bad_adjust_k`.

### 3.3 Item resolution

For each item, in order:

1. `layer` must be a key of the running policy's `bits_per_expert` and a
   registered `SwapEngine` layer → else `404 layer_not_registered`.
2. `0 <= expert < num_experts` → else `400 expert_out_of_range`.
3. `(layer, expert)` must be unique across `items` → else `400 duplicate_item`
   (`SwapPlan.__init__` would reject it anyway at `swap.py:418-426`, but the API
   must say so first, with both indices).
4. `k_from = tier_of[row(layer)][expert]` — read from the **live** loop state,
   not from anything the client sent.
5. `k_to` = `k` (absolute) or `k_from + delta_k` (relative).
6. `k_to == k_from` → the item is a **no-op**. It is not an error; it is dropped
   from the plan and reported as `"outcome": "noop"`. Balance in §4 is evaluated
   on the surviving items, so a no-op demotion turning a balanced pair into a
   lone promotion produces `409 cardinality_unbalanced` with the no-op called
   out in `details`.
7. `k_to ∉ {3, 4}` → `501 tier_not_servable`. Two independent reasons, both
   stated in the message:
   * `MixedLayerState.from_exl3_mixed_trellis` refuses anything but `(3, 4)`
     (`swap.py:514-517`) — the v1 swap engine is a two-tier engine.
   * K5 as a mixed tier does not fit SM120 shared memory at all
     (109568 > 101376; K4 is exactly 101376, zero headroom) —
     `runs/m5-serve/k5-shared-memory-limit.md`.

---

## 4. Cardinality: the semantics decision

### 4.1 What the code actually enforces

Occupancy == capacity is not one check; it is **four independent checks in
three modules**, all fail-closed:

1. `store.validate_policy` (`store.py:48-51`): for any layer present in
   `budget.n_k4_per_layer`, `sum(b == 4) != cap` → `ValueError("K4 occupancy !=
   declared n_k4_per_layer (v1 requires cap == n)")`. This fires at
   `PolicyStore.commit` (`store.py:78`) and at `load_current` (`store.py:99`).
2. `SwapPlan.from_memberships` (`swap.py:441-446`): `len(outs) != len(ins)` per
   layer → `ValueError("same-cardinality plans only (D1)")`.
3. `SwapEngine._validate_layer` (`swap.py:713-718`): the two tier orderings must
   be a partition of `[0, num_global_experts)` — "v1 requires occupancy ==
   capacity (holes go through global_to_combined, not here)".
4. `policy.decide` (`policy.py:104-107`): membership K4 count must equal the
   budget or it refuses to decide at all.

### 4.2 Why "raise the budget if memory allows" is not implementable

This is the option that looks most attractive and is in fact **structurally
impossible at runtime**, for three separate reasons. Recording all three so
nobody re-opens it:

* **The slabs are sized at prepare time.** `SwapEngine._validate_layer` derives
  `w13_words = 2 * count * h16 * i16 * 16 * bits` and `w2_words = count * i16 *
  h16 * 16 * bits` from `count = len(tierN_globals)` and refuses a mismatch
  (`swap.py:727-738`). Growing the K4 tier by one expert needs one more slab
  row, i.e. a **reallocation** of `tier1.w13` / `tier1.w2`. The whole swap engine
  is built on "no reallocation, no recompile" (`swap.py:6-9`).
* **The kernel is compiled against the per-tier counts.** The mixed-runtime memo
  key includes `tier_signature = ((bits, len(ids)), ...)`
  (`exl3.py:1840-1843`, used in the key at `exl3.py:1851`), and
  `compile_mixed_trellis` is called with `tier0_num_experts` /
  `tier1_num_experts` (`exl3.py:1884-1899`). Changing a count is a **recompile**,
  which `_mixed_rank_sliced_runtime` refuses to do under CUDA-graph capture
  (`exl3.py:1859-1864`) and which on SM120 re-enters the tile-fit trap that
  blocks K5.
* **The bytes are not there anyway.** The KV-cache pool is sized from
  post-load free memory at startup. Weights already sit at ~77-80% of a 95.6 GiB
  card (the mixed build loaded 77.83 GiB/rank —
  `k5-shared-memory-limit.md`), and the reference envelope is 73.0187 GiB/rank
  with headroom over all-K3 of exactly 8,042 promotions, i.e. exactly the
  reference's own K4 count (`runs/m5-serve/convergence-demo-plan.md:117-131`).
  There is no slack by construction, and any slack there were belongs to the KV
  cache, which cannot be clawed back without a reload.

**Therefore:** `mode: "grow_budget"` is a recognised value that returns
`501 budget_growth_not_supported` with the three reasons above and a pointer to
the reload path (restart with a policy whose `n_k4_per_layer` is larger; the
committed policy steers the next boot — §5.3).

### 4.3 Recommendation

| Mode | Behaviour | Ship it? |
|---|---|---|
| **`strict_pair`** (default) | Per layer, `#promotions == #demotions` required. Fully explicit; the resolution is a pure function of the request plus the current membership, so every rank derives an identical plan with no coordination. | **Yes — this is the primary mode.** |
| `auto_balance` | Unpaired promotions are matched against the coldest resident K4 experts in the same layer (lowest `policy.score`), unpaired demotions against the hottest resident K3 experts. Requires `VLLM_FQ_ADMIN_ALLOW_AUTO_BALANCE=1`. | Yes, phase 2 — see §4.4. |
| `grow_budget` | `501`. | No. |

Justification for `strict_pair` as default: forced re-tiering is a *manual
override*. An override that silently makes a second, unrequested policy decision
(which expert to sacrifice) is exactly the kind of surprise an operator issues an
override to avoid. The operator's own example is already balanced — they
intuited the right semantics. `GET /fq/layer/23` exists so choosing the partner
is a two-second lookup, not guesswork.

`409 cardinality_unbalanced` must be actionable:

```json
{"error": {"code": "cardinality_unbalanced",
  "message": "layer 23: 1 promotion (e1: K3->K4), 0 demotions. Fixed-cardinality invariant requires a 1-for-1 trade.",
  "details": {"layer": 23, "n_k4_per_layer": 108,
              "promotions": [{"expert": 1, "k_from": 3, "k_to": 4}],
              "demotions": [],
              "noop_items": [],
              "remedies": [
                "add a demotion in layer 23 (see GET /fq/layer/23 for the coldest K4 residents)",
                "retry with mode=auto_balance (requires VLLM_FQ_ADMIN_ALLOW_AUTO_BALANCE=1)",
                "raise n_k4_per_layer['23'] and restart — runtime budget growth is not supported (see /fq/state.budget.growth_supported)"
              ]}}}
```

### 4.4 `auto_balance` mechanics

The partner choice depends on live routing statistics, which live in the worker
processes and are **per-rank**. Letting each rank pick its own partner would
break cross-rank agreement (the T6 property,
`loop.py:454-457`, `tests/exl3_fungible/test_cross_rank_t6_cpu.py`). So
`auto_balance` is a **two-phase RPC** and the mutating phase is always fully
explicit:

1. `collective_rpc("fq_admin_plan", args=(request_json,))` — every rank resolves
   items and returns its view: current membership for touched layers, pins,
   its own candidate partner ranking, fragment availability, memory accounting,
   and a `plan_sha`. **No mutation.**
2. The router takes **rank 0's** resolved plan as authoritative, checks the other
   ranks reported the same current membership (else `500 rank_divergence`), and
   sends that exact explicit plan:
   `collective_rpc("fq_admin_apply", args=(plan_json,))`.

`strict_pair` uses the same two calls — phase 1 is the pre-flight (availability
+ memory + validation + `dry_run`), phase 2 applies. Uniform code path, and
`dry_run: true` is simply "stop after phase 1".

---

## 5. Guards

### 5.1 Memory guard

Unit cost, from `PLAN.md:155` (`3·H·I/8/TP · bits`) and
`convergence-demo-plan.md:107-131`:

```
UNIT = 1,179,648 bytes per rank, per expert, per layer, per bit of K
K3 expert = 3,542,028 B/rank      K4 expert = 4,721,676 B/rank
K3 -> K4  = +1,179,648 B/rank     K4 -> K3  = -1,179,648 B/rank
reference envelope = 78,403,227,648 B/rank = 73.0187 GiB/rank
```

(`UNIT × 3 = 3,538,944` is the trellis alone; the per-expert figures above add
the per-expert rotation rows. Compute both from the live layer geometry at
runtime — `hidden_size`, `intermediate_size_per_partition`, `tier_bits` — and
report the derived unit in the response so the arithmetic is auditable rather
than hard-coded.)

Guard, evaluated in phase 1 on **every** rank:

```
delta_bytes      = Σ over effective items of UNIT × (k_to − k_from)
projected_bytes  = current_expert_bytes_per_rank + delta_bytes
budget_bytes     = VLLM_FQ_ADMIN_MEM_BUDGET_BYTES
                   (default: the boot-time resident expert payload per rank —
                    i.e. "no growth", which fixed cardinality already implies)

REFUSE 409 memory_budget_exceeded  if projected_bytes > budget_bytes
REFUSE 409 memory_headroom_exceeded if delta_bytes > 0 and
        min_over_ranks(torch.cuda.mem_get_info()[0]) − delta_bytes
            < VLLM_FQ_ADMIN_MEM_HEADROOM_BYTES        (default 1 GiB)
```

Under `strict_pair` `delta_bytes` is **exactly 0** by construction, so the guard
never fires. Evaluate and report it anyway: an admin API that asserts an
invariant it never checks is an admin API that will violate it the first time
someone adds a mode. The response carries the full accounting (§7.2) so the
operator sees `delta_bytes_per_rank: 0` and knows the trade was clean.

Two costs the guard must also report but does not block on:

* **Pinned host staging.** `SwapEngine.__init__` pre-allocates `max_pairs`
  (K3, K4) `ExpertStage` pairs (`swap.py:656-662`); `fail_atomic` doubles it
  lazily (`swap.py:683-696`). At GLM-5.2 TP4 geometry a pair is ~8.0 MiB, so
  `max_pairs=32` with fail-atomic is ~512 MiB of **pinned host** RAM, allocated
  once and reused. Set `max_pairs = max(8, VLLM_FQ_ADMIN_MAX_ITEMS)` and report
  `pinned_staging_bytes` in `GET /fq/state`.
* **H2D per apply.** `StagedBatch.bytes_h2d` (`swap.py:956-958`) — reported per
  rank in the response.

### 5.2 Pin guard

`pin` field semantics, applied to the touched `(layer, expert)` pairs only:

| Value | Effect on `policy_doc["pinned"]` |
|---|---|
| `"hold"` (default) | add every moved expert, pinning it to its **new** tier |
| `"none"` | leave `pinned` untouched |
| `"release"` | remove every named expert from `pinned` |

Default is `"hold"` on purpose. `loop.py:371-382` pins to the *current* tier, and
`policy.decide` waives dwell and hysteresis for pinned members
(`policy.py:113-140`). Without a pin, the next interval sees the operator's
demoted expert as a high-score K3 candidate and can swap it straight back once
dwell expires (`dwell_steps` defaults to `2 × interval`, `loop.py:108-110`) — an
operator would reasonably file that as a bug. Pinning makes the override mean
what it says, and `pin: "release"` un-does it.

**Starvation guard.** `policy.decide` raises `"layer {l}: pins incompatible with
budget"` when `pin4.sum() > n_k4[l]` or `E − pin3.sum() < n_k4[l]`
(`policy.py:110-111`) — that would take down every subsequent interval (caught
by `step()`'s blanket handler at `loop.py:412-416`, so the loop would silently
stop deciding). Refuse before it can happen:

```
REFUSE 409 pin_would_starve_layer if, after applying the pin change,
    n_k4[l] − count(pin4 in layer l) < VLLM_FQ_ADMIN_MIN_FREE_SLOTS   (default 2)
 or (E − count(pin3 in layer l)) − n_k4[l] < VLLM_FQ_ADMIN_MIN_FREE_SLOTS
```

Default 2 == `max_swaps_per_layer` (`policy.py:37`), so the loop always keeps at
least a full interval's worth of room to work.

### 5.3 Fragment availability — reject, never degrade

The requested K's fragment may not exist. `FragmentResolver.resolve`
(`fragments.py:774-832`) walks `local dirs → cache → HF sources`, then the
`VLLM_FQ_K_FALLBACK` ladder, and **enqueues a lazy encode** on every miss or
substitution (`fragments.py:797-806`). `ResolverFragmentSource.read_expert`
turns a substitution into `FragmentUnavailable` (`swap.py:374-378`), and
`SwapEngine.stage(on_unavailable="drop")` can convert that into a
`DroppedPair` (`swap.py:543-554, 866-883`) so a policy interval pends the
promotion instead of failing.

**For the admin path: reject, atomically. Never accept-and-degrade.** Two
reasons, one of them absolute:

* *Semantic:* the operator asked for a specific expert at a specific K. Silently
  giving them K3 when they asked for K4 is a lie with a 200 status code.
* *Structural — accept-and-degrade is not even representable.* A K3 payload
  cannot be written into a K4 slab row: the row is `16 × bits` int16 words wide
  (`swap.py:121-132`), so the word counts differ. `ResolverFragmentSource`'s own
  docstring says exactly this (`swap.py:338-345`): `allow_substituted=True`
  "would only be meaningful with per-pair tier retargeting, which v1 does not
  have". There is no degrade mode to choose.

So phase 1 runs an explicit **availability pre-flight** over every item's target
K — calling `resolver.resolve(layer, expert, k_to)` and checking
`fragment.k == k_to` — before any staging. This finds *all* misses in one pass
(`stage(on_unavailable="raise")` would stop at the first) and, as a free side
effect, `resolve()` has already queued each miss for lazy encode. Response:

```json
{"error": {"code": "fragment_unavailable",
  "message": "2 of 2 requested tiers are not available; nothing was applied",
  "details": {"unavailable": [
      {"layer": 23, "expert": 1, "k": 4,
       "reason": "resolver substituted K3 (requested encode has not landed)",
       "encode_queued": true, "queue_position": 7,
       "chain": "local:miss; cache:miss; hf:org/repo@main:miss; FALLBACK K3 local:hit"}],
    "retry_after_hint": "drain the encode queue: python -m ...exl3_fungible.lazy_encode"}}}
```

Status `409`, not 404: the resource may exist shortly. Include the resolver's
own decision chain (`fragments.py:836-848` formats it identically for the log)
so the operator can see *which* source was missing.

`dry_run: true` runs exactly this pre-flight plus the memory accounting and the
plan construction, and returns the whole response with `"applied": false`. That
is the operator's "will this work?" button and it should be the documented first
step of any batch.

### 5.4 Concurrency and state guards

* **One retier at a time.** Module-level `asyncio.Lock` in the router (the
  `set_scaling_elastic_ep` pattern, `elastic_ep/api_router.py:68,87`), plus a
  worker-side re-entrancy flag — `SwapEngine` is explicitly "not thread-safe;
  one staged batch is live at a time (staging slots are reused)"
  (`swap.py:616-619`). Contended → `409 retier_in_flight`.
* **Not while the engine is elsewhere.** Refuse with `503 engine_unavailable`
  when `await engine.is_sleeping()` or `get_scaling_elastic_ep()` or
  `await engine.is_paused()` is already true (a pre-existing external pause
  means someone else owns the drain).
* **Optimistic concurrency.** If `expect_policy_sha` is present and differs from
  the live `policy_sha`, `409 policy_sha_mismatch` with both values. Clients
  should `GET /fq/state`, decide, then send the sha they saw — this closes the
  window against a policy interval committing between read and write.
* **DP is out of scope for v1.** If `parallel_config.data_parallel_size > 1`,
  return `501 dp_not_supported`. The per-engine policy stores and the
  cross-engine agreement story are unvalidated; say so rather than half-do it.

---

## 6. Worker plumbing

### 6.1 Where the methods live

Three methods on the `Worker` object, reachable by name through
`collective_rpc`:

```python
def fq_admin_describe(self) -> str          # JSON; read-only
def fq_admin_plan(self, request_json: str) -> str   # JSON; read-only, phase 1
def fq_admin_apply(self, plan_json: str) -> str     # JSON; phase 2
```

JSON strings in, JSON strings out — matches the `/collective_rpc` convention
(`serve/dev/rpc/api_router.py:38-41`) and avoids relying on whatever the
engine-core serialiser does with dataclasses.

**Recommended binding (ship this):** three thin methods added to `Worker` in
`vllm/v1/worker/gpu_worker.py`, each a lazy import + delegate, mirroring the
existing FQ hook at `gpu_worker.py:806-811`:

```python
def fq_admin_plan(self, request_json: str) -> str:
    from vllm.model_executor.layers.quantization.exl3_fungible.admin import (
        worker_plan,
    )
    return worker_plan(self, request_json)
```

This preserves the package's laziness contract (`integration.py:12-22`: nothing
FQ is imported when the feature is off) and keeps all logic in
`exl3_fungible/admin.py`, where it is CPU-testable.

**Documented alternative (do not ship, but record):**
`--worker-extension-cls vllm.model_executor.layers.quantization.exl3_fungible.admin.FqWorkerAdmin`
requires **zero** core-file changes (`worker_base.py:261-286` injects it into
`Worker.__bases__`). Rejected as the default because vLLM supports exactly one
extension class, and consuming it would collide with RLHF users. Keep
`FqWorkerAdmin` exported as a mixin anyway so this route stays open.

### 6.2 Building the live `SwapEngine`

Nothing constructs a `SwapEngine` over live layers today — `loop.py:26-28`
records that `atomic` mode "is not yet bound to live layers". **This spec's
implementation must add that binding**; it is the prerequisite, not an
afterthought. In `exl3_fungible/admin.py`:

```python
def build_swap_engine(model_runner, *, rank: int, max_pairs: int) -> SwapEngine:
    layers = {}
    for module in model_runner.model.modules():
        mixed = getattr(module, "exl3_mixed_trellis", None)   # exl3.py:1692
        if mixed is None:
            continue
        layers[int(module.layer_id)] = MixedLayerState.from_exl3_mixed_trellis(mixed)
    spec = ProgressiveSpec.from_env()                          # progressive.py:102
    source = ResolverFragmentSource(spec.make_resolver())      # progressive.py:231
    return SwapEngine(
        layers, source,
        hidden_size=<layer.exl3_hidden_size>,
        intermediate_size=<layer.exl3_intermediate_size_per_partition>,
        tier_bits=(3, 4), rank=rank, max_pairs=max_pairs,
        expected_mcg=None,          # learned from the first staged fragment
    )
```

Built lazily on the first admin (or first atomic-mode) request and cached on the
loop state as `state.swap_engine`. Construction runs `_validate_layer` on every
layer (`swap.py:650-651`), which is the honest boot gate: a checkpoint whose
rotations use the broadcast layout, or whose tier orderings are not a partition,
fails here with a clear message instead of at the first swap.

`MixedLayerState.from_exl3_mixed_trellis` requires `tier_bits == (3, 4)`
(`swap.py:514-517`), so a K3-uniform serve has no mixed layers at all and
`build_swap_engine` returns an empty map → `404 fq_not_active` with
`"reason": "no mixed-trellis layers; serve is uniform-K"`.

### 6.3 Constructing the `SwapPlan` — drive, don't reimplement

The endpoint must **not** hand-assemble `(layer, e_out, e_in)` triples. Build the
target policy document and let the existing, tested diff do the pairing:

```python
# 1. target document — same shape as loop._doc_for (loop.py:526-538)
new_doc = {k: v for k, v in state.policy_doc.items()
           if k not in ("bits_per_expert", "provenance", "pinned")}
new_tier = state.tier_of.copy()
for layer, expert, k_to in effective_items:
    new_tier[state.layers.index(layer), expert] = k_to
new_doc["bits_per_expert"] = {str(l): [int(b) for b in new_tier[row]]
                              for row, l in enumerate(state.layers)}
new_doc["pinned"] = <pins after applying the `pin` field>
new_doc["provenance"] = {...}          # §8.2

# 2. the plan — one call; does pairing, ordering and the cardinality check
plan = SwapPlan.from_policies(state.policy_doc, new_doc)   # swap.py:450-461

# 3. the fixed-cardinality invariant, independently, before any device write
validate_policy(new_doc, num_experts=state.num_experts)    # store.py:33-54
```

`from_policies` delegates to `from_memberships` (`swap.py:427-448`), which per
layer takes `outs = old==4 & new==3`, `ins = old==3 & new==4`, requires
`len(outs) == len(ins)`, and zips them in ascending expert-id order with layers
ascending — deterministic, identical on every rank, and already covered by
`tests/exl3_fungible/test_swap_cpu.py`. Then it remaps row indices back to model
layer ids (`swap.py:459-461`), so the plan speaks the same layer ids the
`SwapEngine` was registered with. Two of the four cardinality checks fire here;
the third fires inside `stage()`, the fourth at `commit()`.

Cap check before staging: `len(plan) > engine.max_pairs` → `409 plan_too_large`
(otherwise `stage` raises at `swap.py:823-825`).

### 6.4 Staging and applying

```python
staged = engine.stage(plan, fail_atomic=True, on_unavailable="raise")
report = engine.apply(
    staged=staged,
    quiesce=contextlib.nullcontext(),   # the API server already drained — §6.5
    stream=torch.cuda.current_stream(),
    memo_hook=None,                     # not needed — see below
    policy_store=state.store if state.is_lead else None,
    policy_doc=new_doc,
    policy_num_experts=state.num_experts,
)
```

* **`fail_atomic=True` is mandatory** on the admin path. Without it an abort
  before the visibility flip leaves the layer genuinely torn and recoverable only
  by re-applying or restarting (`swap.py:1055-1059`). With it the pre-swap rows
  and maps are restored inside the same quiesce window (`swap.py:1044-1054`) and
  `staged.restored` is `True` — which the response must surface. It costs a
  second read of each expert and doubles pinned staging; for an operator action
  that is obviously the right trade.
* **`on_unavailable="raise"`**, not `"drop"` — §5.3. Availability was already
  pre-flighted, so this is the belt-and-braces path; if it fires anyway, nothing
  was written (staging mutates no live state until the ops list is applied,
  `swap.py:857-858`) and the response is `409 fragment_unavailable`.
* **`memo_hook=None` is correct and worth stating.** The mixed-runtime memo key
  contains `tier_signature` = *(bits, count)* per tier (`exl3.py:1840-1843,
  1851`), not membership. A 1-for-1 trade leaves both counts unchanged, so the
  key is unchanged and the compiled runtime stays valid. T3 proved the maps are
  read as data at replay time, not baked at capture
  (`runs/m4-swap/report.md`, T3 PASS bitwise), and `apply()` writes in place so
  `data_ptr` never moves. **No re-capture, no recompile.** This is a fourth
  independent reason cardinality must be preserved: change a count and the memo
  key changes, forcing `compile_mixed_trellis` — refused under graph capture
  (`exl3.py:1859-1864`).
* **`policy_store` only on the lead rank.** `loop.build_from_env` gives non-lead
  ranks `store=None` (`loop.py:735-737`); passing a store on every rank would
  race four writers onto one `current.json`.

### 6.5 The quiesce window

Reuse vLLM's own drain rather than inventing one. The **router** does it, around
the collective call, exactly as the RLHF weight-update protocol does:

```
await engine.pause_generation(mode=body.drain_mode, clear_cache=False)   # async_llm.py:750
try:
    results = await engine.collective_rpc("fq_admin_apply", args=(plan_json,),
                                          timeout=body.timeout_s)
finally:
    await engine.resume_generation()
if body.reset_prefix_cache:
    await engine.reset_prefix_cache(False, False)
```

so the worker passes `contextlib.nullcontext()` as `quiesce` (the sanctioned
usage when "nothing can be replaying", `swap.py:990-991`). Drain timeout →
`408 drain_timeout`, matching `scale_elastic_ep`
(`elastic_ep/api_router.py:77-82`).

`clear_cache=False` by default: a K3↔K4 tier change is within the quantisation
noise the model already lives with, and the policy loop performs the identical
mutation every interval without clearing anything. Cached prefix blocks computed
at the old precision remain valid *shapes*; only their numerics predate the swap.
`reset_prefix_cache: true` is offered for operators who want strict consistency,
and the response reports which was done.

Window cost, for scale: `runs/m4-swap/report.md` measured 0.061 ms for a 1-pair
toy and 0.368 ms for 8 pairs (fixed overhead only — toy payloads are ~350× smaller
than GLM-5.2 rank shards). At real geometry the window is H2D-bound: ~8.0 MiB per
pair over pinned PCIe, so **sub-millisecond for a 1-pair operator action** and a
few tens of ms at 32 pairs. The *drain* dominates end-to-end latency, not the
swap.

---

## 7. Responses

### 7.1 Success

`200`:

```jsonc
{
  "request_id": "fqr-01JT8Q2M4",
  "applied": true,
  "dry_run": false,
  "mode": "strict_pair",
  "policy": {"sha_before": "9c1f…", "sha_after": "4ab0…", "committed": true,
             "store_path": "~/.cache/vllm/fq/policy/<manifest>/current.json"},
  "plan": {"pairs": [{"layer": 23, "expert_out": 250, "expert_in": 1,
                      "k_out": 3, "k_in": 4}],
           "layers": [23], "plan_sha": "7d2e…", "generation": 7},
  "items": [
    {"layer": 23, "expert": 250, "requested": -1, "interpretation": "relative",
     "k_from": 4, "k_to": 3, "outcome": "demoted", "paired_with": 1},
    {"layer": 23, "expert": 1, "requested": "+1", "interpretation": "relative",
     "k_from": 3, "k_to": 4, "outcome": "promoted", "paired_with": 250}
  ],
  "memory": {"unit_bytes_per_k_per_rank": 1179648,
             "delta_bytes_per_rank": 0,
             "current_expert_bytes_per_rank": 78403227648,
             "projected_expert_bytes_per_rank": 78403227648,
             "budget_bytes_per_rank": 78403227648,
             "headroom_bytes_per_rank": 0,
             "device_free_bytes_min_rank": 12884901888,
             "bytes_h2d_per_rank": 8357888,
             "pinned_staging_bytes": 536870912},
  "occupancy": {"23": {"3": 148, "4": 108}},
  "pins": {"added": {"23": [1, 250]}, "removed": {}, "free_slots": {"23": 106}},
  "timing": {"drain_ms": 812.0, "stage_ms": 6.4, "window_ms": 0.41,
             "total_ms": 1043.2},
  "ranks": {"count": 4, "agreed": true, "generation": [7, 7, 7, 7],
            "policy_sha_after": ["4ab0…", "4ab0…", "4ab0…", "4ab0…"]},
  "decision_record": "~/.cache/vllm/fq/policy/<manifest>/decisions/00123000-admin-fqr-01JT8Q2M4.json",
  "warnings": []
}
```

`"applied": false` with an otherwise-identical body for `dry_run: true`
(`plan.generation` and `timing.window_ms` omitted).

### 7.2 Error codes

| HTTP | `code` | When |
|---|---|---|
| 400 | `bad_json` | body is not JSON / not `application/json` |
| 400 | `unknown_field` | unrecognised key |
| 400 | `empty_request` | no items and no pin change |
| 400 | `too_many_items` | `len(items) > VLLM_FQ_ADMIN_MAX_ITEMS` |
| 400 | `bad_adjust_k` | wrong type / zero or multiple of `adjust_k`/`k`/`delta_k` |
| 400 | `ambiguous_adjust_k` | positive JSON number outside the K ladder (§3.2) |
| 400 | `expert_out_of_range` | `expert` outside `[0, num_experts)` |
| 400 | `duplicate_item` | same `(layer, expert)` twice |
| 400 | `mixed_input` | query-string shorthand plus a body |
| 403 | `fq_admin_forbidden` | `X-FQ-Admin-Token` missing/wrong |
| 404 | `fq_admin_disabled` | `VLLM_FQ_ADMIN_ENABLE != 1` |
| 404 | `fq_not_active` | no loop state / no mixed-trellis layers |
| 404 | `layer_not_registered` | layer not in policy or not in `SwapEngine.layers` |
| 408 | `drain_timeout` | `pause_generation` did not drain in `timeout_s` |
| 409 | `cardinality_unbalanced` | strict_pair, per-layer promo≠demo |
| 409 | `memory_budget_exceeded` | `projected > budget` |
| 409 | `memory_headroom_exceeded` | device free minus delta below headroom |
| 409 | `fragment_unavailable` | requested K not resolvable (§5.3) |
| 409 | `pin_would_starve_layer` | §5.2 |
| 409 | `policy_sha_mismatch` | `expect_policy_sha` stale |
| 409 | `retier_in_flight` | lock held |
| 409 | `plan_too_large` | `len(plan) > engine.max_pairs` |
| 501 | `tier_not_servable` | `k_to ∉ {3,4}` (v1 engine + SM120 K5 block) |
| 501 | `budget_growth_not_supported` | `mode: "grow_budget"` (§4.2) |
| 501 | `dp_not_supported` | `data_parallel_size > 1` |
| 503 | `engine_unavailable` | sleeping / already paused / scaling |
| 500 | `rank_divergence` | ranks disagree on membership, plan or generation |
| 500 | `apply_failed` | exception from `SwapEngine.apply` |

`apply_failed` **must** carry the recovery state, taken from the staged batch and
the abort semantics at `swap.py:999-1013`:

```json
{"error": {"code": "apply_failed", "message": "...",
  "details": {"flipped": false, "restored": true, "torn": false,
              "generation": 6, "ranks_restored": [true, true, true, true],
              "guidance": "fail-atomic staging restored the pre-swap rows and maps inside the quiesce window; the layer is fully-old and the serve is healthy"}}}
```

If any rank reports `restored: false, flipped: false`, that rank's layer is torn
— log at ERROR, set `"torn": true`, and tell the operator to re-issue the same
request (roll forward) or restart (slabs are a cache, rebuilt from artifacts).

### 7.3 Cross-rank agreement

Phase 1: every rank returns `plan_sha`, `policy_sha_before`, and a digest of the
touched layers' membership. Any mismatch → `500 rank_divergence`, nothing applied.

Phase 2: every rank returns `generation`, `policy_sha_after`, `bytes_h2d`,
`window_ms`, `restored`. `generation` and `policy_sha_after` must be identical
across ranks — this is the same cross-rank agreement property T6 established for
decisions (`loop.py:454-457`). Divergence here means the ranks' weights now
disagree: log at ERROR, still resume generation (a hung serve helps nobody), and
return `500 rank_divergence` with the per-rank values so the operator can decide
whether to restart.

---

## 8. Loop integration

Four requirements from the brief, each with a concrete mechanism.

### 8.1 Visible to the loop, and to `occupancy_table`

Add one public method to `FungibleQuantState` and **refactor `_maybe_apply` to
call it**, so there is exactly one code path that adopts a new policy:

```python
def adopt_policy(self, new_doc: dict, swaps: list, *,
                 origin: str = "policy", record: dict | None = None) -> None:
    """Adopt a committed membership change. `origin` is 'policy' or 'operator'."""
```

It must do, in order:

1. `moved = self.tier_of != new_tier` ; `self.tier_of = new_tier`
2. `self._entered_step = np.where(moved, self._real_steps, self._entered_step)`
   — **dwell restarts** for moved experts, so the next interval cannot instantly
   trade them back.
3. `self.policy_doc = new_doc` ; `self.policy_sha = policy_hash(new_doc)` ;
   `self._policy_step = self._step`
4. **`self.pins = self._pins_from_doc(new_doc)`** — *this line is missing from
   today's `_maybe_apply` (`loop.py:560-569`)*. Benign for the loop, which never
   changes `pinned`; a live bug the moment the admin path does. Fixing it inside
   `adopt_policy` fixes both callers at once.
5. `self.log_composition(title=..., diff_only=True)` — the layer×K matrix, with
   the delta column, lands in the serve log at the moment of the change
   (`loop.py:631-651`, `occupancy_table.render`). Title for the admin path:
   `"expert composition after operator retier <request_id>"`.
6. Metrics (`loop.py:653-663` + two new instruments):
   `fq_swaps_total{layer}` per pair; new
   `fq_forced_swaps_total{layer,direction}` (`direction ∈ {promote, demote}`);
   new `fq_admin_requests_total{outcome}`; `_export_occupancy()`;
   `fq_policy_age_steps` reset.

The loop's own `n_k4` is unchanged (fixed cardinality), so `_decide_cfg()`
(`loop.py:428-435`) needs no adjustment and `policy.decide`'s budget assertion
(`policy.py:104-107`) keeps passing.

### 8.2 Persisted, survives the next interval and a restart

`SwapEngine.apply` already commits as **step 5 of the commit protocol**, inside
the quiesce window (`swap.py:1075-1077`), when handed `policy_store` and
`policy_doc`. `PolicyStore.commit` validates, rotates history, and atomically
renames `current.json` (`store.py:76-91`). So:

* **Next interval**: `run_interval` reads `self.tier_of`, which `adopt_policy`
  updated → the forced membership is the baseline it decides against.
* **Restart**: `loop.build_from_env` prefers the store's `current.json` over
  `VLLM_FQ_POLICY` / artifact synthesis (`loop.py:722-732`), *and*
  `ProgressiveSpec._policy_from_store` (`progressive.py:202-230`) feeds the same
  committed policy to the progressive loader, so the next boot assembles the
  layers with the forced tiers rather than re-deriving them. The override
  genuinely survives a reboot.

One behavioural note to document: in `VLLM_FQ_APPLY_MODE=dryrun` the loop never
commits `current.json` (`loop.py:581-588` writes proposals to `history/` only).
A forced retier **does** commit it. That is intended — dryrun means "the loop
does not write", not "nobody writes" — but it means after the first forced
retier, `current.json` exists and will be rehydrated at the next boot.

### 8.3 Attributable as operator-forced

Three artifacts, all mandatory.

**(a) Policy provenance** — replaces `loop._doc_for`'s block (`loop.py:532-537`):

```json
"provenance": {
  "proposed_by": "fq-admin/retier",
  "origin": "operator",
  "request_id": "fqr-01JT8Q2M4",
  "actor": "michel",
  "reason": "expert 1 is hot on the coder axis",
  "utc": "2026-08-11T04:12:07Z",
  "step": 123000,
  "base_policy": "9c1f…",
  "num_swaps": 1,
  "mode": "strict_pair",
  "items": [{"layer": 23, "expert": 250, "k_from": 4, "k_to": 3, "source": "operator"},
            {"layer": 23, "expert": 1,   "k_from": 3, "k_to": 4, "source": "operator"}]
}
```

`items[].source` is `"operator"` for requested items and `"auto_balance"` for
partners the server chose — so an auto-paired demotion is never mistaken for
something the operator asked for.

**(b) Decision record.** Extend `decision_log.py` with

```python
def explain_forced(tier_of, swaps, *, items, actor, reason, request_id,
                   step, policy_sha_before, policy_sha_after) -> dict
```

emitting the **same `fq-decision/1` schema** so existing consumers keep working,
with:

* a new top-level `"origin": "operator"`. The interval path
  (`decision_log.explain`, `loop.py:495-499`) gains `"origin": "policy"` — a
  one-line change with a test asserting both.
* `swaps[]` entries carrying `"forced": true` and the score fields still
  populated (recomputed from the live stats) so a human can see *what the policy
  thought* about an expert the operator overrode. That comparison is the whole
  value of the record.
* `"guards_waived": ["dwell", "hysteresis", "jaccard", "max_swaps_per_layer"]`
  and zeroed `"blocked"` tallies.
* `"actor"`, `"reason"`, `"request_id"`.

Written to `store.root/decisions/{step:08d}-admin-{request_id}.json`. The
`-admin-` infix avoids colliding with the interval record at
`{step:08d}.json` (`loop.py:579-580`) when a forced retier lands on an interval
boundary.

**(c) Log lines.** One WARNING before and one INFO after (WARNING because a
human mutating live weights should be visible in a grep for warnings):

```
FQ ADMIN retier request=fqr-01JT8Q2M4 actor=michel mode=strict_pair items=2 layers=[23] reason="expert 1 is hot on the coder axis"
FQ ADMIN applied  request=fqr-01JT8Q2M4 pairs=1 gen=7 window=0.41ms delta_bytes/rank=0 policy 9c1f… -> 4ab0… pins+2
```

followed by `log_composition`'s diff table (§8.1 step 5).

---

## 9. Worked example — the operator's own case

Request:

```bash
curl -sS -X POST http://localhost:8000/fq/retier \
  -H 'Content-Type: application/json' \
  -H 'X-FQ-Actor: michel' \
  -d '{"items": [{"layer": 23, "expert": 250, "adjust_k": -1},
                 {"layer": 23, "expert": 1,   "adjust_k": "+1"}],
       "reason": "coder-axis promotion, paired demotion of a cold expert"}'
```

Assume GLM-5.2 at TP4: `E = 256` experts per layer, layer 23 in the bulk class
with `n_k4_per_layer["23"] = 108` (`convergence-demo-plan.md`), expert 250
currently K4-resident, expert 1 currently K3-resident.

**Parse (§3.2).** `-1` is a negative JSON number → relative, `delta_k = -1`.
`"+1"` is a signed string → relative, `delta_k = +1`. Both interpretations are
echoed in the response so the operator can confirm the API read them the way
they meant.

**Resolve (§3.3).** `k_from` is read from live state, never from the client:
`tier_of[row(23)][250] = 4 → k_to = 3` (demotion);
`tier_of[row(23)][1] = 3 → k_to = 4` (promotion). Neither is a no-op; both
targets are in `{3, 4}`.

**Balance (§4).** Layer 23: 1 promotion, 1 demotion → balanced. `strict_pair`
satisfied.

**Memory (§5.1).**

```
delta = 1,179,648 × (3−4)  +  1,179,648 × (4−3)
      = −1,179,648 + 1,179,648
      = 0 bytes/rank
projected = 78,403,227,648 B/rank = 73.0187 GiB/rank = budget    → PASS (equality)
```

**Availability (§5.3).** Pre-flight `resolve(23, 250, k=3)` — hits, the base is
all-K3. `resolve(23, 1, k=4)` — if the K4 encode for L23/e1 has not landed, this
substitutes K3 and the request stops here with `409 fragment_unavailable`, the
encode already queued by `resolve()` itself. Assume it hits.

**Target document.** `bits_per_expert["23"][250] = 3`, `["23"][1] = 4`. K4
occupancy is `108 − 1 + 1 = 108 == n_k4_per_layer["23"]`, so
`validate_policy` (`store.py:48-51`) passes. `pinned["23"]` gains `[1, 250]`
(default `pin: "hold"`); free K4 slots in layer 23 become
`108 − (existing pin4 + 1) ≥ VLLM_FQ_ADMIN_MIN_FREE_SLOTS` → pin guard passes.

**Plan.** `SwapPlan.from_policies(current_doc, new_doc)`:

* row for layer 23: `outs = {e : old==4 ∧ new==3} = [250]`,
  `ins = {e : old==3 ∧ new==4} = [1]`
* `len(outs) == len(ins) == 1` → the D1 cardinality check passes
  (`swap.py:441-446`)
* zip in ascending id order → `(row, 250, 1)`, remapped to the model layer id

```python
SwapPlan([(23, 250, 1)])        # (layer, e_out=250 leaves K4, e_in=1 enters K4)
```

**Staging** (`engine.stage(plan, fail_atomic=True, on_unavailable="raise")`).
Four fragment reads (`swap.py:860-865`):

| read | K | expert | into | why |
|---|---|---|---|---|
| 1 | 3 | 250 | `out_stage` | e250's **new** K3 encoding |
| 2 | 4 | 1 | `in_stage` | e1's **new** K4 encoding |
| 3 | 4 | 250 | `undo_k4` | pre-swap content of the K4 slot (fail-atomic) |
| 4 | 3 | 1 | `undo_k3` | pre-swap content of the K3 slot (fail-atomic) |

Slot arithmetic (`swap.py:886-890`): `slot1 = tier1_globals.index(250)`,
`slot0 = tier0_globals.index(1)`; post-swap `tier1_globals[slot1] = 1`,
`tier0_globals[slot0] = 250`. **e1 inherits e250's K4 slot and vice versa** —
every other expert's slot, slab row and combined row is untouched.

Op lists:

* **slabs (step 1)** — 6 row writes: `w13[0,slot1]`, `w13[1,slot1]`,
  `w2[slot1]` in tier1 from `in_stage`; the same three at `slot0` in tier0 from
  `out_stage`.
* **rotations (step 2)** — 8 combined-table row writes: `rotations.intermediate`,
  `gate_suh`, `up_suh`, `down_svh` at combined slot `t0n + slot1` (from
  `in_stage`) and at combined slot `slot0` (from `out_stage`).
* **maps (step 3, the visibility flip)** — 2 writes: layer 23's
  `global_to_combined` and `descriptor_map`, both validated host-side first as a
  full permutation and as exact `local | 256|local` descriptors
  (`swap.py:759-777`).
* **undo** — the same 14 destinations fed from `undo_k4`/`undo_k3` plus the
  pre-swap maps.

`bytes_h2d` ≈ K3 stage (3,542,028 B) + K4 stage (4,721,676 B) + rotation rows +
2 × 256 × 4 B of maps ≈ **8.0 MiB per rank**.

**Apply.** Router: `pause_generation(mode="wait", clear_cache=False)` → all four
ranks run `apply(staged=..., quiesce=nullcontext(), memo_hook=None,
policy_store=<lead only>, policy_doc=new_doc)` → `resume_generation()`.
Window: 14 `copy_` calls plus 2 map copies on the current stream,
sub-millisecond, H2D-bound. `generation` 6 → 7 on every rank. Lead rank commits
`current.json`; all ranks call `adopt_policy(new_doc, [(23,250,1)],
origin="operator", record=...)`.

**Why cardinality is satisfied, restated once per enforcement point:**

| Check | Location | Why it passes |
|---|---|---|
| policy occupancy == capacity | `store.py:48-51` | K4 count in layer 23 stays 108 |
| plan same-cardinality per layer | `swap.py:441-446` | 1 out, 1 in |
| tier orderings are a partition | `swap.py:713-718` | the swap is a transposition of two entries between the two orderings; the union is still `[0,256)` |
| kernel memo key unchanged | `exl3.py:1840-1851` | `tier_signature` = ((3, 148), (4, 108)) before and after |
| loop budget assertion | `policy.py:104-107` | `n_k4` unchanged, so the next `decide()` runs |

**The unbalanced variant.** Sending only `{"layer": 23, "expert": 1,
"adjust_k": "+1"}` produces `409 cardinality_unbalanced` (§4.3) — no weights
touched, no policy written, and the error body names the three remedies. This is
the case the operator's "guarded by maximum memory usage" instinct was reaching
for: on this stack the binding constraint is not free bytes, it is that the
slabs and the compiled kernel are both sized for exactly 108 K4 experts in that
layer.

---

## 10. Environment knobs

| Variable | Default | Meaning |
|---|---|---|
| `VLLM_SERVER_DEV_MODE` | unset | vLLM's own gate; router does not exist without it |
| `VLLM_FQ_ADMIN_ENABLE` | `0` | second gate, specific to forced re-tiering |
| `VLLM_FQ_ADMIN_TOKEN` | unset | if set, require matching `X-FQ-Admin-Token` |
| `VLLM_FQ_ADMIN_MAX_ITEMS` | `32` | max items per batch; also sets `SwapEngine.max_pairs` |
| `VLLM_FQ_ADMIN_MEM_BUDGET_BYTES` | boot resident expert bytes/rank | hard ceiling |
| `VLLM_FQ_ADMIN_MEM_HEADROOM_BYTES` | `1073741824` | device free must stay above this after a growing change |
| `VLLM_FQ_ADMIN_MIN_FREE_SLOTS` | `2` | per-layer unpinned K4 slots the loop must retain |
| `VLLM_FQ_ADMIN_DRAIN_TIMEOUT_S` | `120` | default `timeout_s` |
| `VLLM_FQ_ADMIN_ALLOW_AUTO_BALANCE` | `0` | enables `mode: "auto_balance"` |

---

## 11. Files touched

| File | Change |
|---|---|
| `vllm/entrypoints/serve/dev/fq/__init__.py` | new, empty |
| `vllm/entrypoints/serve/dev/fq/api_router.py` | new — the four routes, gates, lock, drain, aggregation |
| `vllm/entrypoints/serve/__init__.py` | +3 lines in `register_vllm_dev_api_routers` |
| `.../exl3_fungible/admin.py` | new — request model, disambiguation, guards, plan construction, `build_swap_engine`, `worker_describe/plan/apply`, `FqWorkerAdmin` mixin |
| `.../exl3_fungible/loop.py` | `adopt_policy()`; `_maybe_apply` refactored onto it; `pins` refresh (the §8.1 bug); `swap_engine` slot |
| `.../exl3_fungible/decision_log.py` | `explain_forced()`; `"origin"` on both record paths |
| `vllm/v1/worker/gpu_worker.py` | +3 thin lazy-delegating methods on `Worker` |

No changes to `swap.py`, `store.py`, `policy.py`, `fragments.py`,
`occupancy_table.py`. That is the test of whether this design is right: the
endpoint drives the existing engine and adds guards; it does not modify the
mechanism.

---

## 12. Tests that must land with the code

CPU-only, in `tests/exl3_fungible/test_admin_cpu.py` unless noted. The suite is
currently 142 passed / 11 skipped — do not regress it.

**Parsing and disambiguation**
1. Every row of the §3.2 table, including `adjust_k: 1` → `ambiguous_adjust_k`
   and `adjust_k: -1` → relative.
2. `k` / `delta_k` / `adjust_k` mutual exclusion.
3. Duplicate `(layer, expert)`; expert out of range; unknown field.
4. Query-string shorthand desugars to the canonical body; body + query →
   `mixed_input`.

**Cardinality**
5. Balanced two-item batch → `SwapPlan([(23, 250, 1)])` exactly (the §9 example,
   on a toy 16-expert layer).
6. Lone promotion → `cardinality_unbalanced`, no store write, no engine call
   (assert with a mock `SwapEngine` that `stage` was never entered).
7. A no-op item that unbalances an otherwise-balanced pair →
   `cardinality_unbalanced` with the no-op listed in `details`.
8. Absolute `k` and relative `delta_k` mixed in one balanced batch produce the
   same plan as the all-relative spelling.

**Memory guard**
9. Balanced batch → `delta_bytes_per_rank == 0`, guard passes, reported.
10. A synthetic unbalanced-but-permitted batch (guard tested directly, not
    through the endpoint) with `delta > budget` → `memory_budget_exceeded`, and
    the arithmetic equals `UNIT × Σ(k_to − k_from)` to the byte.
11. `mode: "grow_budget"` → `501` with all three reasons in the message.

**Pins**
12. `pin: "hold"` adds both experts pinned to their new tiers; a subsequent
    `policy.decide` with those pins does not move them.
13. `pin: "release"` removes them.
14. A pin request that would leave fewer than `MIN_FREE_SLOTS` →
    `pin_would_starve_layer`, and (regression) `policy.decide` on the *refused*
    doc would indeed have raised `"pins incompatible with budget"`.

**Fragments**
15. Pre-flight with a fake resolver that substitutes K3 for a K4 ask →
    `409 fragment_unavailable`, listing **every** unavailable item (not just the
    first), with `encode_queued: true`.
16. Nothing is staged or applied when the pre-flight fails.

**Loop integration**
17. `adopt_policy` updates `tier_of`, `policy_doc`, `policy_sha`,
    `_policy_step`, resets `_entered_step` for moved experts only, **and
    refreshes `pins`** — the last assertion fails against today's
    `_maybe_apply`, which is the point.
18. `_maybe_apply` refactored onto `adopt_policy` produces byte-identical state
    to the current implementation for a loop-driven swap (golden test).
19. `log_composition` is called once after a forced retier and its rendered
    table shows the layer-23 row with a `+1 K4 / -1 K3`-free (net-zero) diff and
    the `*` changed flag.
20. The decision record round-trips: interval records carry
    `"origin": "policy"`, admin records `"origin": "operator"` with
    `forced: true`, `actor`, `reason`, `request_id`, and land at
    `decisions/{step:08d}-admin-{request_id}.json` without colliding with an
    interval record at the same step.

**Router** (`tests/entrypoints/openai/test_fq_admin_router.py`, FastAPI
`TestClient` + a fake engine client)
21. Routes absent without `VLLM_SERVER_DEV_MODE`; `404 fq_admin_disabled`
    without `VLLM_FQ_ADMIN_ENABLE`; `403` on a bad token.
22. `pause_generation` / `resume_generation` are called in that order, and
    `resume_generation` is called **even when the collective RPC raises**.
23. Rank divergence in phase 1 → `500 rank_divergence`, and `fq_admin_apply` is
    never called.
24. `retier_in_flight` under two concurrent requests.
25. `dry_run: true` calls `fq_admin_plan` and never `fq_admin_apply`, and never
    pauses generation.

**GPU** (`test_admin_gpu.py`, `@pytest.mark.skipif(not cuda)`) — deferred until
GPUs are free; mirrors T4: force one pair through the real engine on the toy
checkpoint, assert bitwise equality against a freshly built layer with the new
membership, assert `current.json` holds the forced membership, then force the
inverse and assert the original output returns bitwise.

---

## 13. Open questions for the implementer

1. **`expected_mcg`.** `SwapEngine(expected_mcg=None)` learns the layer codebook
   from the first staged fragment (`swap.py:900-911`). The serve knows the
   checkpoint's mcg; passing it explicitly would make a foreign-encoding refusal
   fire on the *first* admin request rather than after a coincidence. Cheap to
   thread through if the prepared layer exposes it.
2. **`auto_balance` scoring source.** Rank 0's live decayed stats
   (`FungibleQuantState._read_stats` + `policy.score`) are the obvious ranking,
   but they are a snapshot. Consider requiring the client to pass the
   `expect_policy_sha` it read from `GET /fq/layer/{layer}` so the ranking the
   operator saw is the ranking that gets used.
3. **Should a forced retier reset the loop's Jaccard history?** `_prev_desired`
   (`loop.py:479-483`) is compared across intervals; a forced change does not
   alter the *desired* set, only the resident set, so probably not — but state
   the reasoning in a comment either way.
