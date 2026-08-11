# Reading guide — the 24 commits on `fq/m1-stats-collector`

`git log e2666d9a65f41fc376607531453cbd57c4c71016..161536085` on
`malaiwah/vllm-voipmonitor` branch `fq/m1-stats-collector`.
Merge base `e2666d9a` is on `local-inference-lab/vllm` `dev/gilded-gnosis`.

**Totals: 46 files, +21,820 / −6.** History is linear; nothing was rebased or
force-pushed.

## Which files are ours and which are the project's

40 of the 46 files are **new** and live in two directories that did not exist
before this branch:

- `vllm/model_executor/layers/quantization/exl3_fungible/` — 16 new files
  (15 modules + `PERFORMANCE.md`)
- `tests/exl3_fungible/` — 24 new files (23 test modules + `toy_segments.py`)

**Six pre-existing project files are touched, +152 / −6 in total.** These are
the ones to review line by line; the rest is additive new code with its own
tests.

| pre-existing file | +/− | nature |
|---|---|---|
| `vllm/model_executor/layers/fused_moe/router/base_router.py` | +26 −6 | **the only non-additive edit.** `capture_fn` / `set_capture_fn` already existed here; this makes the callback arity optional so a callback tagged `wants_topk_weights = True` also receives `topk_weights`. Default one-argument contract unchanged. Needed because `topk_weights` is a local in `_select_experts` and is not reachable from the router object. Resolved once in `set_capture_fn` so the routing hot path reads a plain bool. |
| `vllm/v1/worker/gpu_worker.py` | +59 −0 | `maybe_init_fq_state`, called from `initialize_from_config` strictly after `load_model`. Walks `runner.model.modules()` binding routers; never sees a loader tensor. |
| `vllm/entrypoints/serve/__init__.py` | +31 −0 | two env-gated `attach_router` calls appended to `register_vllm_dev_api_routers`. Imports live inside the guards, so the default path pays nothing. |
| `vllm/model_executor/model_loader/__init__.py` | +18 −0 | one `_LOAD_FORMAT_TO_MODEL_LOADER` entry plus the matching `LoadFormats` literal, resolved lazily through `_progressive_loader_cls`. |
| `vllm/v1/worker/gpu_model_runner.py` | +14 −0 | two `if getattr(self, "fq_collector", None) is not None:` calls, adjacent to `eplb_step()` and to the dummy-run `skip_eplb` block. Both are `None`-guarded, so cost with FQ off is two `getattr` calls per step. |
| `vllm/config/load.py` | +4 −0 | docstring only, describing the `progressive` load format. |

Line-by-line diff of exactly those six:

```bash
git diff e2666d9a65f41fc376607531453cbd57c4c71016..HEAD -- \
  vllm/config/load.py vllm/model_executor/model_loader/__init__.py \
  vllm/model_executor/layers/fused_moe/router/base_router.py \
  vllm/v1/worker/gpu_model_runner.py vllm/v1/worker/gpu_worker.py \
  vllm/entrypoints/serve/__init__.py
```

---

## Theme 1 — Observe: the stats collector and its integration

Where a reviewer should start. This is the part with the closest neighbour in
the tree (EPLB), and the part whose correctness everything downstream inherits.

| commit | date | size | what |
|---|---|---|---|
| `5e36368d6` | 08-10 | 3f +325 | **M1 stats collector.** `stats.py` — graph-safe capture path, capture-fn chaining (so an existing consumer is not displaced), window ring with decay, dummy-step semantics borrowed from EPLB (zero without recording). 7 CPU contract tests. |
| `18822df6d` | 08-10 | 4f +305 | **Integration hook.** `integration.py` binds the collector to `BaseRouter`s found by walking the model; `gpu_worker.py` calls it after `load_model`; `gpu_model_runner.py` advances the window once per engine step. |
| `33363a7f3` | 08-11 | 8f +1298 −28 | **Fast-path histogram, loop wiring, and a `torch.histc` sentinel fix.** The fix matters: `torch.histc`'s last bin is closed at `max`, so `id == num_experts` — the usual padding sentinel for `topk_ids` — was not dropped but folded into the *final expert's* bin. Memory-safe, but it silently biased the routing histogram toward exactly one expert, i.e. corrupted the swap policy's input. Now binned with an overflow slot that is sliced off. |
| `7a16ff318` | 08-11 | 1f +12 −1 | Import `MoERunner` from its defining module (`fused_moe.runner.moe_runner`) with a fallback; `fused_moe.layer` is now only a re-export and the old import is a rebase hazard. |
| `6e08f683d` | 08-11 | 7f +843 −38 | **Record real gate mass, not a copy of the hit count.** `mass` had been byte-identical to `count` in every record, because `bind_router` never passed a getter *and* nothing held `topk_weights` at capture time. This adds the optional second argument at the `base_router.py` call site. Opt-in (`VLLM_FQ_GATE_MASS=1`): it takes the device kernel count from 3 to 8 per MoE layer per forward. 25 CPU tests. |

## Theme 2 — Decide: policy, store, loop, explainability

| commit | date | size | what |
|---|---|---|---|
| `f86200e4c` | 08-10 | 4f +601 | **M2 policy engine + store.** `policy.py` (`decide`/`apply`/`inverse`/`project` with guards), `store.py` (`fq-policy/2` `PolicyStore`: atomic commit, history ring, manifest binding, and topology-neutrality validation that bans `rank`/`world_size`/`tp`/`device` keys). |
| `17069d554` | 08-10 | 2f +231 | **Explainable decision records.** Per-swap rationale (scores, ε-gap, mass, hysteresis ratio), why-not tallies (dwell / hysteresis / cap), an EPLB-style one-line interval log, JSON persistence. |
| `e866c4081` | 08-11 | 4f +414 −1 | Operator-facing expert composition matrix in the engine log (`occupancy_table.py`). |
| `ed91d51d3` | 08-11 | 1f +30 | Env-gated per-expert routing stats dump — this is what produced the `stats-*.jsonl` files behind the convergence scores and the heatmap. |
| `b78eb7f7b` | 08-11 | 4f +2062 −15 | **Memory budget in bytes, not just expert cardinality.** The boot gate measured 76.14 GiB/rank of a 95.6 GiB card, so counting experts is not a budget; counting bytes is. |
| `adc3bcf83` | 08-11 | 5f +510 −44 | Three memory-budget defects found by adversarial review. |
| `9728d2bf6` | 08-11 | 3f +395 −27 | Memory-budget review round 2 — a frozen loop and a 0.67 GiB byte-model error. |

## Theme 3 — Supply: fragments, the progressive loader, trust

| commit | date | size | what |
|---|---|---|---|
| `0d6d54196` | 08-10 | 7f +1730 | **Progressive Loader v2.** `fragments.py` + `progressive.py` + `progressive_loader.py` stream mixed-K EXL3 weights from per-expert segments plus a bitrate policy, with no assembled checkpoint on disk. Registers `--load-format progressive`. |
| `b69feebca` | 08-10 | 4f +1517 −108 | **Operator controls.** Multi-repo sources, attestation trust filtering (`VLLM_FQ_TRUST_SIGNERS` / `_PREDICATES`, armed only with an anchor), lazy-encode fallback ladder, verbose resolve decisions. |
| `2158a69f3` | 08-10 | 1f +17 −5 | Resolve the progressive loader **lazily** in `model_loader/__init__.py`. Load-bearing: an eager import makes `exl3_fungible.progressive_loader` unimportable standalone. Pinned by a test. |
| `304c51259` | 08-11 | 2f +462 −5 | **Loader-variant compatibility tests + shard fd scope.** 12 CPU tests covering owning-vs-borrowed tensor lifetime across every `--load-format` in the tree, including a control test that proves the fake borrowed loader really does corrupt a retaining consumer. The code change closes the dense-shard file object once the mapping exists (`mmap(2)` keeps its own dup), halving open fds during the load window. This commit is where separate report (a) came from. |
| `90140ab28` | 08-11 | 5f +1278 −78 | **A missing K never crashes the serve.** Four crash paths closed (uncaught `JSONDecodeError` / `ValueError` / `OSError` / `IsADirectoryError` / `UnicodeDecodeError` / `KeyError` escaping into the loader and the swap stager), each with a test that fails against the parent commit. 27 CPU tests. |

## Theme 4 — Apply: the atomic swap engine

| commit | date | size | what |
|---|---|---|---|
| `a16c87f73` | 08-10 | 4f +1654 | **M4 atomic swap engine.** `swap.py` — row-write commit protocol into existing tier slabs, tier-map flip inside a quiesce window, no reallocation and no graph recapture. T3 (map mutation under a captured CUDA graph) and T4 (row-write fidelity + bitwise rollback) verified on SM120; see `runs/m4-swap/report.md`. |
| `aaf333e99` | 08-10 | 6f +1919 −66 | **T5 torn-update fault injection + T6 cross-rank agreement**, and `FragmentResolver`-backed swap staging. T6 runs four *simulated* ranks deliberately: the property under test is that the same pure function agrees on identical inputs, and four real GPUs would weaken the test by making accidental agreement easy. |
| `163c9ee7d` | 08-10 | 3f +12 −3 | Make the new T5/T6/resolver CPU tests directly runnable. |

## Theme 5 — Operator surfaces

| commit | date | size | what |
|---|---|---|---|
| `73d5b9e14` | 08-11 | 4f +3659 | **Admin API for operator-forced re-tiering** (`admin.py`, `POST /fq/retier`). Relative (`adjust_k=-1`) or absolute (`adjust_k=3`), batched, budget-guarded. Doubly gated: `VLLM_SERVER_DEV_MODE` gets you to the registration site, and `attach_router` itself returns `False` unless `VLLM_FQ_ADMIN_API=1`. It reuses `SwapEngine`/`PolicyStore` rather than inventing a second way to move an expert between tiers. Design: `runs/m5-serve/admin-api-spec.md`. |
| `c4d7eeb28` | 08-11 | 4f +2689 | **Activation-matrix endpoint** (`heatmap.py`, `GET /fq/heatmap`). Read-only, so it carries its own gate and its own optional token — a dashboard should not need the credential that mutates live weights. Declares `MERGE_RULE = "rank0-canonical"` (under TP the gate is replicated, so summing ranks would inflate every cell 4×) and refuses DP > 1 with `501 dp_not_supported` rather than reporting one replica as the whole serve. |
| `161536085` | 08-11 | 2f +254 −7 | Adversarial-review fixes to the endpoint. Review and verdict: `runs/m5-serve/heatmap/REVIEW.md`. |

## Theme 6 — The standing contract

| commit | date | size | what |
|---|---|---|---|
| `3fa2e901f` | 08-10 | 1f +23 | `exl3_fungible/PERFORMANCE.md` — every change in the package that can influence KLD, runtime memory, or PP/TG must be graph-performant by construction *and* land with measured impact numbers. This is the document against which the PR body reports the M1 decode-overhead gate as **not met**. |

---

## Suggested review order

1. `base_router.py`, `gpu_model_runner.py`, `gpu_worker.py` — the whole
   integration surface, ~100 lines, all `None`-guarded.
2. `stats.py` at `33363a7f3` — the `histc` sentinel fix, and the argument for
   why the capture is graph-safe.
3. `policy.py` + `store.py` at `f86200e4c` — the decision algebra and the
   topology-neutrality validation.
4. `swap.py` at `a16c87f73` — the commit protocol; `stage()` does all IO
   pre-quiesce, `apply()` opens the window only after staging returned.
5. Everything else is additive and test-backed.

## Test files, by theme

`tests/exl3_fungible/` — 24 files, 493 passing CPU tests + 10 GPU tests that
skip without a device.

| theme | files |
|---|---|
| stats / integration | `test_stats_cpu.py`, `test_integration_cpu.py`, `test_gate_mass_cpu.py`, `test_base_router_gate_mass_cpu.py` |
| policy / loop / budget | `test_policy_cpu.py`, `test_store_cpu.py`, `test_loop_cpu.py`, `test_decision_log_cpu.py`, `test_occupancy_table_cpu.py`, `test_memory_budget_cpu.py` |
| supply | `test_fragments_cpu.py`, `test_progressive_cpu.py`, `test_trust_lazy_cpu.py`, `test_missing_k_cpu.py`, `test_loader_compat_cpu.py` |
| swap | `test_swap_cpu.py`, `test_swap_gpu.py`, `test_swap_resolver_cpu.py`, `test_swap_t5_cpu.py`, `test_swap_t5_gpu.py`, `test_cross_rank_t6_cpu.py`, `toy_segments.py` |
| operator surfaces | `test_admin_cpu.py`, `test_heatmap_cpu.py` |
