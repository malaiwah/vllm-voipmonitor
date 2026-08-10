# 03 — Testing & validation

Every layer of the system gets a test that can fail loudly BEFORE the failure
mode becomes silent corruption in serving. Ordered cheapest-first; CPU-only
tests follow the GG convention of directly-runnable contract harnesses
(cf. the fork's existing exl3 CPU contract tests).

## T1 — Stats collector graph-freeze test (the load-bearing one)

The classic failure: a capture fn that silently freezes at CUDA-graph capture
time. Test: bind collector, serve a tiny MoE model (or GLM-5.2 with
`--max-model-len` small) **with graphs enabled**, run 100 decode steps,
assert `count_buf.sum()` grows ~linearly with steps and matches
`tokens × top_k` within tolerance. Run twice: eager and graphed; results must
match. Any `.item()`/host-op regression fails this immediately.

**Build note (2026-08-10) — PASS, but the method above is not what runs**
(`../runs/t1-graph-freeze/report.md`):

- **Binding is gated.** The production binding site
  (`_bind_routed_experts_capturer`) only fires with
  `enable_return_routed_experts=True`, off by default — **three earlier T1
  runs were hollow and would have "passed" with nothing bound.** Any
  liveness test must assert an **absolute** count, never "nonzero" or
  "grew". M1 ships its own env-gated binding call instead of relying on
  that flag.
- **Graphed-only, with an absolute-count referee.** The eager/graphed twin
  comparison is not runnable here: the bf16 eager path is broken on this
  stack and the triton MoE backend lacks the `MoERunner` hooks the test
  chains onto. The referee used instead is exact, not tolerance-based:
  per-layer total == prompt+gen routings minus a **constant** boundary
  offset of 16 (= 2 tokens × 8), validated at two run lengths (143 and 271
  tokens → 1128 and 2152 on all 10 layers), plus monotonic in-replay growth
  (10800 → 21520) and exact cross-layer agreement.
- **Remaining T1 legs:** TP4, and the M1 decode-overhead measurement
  (< 0.5 % at cc8) on the GLM-5.2 serve.

## T2 — Policy engine property tests (CPU, no GPU)

- Determinism: same inputs → same swap list, bit-identical across repeated runs.
- Budget invariant: post-decision membership always has exactly N_L K4 per layer.
- Pin respect, dwell respect, cap respect, hysteresis monotonicity.
- Inverse property: applying a swap list then its inverse restores membership.
- Projection: a policy JSON with different N_L is projected onto running
  cardinality correctly.

## T3 — Tier-map mutation kernel test (GPU, single card)

Build a small mixed_trellis layer (e.g. E=16, N=6). Assert:
1. `build_tiered_maps` on permuted memberships + in-place copy into the
   existing `global_to_combined`/`descriptor_map` tensors changes routing
   correctly (compare against a freshly-built layer with the same membership).
2. Under CUDA-graph capture: capture forward, mutate map CONTENTS, replay —
   output must match the fresh-build reference. This proves maps are read as
   data, not baked. (If this fails, the whole atomic path is dead and
   APPLY_MODE=reload is the ceiling — find out here, not in serving.)

**Build note (2026-08-10) — T3 PASS (bitwise)** on GPU 7 / SM120
(`../runs/m4-swap/report.md`). Maps are read **as data**, not baked at
capture: forward captured in a CUDA graph, map contents mutated in place
through the engine's rebuild path (`copy_`, `data_ptr` unchanged), replay
output `torch.equal` to a fresh-built layer with the new membership.
Non-vacuous — the mutated-map replay differs from the pre-mutation output,
and the permutation included **cross-tier** (K3↔K4) moves as well as
within-tier reorders. **The atomic path is live; `APPLY_MODE=reload` is not
the ceiling**, which retires 04's first abort signal.

## T4 — Row-write fidelity test (GPU)

For one expert: load its K4 rows into a slab that previously held its K3
encoding (and vice versa), update suh/svh rows, rebuild maps, forward a probe
batch. Reference: a layer built from scratch with that membership. Assert
bitwise-equal logits (same kernels, same inputs — equality, not tolerance).
Parametrize over: expert at slab row 0, last row, middle; w13 and w2.
(If K6 resolves to slab-rebuild-only, this test becomes the per-layer slab
rebuild fidelity test — same reference structure.)

**Build note (2026-08-10) — T4 PASS ×3 (bitwise)**; K6 resolved to
**row-write**, so the parenthetical fallback never applied
(`../runs/m4-swap/report.md`). One expert pair swapped end-to-end by
`SwapEngine.apply()` from real `fq-segment/1` byte payloads; post-swap slabs,
the **combined** rotation/suh/svh tables and both maps `torch.equal` to a
fresh-built layer, and the probe-batch forward is bitwise-equal.
`apply(plan.inverse())` restores the pre-swap output bitwise — rollback is
tested, not asserted. Two legs the spec did not anticipate are also covered:
staged fragments must agree on the layer `mcg` word (foreign-mcg refused at
staging) and broadcast/shared-H suh/svh layouts are refused at construction.
`../runs/m3-reload/report.md` provides the reload-mode analogue at serve
scale: post-reload logits **bit-identical** to a fresh boot (max
|Δlogprob| = 0.0 over 356 scored tokens, twice).

## T5 — Torn-update fault injection

Deliberately apply HALF a swap (rows written, maps not flipped; maps flipped,
suh not updated) and assert the quiesce protocol makes that state unreachable:
the commit sequence in `02-swap-engine.md` must order writes so every
intermediate visible state is either fully-old or fully-new. Test by
instrumenting the swap engine with an abort-after-step-k hook and asserting
forward output equals pre-swap output for every k < commit, and post-swap
output for k ≥ commit.

## T6 — Cross-rank agreement (4-GPU)

Run TP4 with the policy engine in debug hash-check mode for 50 intervals under
synthetic traffic; assert no rank ever diverges in decision hash. Then kill
the debug collective and rerun — outputs must be identical (determinism is the
mechanism, the hash was only evidence).

## T7 — End-to-end soak (the release gate)

4×RTX 6000 Pro, real GLM-5.2 FQ artifacts, replayed real traffic (the
4-axis corpus driver from the collector repo), `APPLY_MODE=atomic`,
aggressive knobs (interval=200, caps=8) to force churn:
- 24 h, zero crashes, zero probe-rollback loops (rollback firing repeatedly
  on the same experts = thrash → fail).
- Decode/prefill tok/s within 1% of `VLLM_FQ_ENABLE=0` baseline between swap
  intervals; swap-interval latency spike bounded (< 1 engine step of stall
  in atomic mode).
- KLD vs BF16 on the held-out probe: monotone non-increasing trend over the
  soak (the whole point), never above the starting generic policy's value
  after convergence.
- Memory: `torch.cuda.memory_allocated` flat across 24 h (no leak from
  staging buffers or map rebuilds); zero cudaMalloc in steady state
  (assert via allocator stats delta between intervals).

## T8 — Rehydration & crash-recovery

- Kill -9 mid-swap-batch; restart; assert boot picks `policy/current.json`
  (last committed), never a torn one (write-to-temp + atomic rename).
- Boot with slab cache vs without: identical logits on probe batch.
- Boot with a policy whose manifest hash mismatches artifacts: hard refuse.

## T9 — Quality regression suite (per release)

The GG release-gate pattern (40-config campaigns) applies; minimum additions:
- KLD ladder: generic policy vs converged policy vs all-K3 vs current static
  3.42bpw mix, identical eval set.
- Code-syntax validity + long-context retrieval spot checks (the two metrics
  the brandonmusic card tracks), converged vs generic.
- MAL bench (collector repo `arm_bench.py`) unchanged for MTP-78-pinned v1.

## Instrumentation (ships with M2, not later)

Prometheus: `fq_swaps_total{layer}`, `fq_rollbacks_total`,
`fq_probe_kld`, `fq_jaccard`, `fq_policy_age_steps`,
per-tier occupancy gauges. Log line per interval mirroring EPLB's
balancedness log. These are also the observability story for "watch your
model specialize" — worth a small dashboard for the announcement post.

## Status board (build note, 2026-08-10)

| Test | Status | Evidence |
|---|---|---|
| T1 graph-freeze | **PASS** (1 GPU, graphed); TP4 + overhead leg open | `../runs/t1-graph-freeze/report.md` |
| T2 policy properties | **PASS** (10/10, CPU prototype) | `13-policy-prototype.md` |
| T3 map mutation under CUDA graph | **PASS** (bitwise) | `../runs/m4-swap/report.md` |
| T4 row-write fidelity ×3 + rollback | **PASS** (bitwise) | `../runs/m4-swap/report.md` |
| T5 torn-update fault injection | **not run** — `step_hook` seam exists | — |
| T6 cross-rank agreement | **not run**; 4-rank agreement observed incidentally during M3 swaps | `../runs/m3-reload/report.md` |
| T7 24 h soak | **not run** | — |
| T8 rehydration / crash recovery | **not run as specified** (no kill -9 leg); the M3 gate met the "logits == fresh boot" half | `../runs/m3-reload/report.md` |
| T9 quality ladder | **not run**; probe reference captured (32 held-out prompts, mean logprob −1.7589 on the K3 serve) | `../runs/probe-reference/` |
| Occupancy < capacity (05 §2.1) | **PASS** (7/7 bitwise + leakage control) | `../runs/pre-m4-checks/occupancy-gpu-report.md` |

Everything green above is on a **proxy** (SIQ-Fruit, 10 MoE layers, or a
toy E=32 layer for T3/T4) — see `14-build-findings.md` §12.
