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

## T4 — Row-write fidelity test (GPU)

For one expert: load its K4 rows into a slab that previously held its K3
encoding (and vice versa), update suh/svh rows, rebuild maps, forward a probe
batch. Reference: a layer built from scratch with that membership. Assert
bitwise-equal logits (same kernels, same inputs — equality, not tolerance).
Parametrize over: expert at slab row 0, last row, middle; w13 and w2.
(If K6 resolves to slab-rebuild-only, this test becomes the per-layer slab
rebuild fidelity test — same reference structure.)

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
