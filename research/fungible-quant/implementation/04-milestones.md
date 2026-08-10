# 04 — Milestones & build order

Each milestone ships something runnable and retires a named risk. Estimates
assume one experienced contributor with GG-stack familiarity, part-time.

**Build note (2026-08-10) — status board.** Detail and evidence links in
`14-build-findings.md` §10; run reports in `../runs/`.

| | Status | One-line evidence |
|---|---|---|
| **M0** | **DONE** | all-K3 assembly **79/79 shards sha256-identical**; first mixed-K checkpoint boots TP4 and generates coherently, per-layer K4 counts == policy |
| **M1** | **partial** | T1 PASS on 1 GPU under FULL cudagraph capture; **TP4 leg and the < 0.5 % overhead measurement not done** |
| **M2** | **partial** | engine + T2 properties green (CPU prototype, `policy.py`/`store.PolicyStore` in-tree); **48 h dryrun not run** |
| **M3** | **DONE, budget beaten ~10×** | live reload-under-quiesce in **0.41–0.47 s**, 0 request drops, post-reload logits **bit-identical** to fresh boot (twice) |
| **M4** | **T3/T4 green, not integrated** | T3 PASS (maps read as data under CUDA graph), T4 PASS ×3 + bitwise rollback, window **0.061 ms**/pair; **T5/T6/T7 not run**, live-layer wiring is a seam |
| **M5** | not started | — |
| **M6** | partially pre-empted | Progressive Loader v2 (`--load-format progressive`) already boots mixed-K straight from segments + policy; K2/K5 **encode** works today (execution of K2 still kernel work) |
Phase-0 measurements (`../PLAN.md` §5) run alongside M0-M2 and set knobs —
they no longer gate the build (decision: the loop is the product), but K2
(homogeneous sensitivity) and a failed T3 remain honest abort signals.

## M0 — Artifact production toolchain (≈1 wk elapsed, mostly GPU-hours)

Produce `<GLM-5.2>-EXL3-FQ/`: full K3 + full K4 routed-expert encodings from
one measure campaign, per-expert-addressable index, manifest. Reuses the
existing exllamav3 convert/compile pipeline; new code is packaging + index
generation (~small script). K4 overlay encodes are the long pole (rentable,
one-time).
**Gate:** a script assembles ANY bits_per_expert JSON into a bootable mixed
checkpoint offline (today's static path) from the pair — proves the pair is
complete and the index is right.

**Build note (2026-08-10) — DONE, both halves.** Offline: `fq_assemble.py`'s
all-K3 round trip is **79/79 shards sha256-identical** to
`brandonmusic@9297b9f1` at full-model scale (`../runs/m0-assemble/`).
Bootable: the first true mixed-K checkpoint (`fruit-mixed-042`, K4 counts
42…152 across 10 layers) boots TP4 under the GG fork, loads exactly the
policy's partitions, and generates coherently at **~0 % throughput cost**
vs pure K3 (`../runs/serve-baseline/fruit-mixed-report.md`). Two contract
facts the spec did not have — the mandatory `quantization_config` stub and
the `bits_per_expert` `"file.json:field"` reference — are in
`14-build-findings.md` §2, along with the `fp8_ds_mla` KV requirement that
silently degrades output when omitted.

## M1 — Stats collector + persistence (≈3-4 days)

`exl3_fungible/stats.py`, capture-fn binding, window/decay, stats dump.
**Gate:** T1 (graph-freeze) green on 1 GPU and TP4; overhead < 0.5% decode
tok/s at cc8.

## M2 — Policy engine in dryrun (≈1 wk)

`exl3_fungible/policy.py` + `store.py` + Prometheus metrics + boot
rehydration. `VLLM_FQ_APPLY_MODE=dryrun`: full observe→decide→log→persist
loop, applies nothing.
**Gate:** T2 green; 48 h dryrun on live-ish traffic produces stable,
explainable swap proposals (manual review); decision hash identical across
ranks (T6 debug mode).
**This milestone already has standalone value:** its logs ARE the Phase-0a/0d
evidence on live traffic, and its persisted policy can be fed to the M0
assembler for a manual "restart into the specialized quant" workflow —
fungible quant with a reboot, before any runtime swap exists.

## M3 — Brutal apply path (≈3-4 days)

`APPLY_MODE=reload`: at interval, quiesce (`pause_scheduler(mode="keep")` or
GG equivalent per `gg-integration-surface.md`), rebuild the mixed layers from
the artifact pair under the new policy (in-place, same shapes, startup code
reused), resume. Seconds-long stall, no request drops.
**Gate:** T8 green; correctness: post-reload logits == fresh-boot logits with
same policy (T4-style reference). Retires: decide→apply lifecycle risk, policy
projection bugs, persistence bugs — everything except the atomic mechanism.

**Build note (2026-08-10) — DONE** (`../runs/m3-reload/report.md`). Measured
on a live TP4 HTTP serve under continuous traffic: **0.466 s and 0.410 s**
total stall (pause + RPC + resume), **zero request drops**, zero memory
growth, no throughput change. Correctness gate passed exactly, twice:
max |Δlogprob| = **0.0** across 356 scored tokens, both cross-process
(booted@042 vs live-swapped→042) and round-trip (042b→042→042b), with all
4 ranks' policy sha agreeing at every step. Two spec details were wrong:
the quiesce uses **`mode=wait`** (drain), not `mode="keep"` — `keep` freezes
in-flight requests across the swap (mixed-provenance KV) and `wait` needs
the multiproc serve plus `VLLM_SERVER_DEV_MODE=1`; and the gate actually met
was the logits equality — **T8's kill -9 rehydration leg was not run**.
The honest restart floor (RUNG A, warm compile caches) is **88.0 s**.

## M4 — Atomic swap engine (≈2-3 wk; the hard one)

`exl3_fungible/swap.py` per `02-swap-engine.md` (variant chosen by K6 report):
NVMe→pinned staging→side-stream H2D→row/slab writes→map rebuild→quiesced
commit→probe→rollback.
**Gate:** T3, T4, T5 green; swap-interval stall < 1 engine step; T7 soak 24 h.

**Build note (2026-08-10):** `exl3_fungible/swap.py` exists (gg-vllm
`a16c87f73`) with **T3 PASS** and **T4 PASS ×3** including bitwise rollback,
plus 13 CPU contract tests (`../runs/m4-swap/report.md`). Apply window
**0.061 ms** (1 pair) / **0.368 ms** (8 pairs) on a toy layer — that is the
*fixed* overhead (op issue + map flip + sync), not PCIe: toy payloads are
~350× smaller per expert than GLM-5.2 rank shards. **T5, T6 and T7 remain**,
and the engine is not yet wired to a live layer
(`MixedLayerState.from_exl3_mixed_trellis()` and the
`FragmentSource`→loader-v2 `FragmentResolver` bridge are named seams).

## M5 — Hardening + release (≈1 wk)

T7 full soak as release gate, T9 quality ladder, docs, dashboard, wiki page
(rtx6kpro runbook style: exact launch env, knob table, failure modes,
rollback procedure). Ship as a GG image tag.

## M6 — Extensions (post-v1 backlog, ordered by leverage)

1. **MTP-78 in-loop** with MAL-based probe and its own budget (the collector
   assets make this the cheapest high-visibility win).
2. **Workload-blended ε refresh**: periodic offline re-measure using persisted
   stats (closes the loop TASA-style: reweight, never replace).
3. **K2 tier / three tiers** — kernel work in sparkinfer (per K6 report §3);
   unlocks the downgrade half of the budget.
4. **Additive-residual encoding in exllamav3** (RRQ-style K3+plane): upgrade =
   materialize one plane; makes the artifact pair one artifact and the swap
   payload 4× smaller. The long-term right answer to D3.
5. **Policy exchange**: publish converged profiles per workload domain on HF;
   boot-time `--fq-policy hf://...`.
6. **New-model onboarding runbook**: the "day-one generic pair + converge in
   place" story as a repeatable recipe (this is the headline feature).

## Dependency graph

```
M0 ──► M3 ──► M4 ──► M5
M1 ──► M2 ──┘         │
 Phase 0 (parallel) ──┴─ knob values, announcement numbers
```

M1/M2 need no artifacts (dryrun decides against ε from the measure campaign,
which M0 produces anyway — stub ε with uniform values for pure plumbing work).
M3 needs M0+M2. M4 needs M3 green.

## Abort/downshift signals (kept from PLAN.md, reinterpreted)

- **T3 fails** (maps are baked into graphs after all): ceiling becomes M3
  (reload mode). Still shippable: specialize-with-reboot.
- **K2 fires** (0c: homogeneous sensitivity): pivot the same machinery to
  layer-level allocation (N_L redistribution across layers at boot) — the
  code survives, the granularity changes.
- **Persistent probe-rollback thrash in T7**: raise dwell/hysteresis; if
  thrash survives aggressive damping, freeze policy after convergence
  (write-once mode) — which is exactly the "stabilizes over time" product,
  still delivering the new-model story.

**Build note (2026-08-10) — the first two abort signals are RETIRED.**
**T3 passed**: maps are read as data under CUDA-graph replay, so the atomic
path is live and M3 is not the ceiling (`../runs/m4-swap/report.md`).
**K2 does not fire**: on the 0c proxy leg, Δε per expert is uniform-ish
(median CV **0.047**) but benefit = Δε·φ is strongly concentrated — median
Gini **0.48**, median top-16-of-256 benefit share **0.318** — so per-expert
allocation is the right granularity and the win comes from **routing mass**,
not ε spread (`../runs/0c-campaign/report.md`). The T7 thrash signal is
untested (T7 has not run).
