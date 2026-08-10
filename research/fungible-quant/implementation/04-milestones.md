# 04 — Milestones & build order

Each milestone ships something runnable and retires a named risk. Estimates
assume one experienced contributor with GG-stack familiarity, part-time.
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

## M4 — Atomic swap engine (≈2-3 wk; the hard one)

`exl3_fungible/swap.py` per `02-swap-engine.md` (variant chosen by K6 report):
NVMe→pinned staging→side-stream H2D→row/slab writes→map rebuild→quiesced
commit→probe→rollback.
**Gate:** T3, T4, T5 green; swap-interval stall < 1 engine step; T7 soak 24 h.

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
