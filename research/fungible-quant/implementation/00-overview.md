# Fungible Quant — GG implementation overview

Status: implementation-ready specification. Target: **the GG stack** (vLLM
`dev/gilded-gnosis` + SparkInfer/b12x + exllamav3 toolchain), 4× RTX PRO 6000
(SM120), TP4+DCP4, EP=1, GLM-5.2-class MoE. Upstreaming is a separate,
later concern (see `../PLAN.md` §6); nothing here waits on it.

## Product statement

One generic mixed-precision quant per model, shipped once; **every deployment
specializes itself to its own traffic**. Even if the allocation stabilizes
(kill-criterion K3 in `../PLAN.md`), the loop is the delivery vehicle:

- **New-model day-one story.** When a new MoE checkpoint drops, publish a
  single two-tier artifact pair (K3 base + K4 overlay set) with a *generic*
  starting policy. No per-domain quant releases, no community arguing which
  3.x bpw variant to download. The deployment converges to its own optimum.
- **The policy is the shareable artifact.** A converged `bits_per_expert`
  JSON is a few hundred KB, topology-neutral, and describes "what a coding
  workload needs from this model." Profiles become exchangeable the way
  quants are today — at 5 orders of magnitude less bandwidth.
- **Memory-budget honesty.** Fixed cardinality means the operator picks a
  byte budget once (e.g. "3.42 bpw worth") and the system spends it optimally
  under live evidence, instead of a calibration corpus chosen months earlier
  by someone else.

## Architecture (five components)

```
            ┌────────────────────────────────────────────────┐
            │ vLLM engine (GG fork)                          │
            │                                                │
  forward   │  BaseRouter.set_capture_fn ──► StatsCollector  │  (02)
  pass      │        (graph-safe scatter_add, per layer)     │
            │                                                │
  every     │  FungibleQuantState.step() ──► PolicyEngine    │  (03)
  N steps   │        (argsort Δ_e per layer, guards)         │
            │                                                │
  async     │  SwapEngine: NVMe/host K3+K4 artifacts ──►     │  (04)
  side      │        tier-slab row writes + map rebuild      │
  stream    │        applied at quiesced step boundary       │
            │                                                │
  disk      │  PolicyStore: bits_per_expert JSON + slab      │  (01)
            │        cache, rehydrated at startup            │
            └────────────────────────────────────────────────┘
```

| # | Component | Doc | New code lives in |
|---|---|---|---|
| 1 | Artifacts & policy store | `01-artifacts-policy-stats.md` | exllamav3 toolchain + `vllm/model_executor/layers/quantization/exl3_fungible/store.py` |
| 2 | Stats collector | `01-artifacts-policy-stats.md` | `exl3_fungible/stats.py` |
| 3 | Policy engine | `01-artifacts-policy-stats.md` | `exl3_fungible/policy.py` |
| 4 | Swap engine | `02-swap-engine.md` | `exl3_fungible/swap.py` + hooks in `exl3.py` |
| 5 | Tests & validation | `03-testing-validation.md` | `tests/exl3_fungible/` |

Build order and acceptance gates: `04-milestones.md`.
Code-level integration maps (agent-audited, file:line):
`gg-integration-surface.md` (vLLM side), `k6-sparkinfer-mixed-trellis.md`
(kernel side — resolves the row-write vs slab-rebuild question).

## Decision log (settled in ../PLAN.md; restated as binding)

| # | Decision | Rationale |
|---|---|---|
| D1 | **Fixed per-layer tier cardinality; rebalance = membership permutation only.** | Tier slab shapes and `tier_signature` become invariants → no recompile, no arena resize, no CUDA-graph hazard. All-K4 headroom (86.6 GiB/rank) does not exist. |
| D2 | **Two tiers, K3/K4, matching the existing `mixed_trellis` path.** | Already shipped in GG + b12x. K2/three-tier is a later kernel project, not v1. |
| D3 | **No online re-encode in the loop. Page pre-built K3+K4 artifacts** (~607 GB NVMe). | Kills Viterbi cost, Hessian persistence (~210 GiB), stale-Hessian risk. Every published system pages; none re-encodes. |
| D4 | **Policy domain = logical expert IDs.** Policy artifact excludes rank/world_size/tp. | Topology neutrality; EP=1 makes per-expert counts logical for free. |
| D5 | **Signal = static error curve ε × routing mass w × frequency φ. Never raw hotness alone. Objective = KL, never PPL.** | Router-norm paper beats frequency baselines; TASA shows PPL τ≈0 with reasoning sensitivity. |
| D6 | **Apply at a quiesced step boundary** (`pause_scheduler(mode="keep")` or in-step barrier), metadata flip is the atomic commit point. | Torn tier-map/suh/svh updates corrupt silently. Sleep mode is broken on SM120 (#21336) — not the quiesce mechanism. |
| D7 | **Guards: hysteresis band, dwell time, per-interval swap cap, Jaccard rollback, held-out KL probe.** | EvoPress non-additivity; router-shift feedback loop is real and unstudied. |
| D8 | **Cold start = generic offline allocation, never uniform; rehydrate last policy at boot.** | The JSON loader already exists in `exl3.py`. |
| D9 | **Ship the brutal path first** (full reload under quiesce), atomic row-swap second. | Validates decide→apply end-to-end before the hard engineering. |

## What Phase 0 still gates (unchanged)

Committing to *implementation* does not skip the measurements — it reorders
them from "go/no-go" to "parameter setting":

- 0a stability (Kendall τ across windows) now sets the **rebalance interval**
  and dwell time rather than deciding the project.
- 0c sensitivity variance sets **N_L per layer** (via the global solve).
- 0d generic-vs-blended sets the **specialization claim** in any announcement.
- 0e router-shift Jaccard sets the **rollback threshold**.

Run them on the existing trace + one measure_model campaign; feed the numbers
into the knobs table in `01-artifacts-policy-stats.md` §6.
