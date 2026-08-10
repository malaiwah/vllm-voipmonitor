# M1 overhead + M2 dryrun — measured 2026-08-10 (Fruit proxy, TP4)

Rig: mixed-K Fruit serve (10 MoE layers × 256 experts), TP4 on GPUs 0-3,
CUDA graphs on, `VLLM_FQ_APPLY_MODE=dryrun`, real ε from the 0c campaign.
Assembled by the orchestrator from the benchmark artifacts and serve logs
in this directory after the running agent was cut short by a spend limit;
every number below is read from those files, none re-derived by hand.

## M1 decode overhead — **GATE NOT MET as measured**

Median decode tok/s over repeated runs (`bench-{off,on}-c{1,4}-r*.json`,
n=30 @ cc1, n=600 @ cc4, 96 max_tokens, fixed seeds):

| Arm | cc1 | cc4 |
|---|---|---|
| `VLLM_FQ_ENABLE=0` | 461.64 | 1355.74 |
| `VLLM_FQ_ENABLE=1` | 438.36 | 1305.11 |
| **overhead** | **+5.04 %** | **+3.73 %** |

The M1 gate is **< 0.5 % decode overhead at cc8**. As measured on this
proxy the collector costs 4-5 %, i.e. **7-10× the budget**. Reported as a
failure rather than rounded away.

Why the proxy is the wrong yardstick — and why it must be re-measured on
GLM-5.2 before any pass/fail is claimed:

- The collector's cost is **fixed per layer per step** (a handful of small
  kernels: flatten, bounds-mask, `where`, two `scatter_add_`), while the
  work it rides on scales with model size. Fruit's MoE GEMMs are ~150×
  smaller than GLM-5.2's (hidden 1024 vs 6144, intermediate 512 vs 2048),
  so the same fixed cost lands as a far larger *fraction*.
- The gate is specified at **cc8**; these runs are cc1/cc4.

Two concrete reductions if the real-model number still misses:
1. The out-of-range guard (added after an illegal-memory-access crash in
   this very session) costs ~3 of the ~6 kernels. It can be folded into a
   single `clamp` + one corrective subtract, or gated behind a debug env
   once routing ids are trusted for a given model+backend.
2. `count` and `mass` can share one fused scatter into a `[2, E]` buffer,
   halving the launches.

**Action:** re-measure on the GLM-5.2 serve at cc8 before M1 is called
done. Until then M1 is code-complete and T1-proven, with an open
performance gate.

## M2 dryrun — **working end to end**

`loop.py` drove the full observe → decide → explain → persist cycle
against live traffic: **208 decision lines** across the run.

```
FQ interval step=3000:  0 swaps across 0 layers (blocked: dwell=333 hysteresis=0 cap=…) policy … -> …
FQ interval step=81000: 0 swaps across 0 layers (blocked: dwell=0 hysteresis=6 cap=332) policy 0451f1a9 -> 0451f1a9
```

What this evidences:

- **The guards are doing their job, visibly.** Early intervals are
  dwell-blocked (333 candidates held by the dwell timer); by step 81000
  dwell has expired and the *cap* becomes the binding constraint (332
  candidates over the per-interval limit) with 6 more held by hysteresis.
  Zero swaps proposed is the correct answer here: the traffic is a fixed
  synthetic prompt mix, so the routing distribution never shifts enough to
  beat the incumbent set by the hysteresis factor.
- **Cross-rank agreement in a live engine**: every interval line is
  emitted identically by all four TP ranks (`Worker_TP0..TP3`), matching
  the T6 result obtained offline.
- **The policy hash is stable** across intervals (`0451f1a9 -> 0451f1a9`)
  — dryrun persists proposals without mutating the committed policy, as
  specified.

Caveat: because the synthetic traffic never produced a swap, this run
exercises the *decision* path and the guards, not the proposal-with-swaps
path. A run with shifted traffic (or relaxed hysteresis/dwell) is needed
to see non-empty swap lists with their per-swap rationale lines — the
decision_log unit tests cover that shape, a live demo does not yet exist.

## Not captured

`mem-{off,on}.json` contain a 404 body (the metrics endpoint was not
reachable at that point), so the runtime-memory delta is **not measured**;
`torch.cuda.memory_allocated` steady-state comparison remains open.
