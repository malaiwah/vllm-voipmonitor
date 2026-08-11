# BT-6c — the loop installs repeatedly, and the two bugs that hid it

**PASS**, demo2, 2026-08-11 20:07–20:13Z. GPUs 4–7, eager (`FQ_FAST=1`),
`VLLM_FQ_LIVE_APPLY=1`, policy `policy-demo1-fitted.json`, sustained math load
at concurrency 8. Log: `results/demo2/serve-bt6c3.log`.

## What passed

| | |
|---|---|
| install events | **2** (20:08:14, 20:12:20), all 4 ranks each |
| `invalid swap` | **0** |
| adoption fired (`differ from this interval`) | 8 = 2 × 4 ranks |
| `NameError` / failed intervals / worker deaths | 0 / 0 / 0 |
| experts moved vs the BOOT POLICY FILE | **256 across 44 layers** |
| K4 cardinality | 2658 → **2658**, preserved |

256 is exactly 2 installs × 64 swaps × 2 experts per swap, so no expert was
swapped twice and nothing was double-counted.

**The evidence is a diff of the committed policy document against the boot
policy file on disk** — deliberately *not* `/fq/layer`, which reports the
loop's own view and therefore cannot corroborate the loop (see OBS-2).

## What this does NOT prove

- **That the bytes moved on the device.** M4 separately showed a forced re-tier
  moves real weights with `delta_bytes_per_rank: 0`, and `engine.apply()`
  returns ok here, but this test verifies *membership bookkeeping*, not
  silicon. OBS-2 (a device-truth surface) is what would close that.
- **That output quality improves.** That is BT-7, and it is still unmeasured.

## The two bugs, and why they took so long

Both were on paths that only execute *after a successful install*, so neither
could appear until the engine started installing — and each one prevented the
next from being reached.

**1. Two SwapEngine objects per worker** (fixed `40b6f5e8a`).
`MixedLayerState.from_exl3_mixed_trellis` *copies* the tier orderings out of
the module, and a committed `apply()` updated only that copy. The admin API
cached its engine on `state.swap_engine`; `integration.py` built a second one.
Two private host views over one set of device maps. Fix: one engine per worker,
plus `MixedLayerState.publish()` writing committed orderings back to the module
at the same visibility point as the maps — so an engine built *later* starts
from what the device holds. (M3's `fq_reload` had this write-back right all
along, in its step 4; M4 dropped it.)

**2. `NameError: name 'K3' is not defined`** (fixed `505ffaa7b`).
The adoption branch used bare `K3`/`K4`; `loop.py` imports the policy module as
`P`. The branch only runs after `apply_fn` reports success, so it had never
executed. The swaps landed on the device and the loop then died before
recording them, leaving `tier_of` at the pre-swap membership — after which
every later interval proposed a move off a non-resident expert:

```
FQ live apply: 64 swap(s) INSTALLED
ERROR loop.py:697 FQ interval at step 300 failed - continuing
NameError: name 'K3' is not defined
...
ValueError: invalid swap (3, 51, 60): e_out must be resident K4
```

**This, not the two-engine bug, was the proximate cause of `invalid swap`.** It
also explains the two observations that made the first diagnosis wrong:
`differ from this interval` never logged because the crash preceded that line,
and the composition table reported "no tier changes" because `tier_of` never
moved.

### The test that should have caught it

`test_apply_fn_may_report_what_it_actually_installed` recomputed the adoption
arithmetic inline and asserted on its own copy. It stayed green against code
that could not execute. Rewritten to drive the loop through
`drive_hot_interval()` with an `apply_fn` returning a *different* swap than
proposed, asserting on `state.tier_of` and the committed document.

**Any branch that only runs on a success path needs a test that actually
reaches it.**

## The real limit: staging is per-expert random IO

Installs are **IO-bound, not policy-bound**. The loop proposes 64 swaps every
interval (~18 s); staging cannot keep up, so most proposals are discarded
unused.

```
20:07:29 stage 64  ->  20:08:14 INSTALLED    =  45 s   (warm page cache)
20:08:31 stage 64  ->  20:12:20 INSTALLED    = 229 s   (cold)
disk during the cold batch: 102,992 sectors / 5 s = ~10.5 MB/s, no network
```

A batch is roughly 64 swaps × 4 expert-payloads × ~9 tensors ≈ 2,300 small
ranged reads scattered over ~40 segment files, ~1.2 GB. At 10.5 MB/s that is
~115 s — consistent with what was measured.

This project already quantified the same asymmetry for the **loader**: bulk
whole-segment fetch runs at 142–149 MiB/s versus ~0.75 MiB/s per-expert, a
~190× gap, which is exactly why the loader prefers whole segments and why
`VLLM_FQ_KEEP_LAYERS=1` exists. **The swap engine never got that treatment.**

Hypothesis, flagged as such: staging should slice from the cached whole segment
— 299 GB of them are already on disk — or hold it mmap'd for the batch, rather
than issuing thousands of scattered preads. If that holds, the install cadence
moves from minutes to seconds. Tracked as PERF-1 (#74), together with the
instrumentation gap that made this take timestamp-differencing to find:
`StagedBatch` already computes `bytes_h2d` and `stage_seconds` and
`ApplyReport` carries `window_seconds`, and none of the three is logged
anywhere.

## Observability note

The periodic composition table reported *"no tier changes across 75 layers"* at
every interval while 256 experts moved. That is not a bug in the swap path:
`_occupancy_map` returns per-tier **counts**, and under D1 fixed cardinality a
paired swap preserves every count exactly. The table is structurally incapable
of showing a swap. Tracked as OBS-1 (#67), reopened — an earlier closure blamed
the `NameError`, which was wrong for this symptom.
