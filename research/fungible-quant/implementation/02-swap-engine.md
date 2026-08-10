# 02 — Swap engine (`exl3_fungible/swap.py`)

Design finalized against the K6 audit (`k6-sparkinfer-mixed-trellis.md`) and
the GG surface map (`gg-integration-surface.md`). **The row-write variant is
selected** — the fallback (per-layer slab rebuild) is retired from v1 scope.

## K6 verdict, applied

- `prepare_trellis256_moe_weights` is **zero-copy**: tier objects alias the
  slab tensors we hand them (view-only; data_ptr aliasing asserted in
  sparkinfer's own `test_fused_moe_trellis.py:625-626`). Mutating slab rows
  mutates what the kernel reads. No repack, no permutation, no scale
  co-packing.
- Slab layout is expert-major, checkpoint-native EXL3 tiles:
  w13 `[2, E_t, H/16, I/16, 16·bits] i16`, w2 `[E_t, I/16, H/16, 16·bits] i16`.
  One expert = **three contiguous regions** (gate, up, down) totaling
  3.375 MiB (K3) / 4.5 MiB (K4) per rank. A "repack" is a memcpy.
- `global_to_combined` (int32, global→combined slot) and `descriptor_map`
  (int32, combined→`(tier<<8)|local`) are launch arguments indexed in-kernel
  per tile — mutating their **contents** via `copy_` is CUDA-graph-safe.
- Per-expert side tensors are row-independent: suh `[E,H]`, svh `[E,H]`,
  rotations `[E,3I]`, indexed by expert id in-kernel.
- Buffers are sized by geometry only; compile is keyed by structure only —
  same-shape content rewrites never resize or recompile. (Confirms D1.)
- Two tiers and K∈{3,4,5,6} are structural (compiled two-arm dispatch;
  `_TRELLIS256_BITS`). K2/three-tier stays in M6.

## Data flow

```
fq artifact pair (NVMe)
   │  (1) stage: read e_in's K4 + e_out's K3 native tensors → pinned host
   ▼      [background thread, before quiesce; ~1 ms/expert from NVMe]
pinned staging buffer (host, ≤ MAX_SWAPS_TOTAL × 7.875 MiB ≈ 504 MiB)
   │  (2) quiesce: pause_scheduler(mode="keep")   [engine drained, graphs idle]
   ▼
   (3) H2D copy_ rows: 3 slab regions + suh/svh/rotation rows, both experts
   (4) rebuild maps: build_tiered_maps(new tier_ids) → copy_ INTO the live
       global_to_combined / descriptor_map tensors (identity preserved)
   (5) null FusedMoEQuantConfig memo (staleness guard, see GG surface map)
   (6) resume scheduler
   ▼
   (7) probe: held-out KL batch; on regression → apply inverse swap list
       (same mechanism, sources re-read from artifacts) and freeze experts
```

## Why row writes happen inside the quiesce window

In a membership swap the two destination rows are **both live**: e_in's K4
bytes land in the row the maps still route e_out to, and vice versa. Writing
them during serving would make the displaced expert compute garbage for the
duration of the copy. Two resolutions:

- **v1 (selected): write inside the quiesce window.** Worst-case payload at
  caps (2/layer × 77 layers) ≈ 1.2 GiB → ~25–50 ms of H2D on PCIe Gen4/5,
  plus map rebuild (µs) — one pause of well under 100 ms per interval
  (minutes apart). Zero extra device memory: writes land in existing slabs.
  Staging from NVMe happens *before* the pause, so the pause covers H2D only.
- **Later (optional): spare-slot ring.** +1 row per tier per layer
  (≈606 MiB/rank at 1 spare, ×cap for more) lets incoming encodings be
  written outside the pause, with only the map flip inside it (sub-ms
  commit). Buy this only if T7 shows the pause budget matters.

## Commit protocol (torn-update ordering, tested by T5)

Within the quiesce window, per layer, strictly:
1. slab rows (w13 gate+up, w2) for all swapped experts;
2. suh/svh/rotation rows;
3. map contents (the atomic visibility flip — until this step, the kernel
   still routes every expert to its old, intact encoding *except* the
   overwritten rows, which nothing references mid-window because the engine
   is paused);
4. `FusedMoEQuantConfig` memo nulled;
5. bump policy generation; persist `policy/current.json` via
   write-temp + atomic rename (startup_plan pattern).
Crash at any point before (5) → boot rehydrates the previous committed
policy; slab state is rebuilt from artifacts at startup anyway (slabs are a
cache, never authoritative).

## Rank choreography

All ranks compute identical swap lists (deterministic policy, logical
domain — see 01 §3.4). Each rank slices its own `rank_r` shard of the staged
tensors (or slices the unsharded artifact at 16-column tile granularity).
Quiesce/resume runs through the existing engine-wide pause path; no new
collectives. Debug mode: hash-check swap lists across ranks pre-apply.

## Rollback

A swap list is its own inverse. Rollback = re-stage the previous encodings
from the artifact pair and run the same protocol. The probe keeps the last
committed probe score; regression beyond `probe_regression_limit` triggers
rollback + `freeze_steps` on the offending experts. Repeated
rollback on the same experts = thrash signal (T7 gate).

## Pre-M4 verification checklist (from the K6 audit's residual uncertainty)

The clone used for the audit lacks the pinned `mixed_trellis.py` module (the
in-clone hybrid kernel is its direct ancestor; prepare/layout claims are
source-verified, mixed-kernel runtime claims inferred). Before M4 coding
starts, three one-file checks against the pinned b12x build GG actually
ships:
1. `mixed_trellis` map semantics match the ancestor encoding
   (`(tier<<8)|local`; decode sites).
2. No host-side reads of map contents at launch time (graph-safety of
   content mutation).
3. `combine_trellis_rotations` in GG: aliases the per-expert rotation rows
   (row write visible) or copies (must rewrite the combined tensor instead).

Also carried from the GG surface map: checkpoint-form tensors are **freed
post-prepare** (`exl3.py:1706-1710`) — the swap engine must never assume it
can read old encodings from the live layer; the artifact pair is the only
source of truth for both directions of a swap.

## Test hooks

Reuse sparkinfer's `test_w4a16_hybrid_moe.py` (two-tier + serial oracle) and
`test_fused_moe_trellis.py` (`_reconstruct_native` oracle, capture/replay
discipline) as the T3/T4 harness base; `benchmark_w4a16_hybrid_moe.py` for
GLM-5.2-geometry perf regression.
