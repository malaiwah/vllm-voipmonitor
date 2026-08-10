# 05 — Variable per-layer cardinality & generalized tiers (design note)

Question: what does it take to give layers different K3/K4 counts — and other
Kx pairs — and change those counts at runtime, under the same usage
accounting? Some layers deserve more accurate experts than others.

## Ladder of generality (cost-ordered)

| Level | Capability | Status | Cost |
|---|---|---|---|
| L0 | Different N_L per layer, fixed at startup | **Already in spec** (global budget solve, per-layer `tier_signature`) | Signature-menu guard (§3) |
| L1 | Per-layer choice of bit-pair, e.g. (K3,K4) here, (K4,K6) there | **Already representable** — signature carries bits; K∈{3,4,5,6} allowed by intrinsics | ε curves + artifact variant per K used; storage (full K6 set ≈ +519 GB) |
| L2 | **Runtime cardinality change within pre-provisioned capacity** | Capacity/occupancy split (§1) — relaxes D1 without breaking its safety argument | Overprovision memory + 1 b12x verification (§2) |
| L3 | Runtime cardinality beyond capacity | Slab reallocation → new pointers → graph recapture | Maintenance-op only (M3 reload path already does it) |
| L4 | ≥3 tiers in one layer | b12x kernel: N-arm dispatch (map encoding already has 8 tier bits: `(tier<<8)\|local`) | Kernel project (M6) |
| L5 | K2 tier | `_TRELLIS256_BITS=(3,4,5,6)` blocks it: intrinsics/codebook work | Kernel project (M6) |

**Build note (2026-08-10):** L5's blocker is **execution only**. The
sha-pinned encoder accepts **bits 2–5** (K2 and K5 smoke tests PASS on
SM120, byte-exact round trip), and K2 segments for GLM-5.2 layers 3–10 are
encoded and published today — at **~4.8 s/expert**, roughly 2× K3's cost
(`../runs/0c-campaign/MULTI-K-PLAN.md`, `glm52-encode-k2.log`). K5
*executes* on today's kernel. So the K2 fast-load base can be produced and
shipped ahead of the kernel work that will run it. L2 also gained an
operator knob during the build: **`VLLM_FQ_CAPACITY_UTILIZATION`**
(default 1.0) sets `C = ceil(N / util)` bounded by E and the global byte
budget — `util=1.0` reproduces v1 exactly, `util=0.9` pre-provisions ~11 %
spare upper-tier rows and makes displacement-free upgrades a slider rather
than an arithmetic exercise.

## 1. The capacity/occupancy split (L2 — the door-opener)

D1 ("cardinality is compiled state") conflates two things the K6 audit lets
us separate:

- **Capacity** C_L^t: slab rows allocated + launch compiled for tier t of
  layer L. Shapes, pointers, `tier_signature`, buffers — all keyed to C.
  Fixed at startup. All of D1's safety properties attach to C, not N.
- **Occupancy** N_L^t(t) ≤ C_L^t: how many rows the maps actually reference.
  Pure map data (`global_to_combined`/`descriptor_map` contents), already
  established as runtime-mutable and CUDA-graph-safe.

Under the split, a **cardinality change is just an asymmetric membership
swap**: promote expert e without demoting anyone — write e's K4 encoding
into a free K4 capacity row, flip maps; its K3 row becomes spare. The row
ring from `02-swap-engine.md`'s "spare-slot" option and cardinality growth
become the same mechanism. Commit protocol, torn-update ordering, rollback,
and rank choreography carry over unchanged.

**Invariant replacing D1's permutation rule:** every policy transaction must
be **byte-conserving at the global level** — Σ_L Σ_t N_L^t · bytes(t) ≤ B
(the startup budget), enforced structurally by the policy engine emitting
paired grow/shrink transactions (layer A gains a K4 slot only when layer B
releases one, or when global slack exists). Per-layer flexibility is bounded
by capacity; global spend is bounded by B. Memory honesty survives.

Overprovision arithmetic (per rank, 77 layers): ±k rows per direction per
layer costs k × (4.5 + 3.375) MiB × 77 ≈ k × 606 MiB. ±4 ≈ 2.4 GiB —
affordable against the ~12 GiB headroom at the 3.42 bpw operating point if
spent selectively: give wide capacity bands only to layers 0c flags as
high-variance, ±1 elsewhere.

## 2. What b12x must confirm/provide (adds to the pre-M4 checklist)

1. **Occupancy < compiled count is safe**: the mixed kernel must tolerate
   capacity rows that no descriptor references. Token-driven gather suggests
   yes (unused rows are dead memory), but any dense per-expert loop keyed to
   the compiled `tier_num_experts` (e.g. a prefetch or scale-preload loop)
   breaks it. One file read + one test (forward with occupancy N < C vs
   fresh-built layer at N: bitwise equal).

   **Build note (2026-08-10) — CONFIRMED, both halves**
   (`../runs/pre-m4-checks/occupancy-gpu-report.md`). Source: the kernel is
   fully gather-driven (tile expert read from route metadata,
   `mixed_trellis.py:339-345`); skipped tiles no-op; no dense per-expert
   capacity loop; the grid is never expert-sized. GPU: C=16 (12 K3 + 4 K4),
   N=10 with 6 slots retired, full-range random int32 scribbled into their
   slab rows and NaN into their global scales and rotation/suh/svh rows —
   **all 7 cases bitwise-equal** to the clean reference (which matches the
   fresh full-map layer and a serial per-tier oracle at rel 4.7e-08). The
   leakage control (routing that *does* reference the scribbled slots) turns
   **1024/1024 outputs NaN**, so the equalities are not vacuous. No runtime
   active-count scalar is needed (§2.2 is moot). **Hard prerequisite:**
   absence must be expressed in `global_to_combined` (−1 / out of range);
   marking only `descriptor_map` is a silent-garbage bug — see
   `02-swap-engine.md` §Commit protocol build note.
2. If the kernel needs an explicit active-count, add a runtime scalar (or
   derive from map contents) — small API addition, no layout change.
3. **Signature/buffer economics** (also L0's cost): distinct per-layer
   signatures each hold a compiled launch + buffer set in
   `_MIXED_TRELLIS_RUNTIMES` (never evicted). 77 distinct signatures may be
   fine or may be gigabytes — measure one buffer set's size. Mitigations,
   in order: quantize capacities to a small menu (e.g. C ∈ {64+r, 96+r,
   108+r, 128+r}) so layers share runtimes; or pool/share buffers across
   signatures in b12x (they are geometry-keyed workspace, sharing by max
   is natural).
4. All signatures in the menu must be compiled during the eager profile pass
   (the existing pre-capture constraint) — enumerate the menu at boot, not
   lazily.

## 3. Policy engine: two loops, same accounting

The collectors are tier-blind (per logical expert per layer) — **zero change
to usage accounting** at any level of this ladder.

- **Fast loop** (per interval, unchanged): within-layer argsort against
  current N_L^t; guards as specced.
- **Slow loop** (every ~10 intervals): re-run the global knapsack (same
  MC-MoE-form objective, same ε curves, live φ/w) over cardinalities,
  bounded by per-layer capacity bands and global bytes; emit byte-conserving
  grow/shrink transaction pairs, capped (e.g. ≤2 cardinality moves
  model-wide per slow tick), same dwell/hysteresis discipline applied at the
  layer level (a layer's N must persist ≥ dwell before moving again).
- EvoPress caution applies doubly to cross-layer moves (non-additivity is a
  layer-level phenomenon): the held-out KL probe gates slow-loop
  transactions with rollback, exactly as fast-loop swaps.

## 4. Schema & artifact changes (do these in v1 so nothing needs migration)

- `fq-policy/2`: per-layer `{"tiers": [{"k": 3, "n": 148, "cap": 152},
  {"k": 4, "n": 108, "cap": 112}]}` replaces flat `n_k4_per_layer`.
  v1 writes `cap == n` (rigid D1 behavior) — readers built now accept the
  general form. `bits_per_expert` stays as-is (it is the membership record).
- `fq-manifest/1` already lists `k_variants`; a layer's usable Ks are the
  intersection with its tier spec. Adding K6 later = add the variant dir +
  extend `k_variants` — no schema change.
- ε curves in the manifest: measure at every K in `k_variants` from day one
  (the measure campaign's marginal cost per extra K is small next to the
  encode campaign).

## 5. What to do in v1 (the actual "leave the door open" list)

1. Ship `fq-policy/2` schema with capacity fields (write cap==n).
2. Key slabs, signatures, and buffers by **capacity**, occupancy by maps —
   even while v1 never varies occupancy. Costs nothing; makes L2 a policy
   feature, not a format break.
3. Add b12x checklist items §2.1–.4 to the pre-M4 checks.
4. Keep the policy engine's decision function signature
   `(stats, eps, tier_spec) → transactions` where a transaction is
   `swap(L, e_out, e_in) | grow(L, t, e) + shrink(L', t', e')` — v1 emits
   only swaps.
5. Milestone placement: L2 lands as **M6.0** (before K2/three-tier, after
   the T7 soak), gated on §2's verification; L4/L5 stay M6 kernel projects.
