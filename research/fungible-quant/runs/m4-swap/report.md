# M4 atomic swap engine — T3/T4 GPU verdicts — 2026-08-10

Deliverable: `exl3_fungible/swap.py` (gg-vllm `fq/m1-stats-collector`,
commit `a16c87f73`) + `tests/exl3_fungible/test_swap_{cpu,gpu}.py` /
`toy_segments.py`. All GPU work on GPU 7 (RTX PRO 6000 Blackwell, SM120,
0% util / 0 MiB before launch, re-checked pre-flight) through
`runs/gg-env/gg-run.sh` (r33 rootfs, torch 2.12.0+cu132, r33 b12x).

## Verdicts

| Test | Verdict | What it proves |
|---|---|---|
| **T3 — map mutation under CUDA graph** | **PASS (bitwise)** | Forward captured in a CUDA graph; map CONTENTS mutated in place through the engine's rebuild path (`copy_` into the live `global_to_combined`/`descriptor_map`, data_ptr unchanged); replay output `torch.equal` to a fresh-built layer with the new membership. **Maps are read as data, not baked at capture — the atomic swap path is live; APPLY_MODE=reload is NOT the ceiling.** Non-vacuous: the mutated-map replay differs from the pre-mutation output, and the membership permutation included cross-tier moves (K3<->K4 slot re-attribution) plus within-tier reorders. |
| **T4 — row-write fidelity x3** (slab rows first / last / middle) | **PASS (bitwise)** | One expert pair swapped end-to-end by `SwapEngine.apply()` from real fq-segment/1 byte payloads (toy segments in fq_repack's exact layout: per-expert contiguous body, `index-kK.json` ranges, per-proj/rank trellis+suh+svh+mcg). Post-swap slabs, combined rotation/suh/svh tables and both maps `torch.equal` to a fresh-built layer with that membership; probe-batch forward bitwise-equal. Rollback (`apply(plan.inverse())`, fragments re-read from segments) restores the pre-swap output bitwise. |
| CPU contract tests | 13/13 (package 41/41) | SwapPlan diff/inverse algebra (policy.py conventions), LocalSegmentSource byte fidelity + fail-closed range/dtype/mcg checks, full apply()/rollback byte fidelity on hand-assembled layer state, commit-step ordering, PolicyStore persist on commit, geometry/broadcast-layout/hole refusals. |

## Timings (toy layer E=32: 24 K3 + 8 K4, H=I=128, GPU 7)

`apply()` measured around the quiesce window (H2D copies + map flip +
sync-on-exit); staging (segment IO from disk cache into pinned host,
slot resolution, map rebuild + validation) runs pre-quiesce. Best of 6
(3x plan + inverse):

| Pairs | Window (best / median) | Stage (best) | H2D bytes |
|---|---|---|---|
| 1 | **0.061 ms** / 0.065 ms | 0.114 ms | 0.046 MB |
| 8 | **0.368 ms** / 0.369 ms | 0.641 ms | 0.369 MB |

Toy payloads are ~350x smaller per expert than GLM-5.2 rank shards
(46 KB vs ~7.9 MB/pair staged), so these numbers validate the fixed
overhead of the window (per-pair op issue + map flip + sync), not PCIe
transfer time; at caps (64 pairs ~ 504 MB) the 02-swap-engine H2D budget
estimate (~25-50 ms) remains governed by link bandwidth. Zero device
allocations inside the window (all writes are `copy_` into existing slabs
/ tables from pre-allocated pinned staging on the caller's stream); no
host reads of device data anywhere in the apply path (PERFORMANCE.md).

## Design consequences honored (pre-m4-checks/report.md)

1. Engine never expresses absence: rebuilt `global_to_combined` must be a
   full permutation (fail-closed check before the live copy).
2. Rotation/suh/svh writes target the COMBINED tensors at combined-slot
   indices; per-tier sources ignored. A tier swap rewrites BOTH experts'
   slab rows and all four combined-table rows at their NEW slots.
3. Broadcast suh/svh layouts refused at engine construction (v1 targets
   per-expert layout).
4. Rebuilt descriptors validated to be exactly `local` / `256|local`.
5. All device writes on the caller's stream inside the caller-provided
   quiesce context; commit order slabs -> rotations -> maps -> memo ->
   generation+persist (`step_hook` seam ready for T5 torn-update tests).

Slot assignment: `e_in` inherits `e_out`'s tier1 local slot and vice
versa — every other expert's slot, slab row and combined row is untouched,
so a 1-pair swap moves exactly 2 slab-row groups + 8 combined rows + 2 maps.

MCG: staged fragments must agree on the layer codebook word (fruit
segments: `-877912083` = 0xCBAC1FED across all experts/projs sampled);
foreign-mcg fragments are refused at staging (r33 pins the MCG LUT ABI —
no per-expert mcg plumbing exists in the mixed kernel).

## Seams left for integration (next, not in this run)

- `MixedLayerState.from_exl3_mixed_trellis()` adapts GG's live
  `layer.exl3_mixed_trellis` dict directly.
- `FragmentSource` is the loader-v2 seam: the concurrent
  `fragments.py` FragmentResolver (HF ranged reads + sha verification)
  can back it by filling an `ExpertStage` from its resolved bytes; the
  per-tensor-range-within-expert-range validation done here is exactly
  the property that makes one ranged GET per expert sufficient.
- Quiesce is caller-provided (engine pause path); `memo_hook` nulls the
  FusedMoEQuantConfig memo; `policy_store`/`policy_doc` persist the
  committed policy (write-temp + atomic rename via store.PolicyStore).

Full test log: 5 passed in 12.44s (T3, T4x3, timing);
`tests/exl3_fungible/test_swap_gpu.py` is also directly runnable via
`CUDA_VISIBLE_DEVICES=<free> gg-run.sh python .../test_swap_gpu.py`.
