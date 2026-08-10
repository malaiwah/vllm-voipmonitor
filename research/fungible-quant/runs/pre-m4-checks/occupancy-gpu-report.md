# Occupancy < Capacity GPU Test — b12x Mixed-Trellis MoE Kernel

**Verdict: PASS** (all 7 cases, reproducible across two runs, exit 0)

Last open item of the pre-M4 verification checklist. Tests the claim from
design doc `05-variable-cardinality.md` §2: the mixed K3/K4 trellis kernel
tolerates capacity rows that no descriptor references — garbage bytes in
unreferenced slots never leak into live-routed outputs.

- Test: `research/fungible-quant/runs/pre-m4-checks/test_occupancy_under_capacity.py`
- Run: `env CUDA_VISIBLE_DEVICES=4 .../gg-env/gg-run.sh python test_occupancy_under_capacity.py`
  from a neutral cwd (`/home/mbelleau/fq-0c/occupancy-test/`)
- Hardware: RTX PRO 6000 (SM120), torch 2.12+cu132, installed b12x 1.1.0
  (imports `b12x.moe._shared.kernels.w4a16.{mixed_trellis,host,kernel,prepare}`
  from the wrapper env — verified identical public API to the fresh
  `/home/mbelleau/src/b12x` master clone; installed adds only route-pack
  warmup and LUT ABI slots, both internal)

## Exact configuration

| Parameter | Value |
|---|---|
| hidden / intermediate | 128 / 128, tile config (128, 128, 128, 128) |
| tokens m / top_k | 8 / 4, `moe_block_size=8`, `max_m_blocks=18` (from `max_packed_route_slots(32, 8, 16)=144`) |
| Capacity C | tier0 = 12 K3 slots, tier1 = 4 K4 slots → 16 combined slots |
| Global namespace | 16 ids, tiers interleaved: tier0 globals (0,1,2,4,5,6,8,9,10,12,13,14), tier1 globals (3,7,11,15) via `build_tiered_maps` |
| Occupancy N | 10 live experts (8 K3 + 2 K4) |
| Retired slots | tier0 locals {2,5,7,10} + tier1 locals {1,3} → combined slots {2,5,7,10,13,15}, globals {2,6,9,13,7,15} |
| Occupancy maps | `global_to_combined[retired globals] = -1`; `descriptor[retired combined slots] = -1` |
| Routing | int32 `topk_ids`, live globals only; covers every live expert and the highest live id of each tier (14, 11); weights = softmax fp32; x = bf16 · 1e-3 |
| Launch | one `compile_mixed_trellis` result serves every configuration (same compiled object); fresh `make_mixed_trellis_buffers` per forward so stale clean state cannot mask a skipped write |
| Weights | `prepare_trellis256_moe_weights` synthetic oracle path, `w13_layout="trellis3_t256_proj"`, per-expert suh/svh/intermediate rotations — mirrors `tests/moe/test_w4a16_mixed_trellis.py::_prepared` |

## What was scribbled where

Only rows that **no live mapping can reach** (retired combined slots are
unreachable: their globals map to -1 and no other global maps to them):

- **Weight slabs (per retired tier-local expert):** full-range random int32
  over the expert's rows of `w13` — both projection halves, since
  `trellis3_t256_proj` backing is projection-major `[2, E, H/16, I/16, 16*bits]i16`
  — and of `w2` (expert-major `[E, I/16, H/16, 16*bits]i16`).
- **Global scales:** `w13_global_scale[retired] = NaN`, `w2_global_scale[retired] = NaN` (fp32).
- **Rotation tables (per retired combined slot, in the combined tier-ordered
  tables from `combine_trellis_rotations`):** NaN fp16 rows in
  `intermediate` ([16, 3·I]), `gate_suh`, `up_suh`, `down_svh` ([16, H]).
- **Not scribbled:** `w13_scale`/`w2_scale` (4-byte dummy ABI tensors, not
  per-expert), and any row reachable via `global_to_combined` — per the
  source-phase finding that reachable rows must stay valid.

## Case results

| Case | Check | Result |
|---|---|---|
| determinism | clean occupancy config twice, fresh buffers → `torch.equal` | PASS |
| occ-vs-full | occupancy maps vs full maps (fresh 16-expert layer), same live-only routing → `torch.equal` | PASS |
| garbage-k3-slabs | garbage only in unreferenced K3 (tier0) slab rows → `torch.equal` vs clean reference | PASS |
| garbage-k4-slabs | garbage only in unreferenced K4 (tier1) slab rows → `torch.equal` | PASS |
| garbage-all | both slabs + NaN global scales + NaN rotation/suh/svh rows → `torch.equal` | PASS |
| serial-oracle | occupancy output vs serial per-tier `run_w4a16_moe` oracle | PASS, rel = 4.7e-08 (bound 4e-3) |
| garbage-visible-control | full maps + routing that references the scribbled slots: clean vs scribbled must differ | PASS — output differs; 1024/1024 output elements NaN |

The control case proves the harness is sound in both directions: the same
scribbles that are invisible under occupancy-N routing poison **every**
output element (8×128 NaNs) the moment the slots are actually referenced,
so the bitwise-equal cases are not vacuous.

## Caveats

- Bitwise equality needed no special accommodations: no capacity alignment
  tricks, int32 route ids (the default; int64 not exercised here — the
  upstream b12x test covers both dtypes for the mixed path), one shared
  compiled launch, fresh buffers per run.
- The "fresh-built layer with exactly the N experts" comparison is realized
  at the same capacity C (full maps + live-only routing, `occ-vs-full`);
  a fresh build at a *smaller* capacity would change route packing geometry
  and is not required by the claim. Correctness of the N-expert math is
  instead grounded by the serial per-tier oracle (rel 4.7e-08).
- `descriptor[retired] = -1` is set per the design doc; it is
  belt-and-suspenders — route packing already drops routes whose
  `global_to_combined` entry is negative, so those descriptor slots are
  never read under live-only routing.
