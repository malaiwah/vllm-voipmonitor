# Pre-M4 b12x verification checklist — source phase — 2026-08-10

Method: 4 parallel adversarial verify agents over
`b12x/moe/_shared/kernels/w4a16/mixed_trellis.py` (fresh master `7cecbb2`,
1524 lines — the module the original K6 audit clone lacked), instructed to
hunt for disconfirming evidence. Full verdicts with line-cited evidence in
the workflow journal; synthesis below. Image-tree delta (r33 composed vs
master) diffed separately: 78 lines — LUT ABI slots pinned to 0 for MCG,
`warmup_mixed_trellis_route_pack` added (refuses to run during capture),
imports. **No verdict affected**; env import of the same symbols validated
via `../gg-env/gg-run.sh` against the extracted r33 rootfs.

| # | Check (02-swap-engine §Pre-M4 / 05 §2) | Verdict | Key evidence |
|---|---|---|---|
| 1 | Map semantics match `(tier<<8)\|local` | **PASS** | build: `mixed_trellis.py:1118` `(1<<8)\|i`; decode: `:342-345` `descriptor>>8`, `&0xFF`; single decode site; tier size capped 256 (`:1102`) |
| 2 | No host-side reads of map contents at launch | **PASS** | all host touches metadata-only (dtype/device/numel/data_ptr, `:1268,1432,1496`); grid = compiled constants (`:1466`); zero `.item()/.tolist()/.cpu()` hits on map values |
| 3 | `combine_trellis_rotations` aliases or copies | **COPIES** (`torch.cat(...).contiguous()`, `:1125-1136`) | forward binds pointers ONLY from the combined struct (`:1447-1502`); per-tier sources dead after prepare |
| 4 | Occupancy < capacity tolerated | **PASS** (source half) | fully gather-driven: tile expert from route metadata (`:339-345`), skipped tiles no-op; no dense per-expert capacity loop; grid never expert-sized |

## Design consequences (bind on `02-swap-engine.md`)

1. **Absence/holes must be expressed in `global_to_combined`** (set −1 /
   out-of-range): route packer drops such routes (`route_pack.py:177-178`)
   and topk_sum zeroes them (`kernel.py:7960-7967`). Marking only
   `descriptor_map` is a **silent-garbage bug**: the route still packs, GEMM
   tiles skip, and topk_sum blends never-written fc2 rows into output.
2. **Rotation/suh/svh writes target the COMBINED tensors** at
   combined-slot indices (tier0 slots `[0,t0)`, tier1 `[t0,t0+t1)`):
   `rotations.intermediate` rows of `3*intermediate_size`, `gate_suh`/`up_suh`
   /`down_svh` rows of `hidden_size`. The per-tier source tensors are dead
   after prepare. A tier swap changes BOTH experts' combined slots → both
   experts' rotation/suh/svh rows must be rewritten at their new slots, plus
   both maps, inside the quiesce window (extends 02's row inventory; commit
   ordering unchanged).
3. **`broadcast_suh`/`broadcast_svh` compile modes** (shared-H checkpoints,
   r19+) have ONE shared row — per-expert suh/svh writes don't exist there.
   For FQ v1 target the per-expert (non-broadcast) layout; shared-H
   interaction deferred (note: under shared-H, suh/svh need no per-expert
   swap writes at all — only trellis rows + `mcg` + rotations.intermediate).
4. **Descriptor writes must be exactly `local` or `256+local`** — tier is
   the unmasked upper bits (`>>8` on all bits); stray high bits change tier.
5. **Stream discipline**: maps are read at replay time on the compute
   stream; content mutation must be ordered against replay (quiesce window
   covers this; no cross-stream copy_ during serving).
6. **Occupancy-under-capacity residual (GPU half, pending)**: forward with
   occupancy N&lt;C must be bitwise-equal to fresh-built layer at N — planned
   with the T3/T4 harness (reuse `tests/moe/test_fused_moe_trellis.py`
   oracle). Spare slab rows may hold garbage; any rotation/suh row still
   reachable via `global_to_combined` must stay valid.

## Carried item (from GG surface map)

Checkpoint-form tensors are freed post-prepare (`exl3.py:1706-1710`) — the
artifact pair (Progressive Tensors segments) is the only source for both
directions of a swap. Unchanged.

## Environment note

All GPU work runs through `runs/gg-env/gg-run.sh`: extracted r33 rootfs
(`/home/mbelleau/rootfs/gg-v20-r33`) via the image's own ld-linux +
library path + custom NCCL preload (`ncclCommResume` symbol) + cutlass-DSL
`.pth` path materialized. Validated: torch 2.12.0+cu132, vllm r33, b12x
1.1.0, cutlass DSL 4.6.0, 8× SM120 visible, bf16 matmul on GPU 4 OK
(phantom-util cosmetic, per operator).
