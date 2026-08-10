"""Pre-M4 GPU check: mixed-trellis MoE tolerates occupancy < capacity.

Claim under test (design doc 05-variable-cardinality.md section 2): the mixed
K3/K4 trellis kernel tolerates capacity rows that no descriptor references.

Setup: one mixed layer at capacity C = 16 combined slots (tier0 = 12 K3 slots,
tier1 = 4 K4 slots). A second configuration retires 6 experts (occupancy
N = 10 < C): their global_to_combined entries are -1 and their descriptor
slots are -1 (unreferenced). The unreferenced capacity rows are then scribbled
with garbage (full-range random int32 in the weight slabs, NaN in per-expert
global scales and rotation/suh/svh rows). Routing only ever selects mapped
experts, so garbage in unreferenced rows must not leak: the occupancy-N run
over the scribbled layer must be BITWISE equal (torch.equal) to the clean
reference layer holding exactly the N live experts at the same capacity.

Cases:
  determinism        clean occupancy config run twice, fresh buffers -> equal
  occ-vs-full        occupancy maps vs full maps, live-only routing -> equal
  garbage-k3-slabs   garbage in unreferenced K3 (tier0) slab rows -> equal
  garbage-k4-slabs   garbage in unreferenced K4 (tier1) slab rows -> equal
  garbage-all        garbage in both slabs + unreachable rotation rows -> equal
  serial-oracle      occupancy output matches the serial per-tier oracle
  garbage-visible    CONTROL: full maps + routing that references the
                     scribbled slots must CHANGE the output (proves the
                     scribbles land in rows the kernel actually reads)

Run:
  env CUDA_VISIBLE_DEVICES=4 \
    /home/mbelleau/protensors-work/vllm-voipmonitor/research/fungible-quant/\
runs/gg-env/gg-run.sh python test_occupancy_under_capacity.py
"""

from __future__ import annotations

import dataclasses
import sys

import torch

from b12x.moe._shared.kernels.w4a16.host import (
    make_w4a16_packed_buffers,
    max_packed_route_slots,
)
from b12x.moe._shared.kernels.w4a16.kernel import run_w4a16_moe
from b12x.moe._shared.kernels.w4a16.mixed_trellis import (
    MixedTrellisRotations,
    build_tiered_maps,
    combine_trellis_rotations,
    compile_mixed_trellis,
    make_mixed_trellis_buffers,
    run_mixed_trellis,
)
from b12x.moe._shared.kernels.w4a16.prepare import prepare_trellis256_moe_weights

# ----------------------------------------------------------------- geometry
HIDDEN = 128
INTERMEDIATE = 128
TILE_CONFIG = (128, 128, 128, 128)
M = 8
TOPK = 4
MOE_BLOCK = 8
CAP0, CAP1 = 12, 4  # tier capacities: 12 K3 slots + 4 K4 slots
TOTAL = CAP0 + CAP1  # combined capacity C = 16 (also the global namespace)

# Global expert ids deliberately interleave the two tiers (combined slots stay
# tier-ordered: tier0 locals -> combined [0,12), tier1 locals -> [12,16)).
TIER0_GLOBALS = (0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14)
TIER1_GLOBALS = (3, 7, 11, 15)

# Retired (unreferenced) slots: occupancy N = 8 K3 + 2 K4 = 10 < C = 16.
RETIRED_T0_LOCALS = (2, 5, 7, 10)  # globals 2, 6, 9, 13
RETIRED_T1_LOCALS = (1, 3)  # globals 7, 15
RETIRED_GLOBALS = tuple(
    sorted(
        [TIER0_GLOBALS[i] for i in RETIRED_T0_LOCALS]
        + [TIER1_GLOBALS[i] for i in RETIRED_T1_LOCALS]
    )
)
RETIRED_COMBINED = tuple(
    sorted(list(RETIRED_T0_LOCALS) + [CAP0 + i for i in RETIRED_T1_LOCALS])
)
LIVE_GLOBALS = tuple(i for i in range(TOTAL) if i not in RETIRED_GLOBALS)

# Live-only routing: every live expert appears, both tiers in most rows, and
# the highest live global id of each tier (14 for K3, 11 for K4) is exercised.
LIVE_ROUTES = (
    (0, 3, 14, 5),
    (1, 11, 4, 8),
    (10, 12, 3, 14),
    (5, 0, 11, 1),
    (14, 8, 12, 10),
    (4, 3, 1, 0),
    (11, 14, 5, 12),
    (8, 10, 4, 3),
)
# Control routing references retired experts (valid only under the FULL maps).
CONTROL_ROUTES = (
    (0, 2, 14, 7),
    (1, 15, 4, 8),
    (6, 12, 3, 13),
    (9, 0, 11, 1),
    (2, 8, 15, 10),
    (4, 6, 13, 0),
    (7, 14, 9, 12),
    (15, 10, 2, 3),
)


def _prepared(*, experts: int, bits: int, seed: int, device: torch.device):
    """Mirror tests/moe/test_w4a16_mixed_trellis.py::_prepared exactly."""
    generator = torch.Generator(device=device).manual_seed(seed)

    def scales(shape):
        return (
            0.875 + 0.25 * torch.rand(shape, generator=generator, device=device)
        ).to(torch.float16)

    return prepare_trellis256_moe_weights(
        hidden_size=HIDDEN,
        intermediate_size=INTERMEDIATE,
        num_experts=experts,
        activation="silu",
        fc1_tile_n=TILE_CONFIG[1],
        fc2_tile_n=TILE_CONFIG[3],
        device=device,
        seed=seed,
        params_dtype=torch.float16,
        w13_layout="trellis3_t256_proj",
        trellis_bits=bits,
        gate_suh=scales((experts, HIDDEN)),
        up_suh=scales((experts, HIDDEN)),
        intermediate_rotations=scales((experts, 3 * INTERMEDIATE)),
        down_svh=scales((experts, HIDDEN)),
        tile_config=TILE_CONFIG,
    )


def _scribble_tier(tier, retired_locals, generator):
    """Garbage the weight-slab rows + global scales of retired local slots.

    w13 for w13_layout='trellis3_t256_proj' is projection-major
    [2, E, H/16, I/16, 16*bits]i16, so both projection halves of each retired
    expert row are scribbled. w2 is plain expert-major [E, I/16, H/16, ...].
    """
    experts = int(tier.num_experts)
    device = tier.w13.device

    def int_garbage(shape):
        return torch.randint(
            torch.iinfo(torch.int32).min,
            torch.iinfo(torch.int32).max,
            shape,
            dtype=torch.int32,
            device=device,
            generator=generator,
        )

    w13 = tier.w13.clone()
    w13_rows = w13.view(2, experts, -1)
    w2 = tier.w2.clone()
    w2_rows = w2.view(experts, -1)
    for local in retired_locals:
        w13_rows[:, local] = int_garbage(w13_rows[:, local].shape)
        w2_rows[local] = int_garbage(w2_rows[local].shape)
    w13_gs = tier.w13_global_scale.clone()
    w2_gs = tier.w2_global_scale.clone()
    idx = torch.tensor(retired_locals, dtype=torch.long, device=device)
    w13_gs[idx] = float("nan")
    w2_gs[idx] = float("nan")
    return dataclasses.replace(
        tier, w13=w13, w2=w2, w13_global_scale=w13_gs, w2_global_scale=w2_gs
    )


def _scribble_rotations(rotations, retired_combined, device):
    """NaN-fill rotation/suh/svh rows of combined slots no mapping can reach."""
    idx = torch.tensor(retired_combined, dtype=torch.long, device=device)

    def poison(table):
        poisoned = table.clone()
        poisoned.view(TOTAL, -1)[idx] = float("nan")
        return poisoned

    return MixedTrellisRotations(
        intermediate=poison(rotations.intermediate),
        gate_suh=poison(rotations.gate_suh),
        up_suh=poison(rotations.up_suh),
        down_svh=poison(rotations.down_svh),
    )


def _serial_tier(x, prepared, topk_weights, topk_ids, expert_map):
    """Reconstruction oracle: single-tier packed path, expert_map over the
    16-wide global namespace (mirrors the b12x test's _serial_tier)."""
    buffers = make_w4a16_packed_buffers(
        prepared,
        m=M,
        topk=TOPK,
        dtype=torch.float16,
        device=x.device,
        route_num_experts=int(expert_map.numel()),
        full_rotation=True,
        block_size_m=MOE_BLOCK,
    )
    assert buffers.rotation_a_gate is not None
    assert buffers.rotation_a_up is not None
    return run_w4a16_moe(
        x,
        prepared,
        topk_weights,
        topk_ids,
        activation="silu",
        intermediate_cache13=buffers.intermediate_cache13,
        intermediate_cache2=buffers.intermediate_cache2,
        output=buffers.output,
        fc1_c_tmp=buffers.fc1_c_tmp,
        fc2_c_tmp=buffers.fc2_c_tmp,
        packed_route_indices=buffers.packed_route_indices,
        block_expert_ids=buffers.block_expert_ids,
        packed_route_count=buffers.packed_route_count,
        expert_offsets=buffers.expert_offsets,
        expert_counts=buffers.expert_counts,
        expert_map=expert_map,
        output_expert_map=expert_map,
        route_block_size_m=MOE_BLOCK,
        intermediate_rotation_scales=prepared.intermediate_rotations,
        full_rotation=True,
        suh_gate_table=prepared.gate_suh,
        suh_up_table=prepared.up_suh,
        svh_table=prepared.down_svh,
        rotation_a_gate=buffers.rotation_a_gate,
        rotation_a_up=buffers.rotation_a_up,
    )


def main() -> int:
    torch.manual_seed(20260810)
    if not torch.cuda.is_available():
        print("FAIL: CUDA is not available")
        return 1
    device = torch.device("cuda", torch.cuda.current_device())
    major, minor = torch.cuda.get_device_capability(device)
    if major != 12:
        print(f"FAIL: requires SM120/SM121, found SM{major}{minor}")
        return 1
    props = torch.cuda.get_device_properties(device)
    sms = int(props.multi_processor_count)

    # ---- layer at capacity C: all 16 expert slots hold valid fresh weights.
    tier0 = _prepared(experts=CAP0, bits=3, seed=301, device=device)
    tier1 = _prepared(experts=CAP1, bits=4, seed=401, device=device)
    rotations = combine_trellis_rotations(tier0, tier1)

    # ---- maps.
    full_g2c, full_descriptor = build_tiered_maps(
        TIER0_GLOBALS, TIER1_GLOBALS, device=device
    )
    occ_g2c = full_g2c.clone()
    occ_g2c[torch.tensor(RETIRED_GLOBALS, dtype=torch.long, device=device)] = -1
    occ_descriptor = full_descriptor.clone()
    occ_descriptor[
        torch.tensor(RETIRED_COMBINED, dtype=torch.long, device=device)
    ] = -1
    assert int((occ_g2c >= 0).sum()) == len(LIVE_GLOBALS)

    # ---- token batch and routing.
    x = (torch.randn((M, HIDDEN), device=device) * 1.0e-3).to(torch.bfloat16)
    live_ids = torch.tensor(LIVE_ROUTES, dtype=torch.int32, device=device)
    control_ids = torch.tensor(CONTROL_ROUTES, dtype=torch.int32, device=device)
    topk_weights = torch.softmax(
        torch.randn((M, TOPK), dtype=torch.float32, device=device), dim=-1
    )

    # ---- one compiled launch serves every configuration below.
    route_slots = max_packed_route_slots(M * TOPK, MOE_BLOCK, TOTAL)
    launch = compile_mixed_trellis(
        size_m=M,
        hidden_size=HIDDEN,
        intermediate_size=INTERMEDIATE,
        tier0_num_experts=CAP0,
        tier1_num_experts=CAP1,
        top_k=TOPK,
        max_m_blocks=(route_slots + MOE_BLOCK - 1) // MOE_BLOCK,
        sms=sms,
        max_shared_mem=int(props.shared_memory_per_block_optin),
        force_tile_config=TILE_CONFIG,
        moe_block_size=MOE_BLOCK,
        route_ids_dtype=torch.int32,
    )

    def forward(tiers, g2c, descriptor, rot, topk_ids):
        # Fresh buffers per run so a skipped write cannot be masked by clean
        # state left behind by an earlier run.
        buffers = make_mixed_trellis_buffers(launch, device=device, sms=sms)
        out = run_mixed_trellis(
            x,
            tiers[0],
            tiers[1],
            topk_weights,
            topk_ids,
            g2c,
            descriptor,
            rot,
            launch,
            buffers,
        )
        torch.cuda.synchronize(device)
        return out.clone()

    # ---- scribbled variants (garbage only in rows no live mapping reaches).
    generator = torch.Generator(device=device).manual_seed(0xDEADBEEF)
    tier0_garbage = _scribble_tier(tier0, RETIRED_T0_LOCALS, generator)
    tier1_garbage = _scribble_tier(tier1, RETIRED_T1_LOCALS, generator)
    rotations_garbage = _scribble_rotations(rotations, RETIRED_COMBINED, device)

    clean = (tier0, tier1)
    occ = (occ_g2c, occ_descriptor)
    full = (full_g2c, full_descriptor)

    reference = forward(clean, *occ, rotations, live_ids)
    if torch.isnan(reference).any():
        print("FAIL: clean occupancy reference already contains NaN")
        return 1

    results: list[tuple[str, bool, str]] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        results.append((name, ok, detail))
        print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))

    # 1. Determinism of the harness across fresh buffer allocations.
    repeat = forward(clean, *occ, rotations, live_ids)
    check("determinism", torch.equal(repeat, reference))

    # 2. Occupancy-N maps vs the fresh full-capacity layer with the same
    #    live-only routing: the N-expert configuration must be bitwise
    #    identical to the layer that "really" has those experts.
    full_clean = forward(clean, *full, rotations, live_ids)
    check("occ-vs-full", torch.equal(full_clean, reference))

    # 3. Garbage in unreferenced K3 slab rows only.
    out = forward((tier0_garbage, tier1), *occ, rotations, live_ids)
    check("garbage-k3-slabs", torch.equal(out, reference))

    # 4. Garbage in unreferenced K4 slab rows only.
    out = forward((tier0, tier1_garbage), *occ, rotations, live_ids)
    check("garbage-k4-slabs", torch.equal(out, reference))

    # 5. Garbage everywhere no live mapping can reach: both tiers' slab rows,
    #    per-expert global scales, and rotation/suh/svh rows of the retired
    #    combined slots.
    out = forward((tier0_garbage, tier1_garbage), *occ, rotations_garbage, live_ids)
    check("garbage-all", torch.equal(out, reference))

    # 6. The occupancy output is the RIGHT answer, not merely a stable one:
    #    compare against the serial per-tier reconstruction oracle.
    map0 = torch.full((TOTAL,), -1, dtype=torch.int32, device=device)
    map1 = torch.full((TOTAL,), -1, dtype=torch.int32, device=device)
    for local, global_id in enumerate(TIER0_GLOBALS):
        if local not in RETIRED_T0_LOCALS:
            map0[global_id] = local
    for local, global_id in enumerate(TIER1_GLOBALS):
        if local not in RETIRED_T1_LOCALS:
            map1[global_id] = local
    serial = _serial_tier(x, tier0, topk_weights, live_ids, map0) + _serial_tier(
        x, tier1, topk_weights, live_ids, map1
    )
    torch.cuda.synchronize(device)
    relative = float(
        (reference - serial).norm() / serial.norm().clamp_min(1.0e-12)
    )
    check("serial-oracle", relative < 4.0e-3, f"rel={relative:.3e}")

    # 7. CONTROL: prove the scribbles sit in rows the kernel actually reads
    #    when they ARE referenced. Under the full maps, with routing that
    #    selects the retired experts, the same scribbles must change the
    #    output (NaN scales/rotations make any read visible).
    control_clean = forward(clean, *full, rotations, control_ids)
    control_garbage = forward(
        (tier0_garbage, tier1_garbage), *full, rotations_garbage, control_ids
    )
    check(
        "garbage-visible-control",
        not torch.equal(control_garbage, control_clean),
        f"garbage output NaNs={int(torch.isnan(control_garbage).sum())}",
    )

    failed = [name for name, ok, _ in results if not ok]
    print()
    if failed:
        print(f"VERDICT: FAIL ({', '.join(failed)})")
        return 1
    print(
        "VERDICT: PASS - occupancy "
        f"N={len(LIVE_GLOBALS)} under capacity C={TOTAL} "
        f"(tier0 {CAP0} K3 slots, tier1 {CAP1} K4 slots); garbage in "
        f"unreferenced rows (combined slots {list(RETIRED_COMBINED)}) never "
        "leaks into live-routed outputs (bitwise-equal)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
