#!/usr/bin/env python3
"""Build the scenario-1 boot policy: all-K3, with the Coder quant's K4 budget.

The experiment: start every expert at K3 and give the loop exactly as many K4
slots per layer as a human used when building
`willfalco/GLM-5.2-EXL3-TR3-3.42bpw`, then let live routing decide WHICH
experts fill them. Same memory envelope, same cardinality — only the selection
differs. That makes the comparison a pure selection test rather than a
budget test, which is the only way the overlap number means anything.

Note the loop refuses to start without a policy source, and a uniform-K3
checkpoint has no mixed tier bitmap to synthesize one from — this file is that
missing input, not a workaround.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

K3, K4 = 3, 4

# Per-expert cost of a K3->K4 promotion, model-wide, in the shared_h_v1
# accounting used by reference-coder-quant.json (18,886,704 - 14,168,112).
# Divide by the tensor-parallel degree for the per-rank figure the GPU
# actually sees: 4,718,592 / 4 = 1,179,648 B/rank, which is the number the
# envelope arithmetic in convergence-demo-plan.md closes on.
PROMOTION_BYTES = 4_718_592

# Where K4 fragments are looked for when --coverage-dir is not given. These
# mirror what serve-glm52.sh puts on VLLM_FQ_MANIFEST_DIR /
# VLLM_FQ_LOCAL_SEGMENTS, so the policy cannot promise a promotion the
# resolver would then miss.
DEFAULT_COVERAGE_DIRS = (
    "/home/mbelleau/glm52-segments",
    "/home/mbelleau/fq-primed/segments-342/expanded",
)


def discover_k_coverage(
    dirs, k: int = K4, num_experts: int | None = None,
) -> tuple[dict[int, set[int]], list[dict]]:
    """Which (layer, expert) pairs actually have a K``k`` fragment on disk.

    Deliberately a mirror of ``fragments.FragmentResolver._local_segment``:
    a local dir supplies ``(layer, K)`` only when BOTH ``index-k{K}.json``
    and ``layer-{L:03d}.k{K}.safetensors`` exist and the index has an entry
    for that layer. Discovering coverage any other way — globbing the
    segment files, or trusting the manifest's advertised layer list — would
    let this script declare K4 slots the resolver then cannot fill, and a
    missed promotion is silently served as K3 while the occupancy table
    still claims K4.

    Expert-level, not layer-level, on purpose. A primed segment cut from a
    human's mixed quant contains ONLY that human's chosen K4 experts (the
    3.42bpw prime carries 50 experts for L3 and 108 for L4-L10, not 256), so
    "layer 4 has K4 coverage" is true while "expert 7 of layer 4 has K4
    coverage" is usually false. Seeding by layer would name experts that do
    not exist.

    Known conservative gap: the resolver's final authority is the segment
    FILE header (``_expert_tables_from_header``), which must be a subset of
    the index. A segment whose index lists an expert its header does not
    contain would be over-reported here — that is a corrupt segment, and the
    resolver treats it as a miss, so the run would show a failed promotion
    rather than a silent one.

    Returns ``(coverage, provenance)`` where coverage maps layer -> set of
    expert ids and provenance records what each directory contributed.
    """
    coverage: dict[int, set[int]] = {}
    provenance: list[dict] = []
    for raw in dirs:
        base = Path(raw)
        index_path = base / f"index-k{k}.json"
        note = {"dir": str(base), "index": index_path.name,
                "layers": [], "experts": 0}
        if not index_path.is_file():
            note["skipped"] = "no index-k%d.json" % k
            provenance.append(note)
            continue
        try:
            index = json.loads(index_path.read_text())
        except (OSError, ValueError) as exc:
            # A broken local dir is a MISS for the resolver too; refuse to
            # guess at what it might have held.
            note["skipped"] = f"unreadable index: {type(exc).__name__}"
            provenance.append(note)
            continue
        if not isinstance(index, dict):
            note["skipped"] = "index is not an object"
            provenance.append(note)
            continue
        for layer_key, entry in index.items():
            try:
                layer = int(layer_key)
            except (TypeError, ValueError):
                continue
            seg = base / f"layer-{layer:03d}.k{k}.safetensors"
            if not seg.is_file():
                continue
            experts = (entry or {}).get("experts")
            if not isinstance(experts, dict):
                continue
            ids = set()
            for e in experts:
                try:
                    eid = int(e)
                except (TypeError, ValueError):
                    continue
                if num_experts is None or 0 <= eid < num_experts:
                    ids.add(eid)
            if not ids:
                continue
            coverage.setdefault(layer, set()).update(ids)
            note["layers"].append(layer)
            note["experts"] += len(ids)
        note["layers"] = sorted(note["layers"])
        provenance.append(note)
    return coverage, provenance


def promotions_for_gib(gib: float, tp: int) -> int:
    """How many K3->K4 promotions fit in ``gib`` GiB of PER-RANK headroom."""
    if tp <= 0:
        raise ValueError("tp must be positive")
    per_rank = PROMOTION_BYTES / tp
    return int((gib * (1 << 30)) // per_rank)


def cap_budget(raw: dict[int, int], limit: int | None) -> dict[int, int]:
    """Scale a per-layer K4 budget down to ``limit`` total slots.

    Largest-remainder proportional scaling, ties broken by lowest layer id,
    so the same inputs always give the same policy. Layers keep their
    relative share rather than being truncated in layer order — cutting the
    tail would silently make the demo a study of the first N layers.
    """
    total = sum(raw.values())
    if limit is None or total <= limit:
        return dict(raw)
    if limit <= 0:
        return {layer: 0 for layer in raw}
    scale = limit / total
    floors = {layer: int(n * scale) for layer, n in raw.items()}
    remainder = limit - sum(floors.values())
    order = sorted(raw, key=lambda layer: (-(raw[layer] * scale
                                             - floors[layer]), layer))
    for layer in order[:remainder]:
        floors[layer] += 1
    return floors


def build(reference: Path, num_experts: int, manifest: str | None,
          mode: str = "observe", exclude: set[int] | None = None,
          coverage: dict[int, set[int]] | None = None,
          fill_fraction: float = 1.0,
          max_promotions: int | None = None,
          coverage_provenance: list[dict] | None = None) -> dict:
    """Two modes, because fq-policy/2 enforces occupancy == capacity.

    ``observe`` — zero K4 budget everywhere. A valid all-K3 policy that lets
    the loop START and record routing, proposing nothing. Convergence is then
    scored OFFLINE from the recorded routing against the reference bitmap.
    This is the only mode that is truthful on a checkpoint which physically
    contains no K4 weights: declaring occupied K4 slots that do not exist
    would make the model mis-decode.

    ``seeded`` — a real K4 budget, but only on layers that have K4 fragments
    and only filled with experts that have them. Everything else stays
    uniform K3 with a zero budget. The question becomes whether the loop
    MOVES from an arbitrary start toward the reference's choice, which is a
    stronger test than initialization.

    Two properties of ``seeded`` that decide whether the resulting run means
    anything, both reported in ``provenance``:

    * ``budget < len(pool)`` is required for the loop to be able to move at
      all. Swaps are 1-for-1 trades within a fixed-capacity K4 slab, so a
      layer whose budget saturates its fragment pool has zero legal
      promotion targets and will show zero swaps forever. ``fill_fraction``
      exists to leave that room.
    * a pool that is a SUBSET of the reference's own K4 set makes any
      overlap score on that layer circular — the loop can only pick experts
      the human already picked. Primed segments cut from the reference have
      exactly this property. Flagged as ``circular_layers``.
    """
    doc = json.loads(reference.read_text())
    sets = doc["per_layer_k4_sets"]
    layers = sorted(int(l) for l in sets)
    if exclude:
        layers = [l for l in layers if l not in exclude]

    coverage = coverage or {}
    if not 0.0 < fill_fraction <= 1.0:
        raise ValueError(f"fill_fraction must be in (0, 1], got {fill_fraction}")

    # Pass 1: what each layer could hold, before the global memory cap.
    pools: dict[int, list[int]] = {}
    wanted: dict[int, int] = {}
    for layer in layers:
        n_ref = int(sets[str(layer)]["n_k4"])
        if mode == "observe":
            wanted[layer] = 0
            pools[layer] = []
            continue
        pool = sorted(e for e in coverage.get(layer, ()) if e < num_experts)
        pools[layer] = pool
        room = min(n_ref, len(pool))
        wanted[layer] = int(room * fill_fraction) if room else 0

    budget = cap_budget(wanted, max_promotions)

    bits: dict[str, list[int]] = {}
    per_layer: list[dict] = []
    saturated: list[int] = []
    circular: list[int] = []
    for layer in layers:
        row = [K3] * num_experts
        n = budget[layer]
        seeded = pools[layer][:n]
        for e in seeded:
            row[e] = K4
        bits[str(layer)] = row
        if mode == "seeded":
            pool = pools[layer]
            ref_k4 = set(int(e) for e in sets[str(layer)]["k4_experts"])
            in_ref = len(set(pool) & ref_k4)
            if pool and n >= len(pool):
                saturated.append(layer)
            if pool and in_ref == len(pool):
                circular.append(layer)
            per_layer.append({
                "layer": layer,
                "n_k4_reference": int(sets[str(layer)]["n_k4"]),
                "fragment_pool": len(pool),
                "budget": n,
                # A promotion needs somewhere to land. `budget == 0` means the
                # layer is assembled uniform-K3 with NO K4 slab at all, so an
                # unoccupied fragment on disk is not a promotion candidate —
                # it is a fragment for a slab that does not exist. Reporting
                # `len(pool)` there told run-demo1.sh's tradability gate that
                # a permanently-frozen layer could swap, which is the exact
                # flat-line-that-looks-like-a-bug the gate exists to refuse.
                # Reachable whenever fill_fraction or the --max-extra-gib cap
                # rounds a layer's share down to zero (e.g. --fill-fraction
                # 0.01 on a 50-fragment layer).
                "promotion_candidates": max(len(pool) - n, 0) if n > 0 else 0,
                "pool_in_reference_k4": in_ref,
            })

    if manifest is None:
        # Deterministic id so two runs of the same experiment share a policy
        # identity and a third party can recompute it. In ``seeded`` mode the
        # MEMBERSHIP is load-bearing — the assembled checkpoint's K4 slabs
        # must physically match it — so it is hashed too; in ``observe`` mode
        # membership is a function of the budget and hashing the budget alone
        # keeps the id stable against earlier runs.
        payload = (json.dumps(budget, sort_keys=True) if mode == "observe"
                   else json.dumps({"budget": {str(k): v for k, v in
                                               budget.items()},
                                    "bits": bits}, sort_keys=True))
        manifest = hashlib.sha256(payload.encode()).hexdigest()

    total = sum(budget.values())
    provenance = {
        "proposed_by": f"scenario1/{mode}",
        "budget_source": doc.get("reference"),
        "note": ("Per-layer K4 cardinality derived from the human-built "
                 "3.42bpw Coder quant so the only free variable is WHICH "
                 "experts get promoted."),
        "promotions": total,
        "promotion_bytes_per_expert": PROMOTION_BYTES,
        "extra_bytes_model": total * PROMOTION_BYTES,
    }
    if mode == "seeded":
        provenance["note"] = (
            "Per-layer K4 cardinality derived from the human-built 3.42bpw "
            "Coder quant, then clamped to the K4 fragments that actually "
            "exist on disk. Layers with no K4 fragment stay uniform K3 with "
            "a zero budget; membership is seeded with the lowest-id experts "
            "IN THE FRAGMENT POOL, not the lowest ids overall, so every "
            "declared K4 slot is backed by a real fragment.")
        provenance["fill_fraction"] = fill_fraction
        provenance["max_promotions"] = max_promotions
        provenance["covered_layers"] = sorted(
            l for l in layers if pools.get(l))
        provenance["uncovered_layers"] = sorted(
            l for l in layers if not pools.get(l))
        provenance["saturated_layers"] = saturated
        provenance["circular_layers"] = circular
        provenance["per_layer_coverage"] = per_layer
        if coverage_provenance is not None:
            provenance["coverage_sources"] = coverage_provenance
        if circular:
            provenance["circularity_warning"] = (
                "On these layers every available K4 fragment is also in the "
                "reference's K4 set, so the loop can only promote experts "
                "the human already chose. Live promotion there demonstrates "
                "the MECHANISM; it is not evidence of agreement with the "
                "human and must not be scored as convergence.")

    return {
        "schema": "fq-policy/2",
        "manifest": manifest,
        "budget": {"mode": "fixed_cardinality",
                   "n_k4_per_layer": {str(k): v for k, v in budget.items()}},
        "bits_per_expert": bits,
        "pinned": {},
        "provenance": provenance,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reference", type=Path,
                    default=Path(__file__).parent / "reference-coder-quant.json")
    ap.add_argument("--num-experts", type=int, default=256)
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--mode", choices=["observe", "seeded"],
                    default="observe")
    # Layer 78 is GLM-5.2's MTP (multi-token prediction) layer. Its MoE is not
    # bound as a main-model MoERunner, so the collector never sees it and the
    # loop refuses to start on a policy that names it ("cannot map policy
    # layers ... onto collector layers"). It is also all-K3 in the reference
    # (n_k4 = 0), so excluding it costs the experiment nothing.
    ap.add_argument("--exclude-layer", type=int, action="append", default=[78],
                    help="repeatable; default excludes the MTP layer 78")
    ap.add_argument("--coverage-dir", type=Path, action="append", default=None,
                    help="repeatable local segment dir to discover K4 "
                         "fragments in; defaults to the dirs serve-glm52.sh "
                         "puts on the resolver's local path")
    ap.add_argument("--fill-fraction", type=float, default=1.0,
                    help="fraction of each layer's available K4 fragments to "
                         "occupy at boot. Below 1.0 leaves promotion "
                         "candidates the loop can actually trade into; at "
                         "1.0 a layer is saturated and can never swap")
    ap.add_argument("--max-extra-gib", type=float, default=None,
                    help="cap total promotions at this much PER-RANK memory "
                         "above uniform K3")
    ap.add_argument("--tp", type=int, default=4,
                    help="tensor-parallel degree used to convert "
                         "--max-extra-gib into a promotion count")
    ap.add_argument("--coverage-out", type=Path, default=None,
                    help="also write the discovered coverage map here")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    dirs = a.coverage_dir if a.coverage_dir else [Path(d) for d in
                                                  DEFAULT_COVERAGE_DIRS]
    coverage, cov_prov = discover_k_coverage(dirs, K4, a.num_experts)

    max_promotions = (promotions_for_gib(a.max_extra_gib, a.tp)
                      if a.max_extra_gib is not None else None)

    doc = build(a.reference, a.num_experts, a.manifest, a.mode,
                exclude=set(a.exclude_layer), coverage=coverage,
                fill_fraction=a.fill_fraction,
                max_promotions=max_promotions,
                coverage_provenance=cov_prov)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(doc, indent=1))

    if a.coverage_out:
        a.coverage_out.parent.mkdir(parents=True, exist_ok=True)
        a.coverage_out.write_text(json.dumps({
            "k": K4,
            "dirs": [str(d) for d in dirs],
            "sources": cov_prov,
            "layers": {str(l): sorted(v) for l, v in sorted(coverage.items())},
        }, indent=1))

    b = doc["budget"]["n_k4_per_layer"]
    prov = doc["provenance"]
    total = sum(b.values())
    print(f"mode          : {a.mode}")
    print(f"layers        : {len(b)}")
    print(f"K4 slots total: {total}")
    print(f"per-layer     : min={min(b.values())} max={max(b.values())}")
    print(f"excluded      : {sorted(set(a.exclude_layer))}")
    print(f"manifest      : {doc['manifest'][:16]}...")
    if a.mode == "seeded":
        cov_l = prov["covered_layers"]
        print(f"coverage dirs : {[str(d) for d in dirs]}")
        print(f"K4-covered    : {len(cov_l)} layers "
              f"{cov_l[:4]}{'...' if len(cov_l) > 4 else ''}")
        print(f"uncovered     : {len(prov['uncovered_layers'])} layers "
              f"(uniform K3, budget 0)")
        print(f"fill fraction : {a.fill_fraction}")
        if max_promotions is not None:
            print(f"memory cap    : {a.max_extra_gib} GiB/rank at tp={a.tp} "
                  f"-> {max_promotions} promotions")
        print(f"extra memory  : {total * PROMOTION_BYTES / a.tp / (1<<30):.3f}"
              f" GiB/rank above uniform K3")
        if prov["saturated_layers"]:
            print(f"WARNING       : {len(prov['saturated_layers'])} layer(s) "
                  f"saturate their fragment pool and CANNOT swap: "
                  f"{prov['saturated_layers']}")
        if prov["circular_layers"]:
            print(f"WARNING       : {len(prov['circular_layers'])} layer(s) "
                  f"have a fragment pool that is a subset of the reference's "
                  f"own K4 set — promotion there is circular, do not score "
                  f"it as convergence: {prov['circular_layers']}")
    else:
        print(f"start state   : all {a.num_experts} experts at K3 in every layer")
    print(f"wrote {a.out}")
    if a.coverage_out:
        print(f"wrote {a.coverage_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
