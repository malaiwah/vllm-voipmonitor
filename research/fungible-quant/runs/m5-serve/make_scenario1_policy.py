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


def build(reference: Path, num_experts: int, manifest: str | None,
          mode: str = "observe") -> dict:
    """Two modes, because fq-policy/2 enforces occupancy == capacity.

    ``observe`` — zero K4 budget everywhere. A valid all-K3 policy that lets
    the loop START and record routing, proposing nothing. Convergence is then
    scored OFFLINE from the recorded routing against the reference bitmap.
    This is the only mode that is truthful on a checkpoint which physically
    contains no K4 weights: declaring occupied K4 slots that do not exist
    would make the model mis-decode.

    ``seeded`` — the reference's per-layer K4 cardinality, filled with an
    arbitrary initial membership (lowest expert ids). Requires a checkpoint
    that actually carries K4 weights. Here the question becomes whether the
    loop MOVES from an arbitrary start toward the reference's choice, which
    is a stronger test than initialization — but it cannot run until K4
    coverage exists for the served layers.
    """
    doc = json.loads(reference.read_text())
    sets = doc["per_layer_k4_sets"]
    layers = sorted(int(l) for l in sets)

    bits, budget = {}, {}
    for layer in layers:
        n_ref = int(sets[str(layer)]["n_k4"])
        if mode == "observe":
            bits[str(layer)] = [K3] * num_experts
            budget[str(layer)] = 0
        else:
            row = [K3] * num_experts
            for e in range(min(n_ref, num_experts)):
                row[e] = K4          # arbitrary seed: lowest ids
            bits[str(layer)] = row
            budget[str(layer)] = n_ref

    if manifest is None:
        # Deterministic id over the budget shape, so two runs of the same
        # experiment share a policy identity and a third-party can recompute it
        manifest = hashlib.sha256(
            json.dumps(budget, sort_keys=True).encode()).hexdigest()

    return {
        "schema": "fq-policy/2",
        "manifest": manifest,
        "budget": {"mode": "fixed_cardinality", "n_k4_per_layer": budget},
        "bits_per_expert": bits,
        "pinned": {},
        "provenance": {
            "proposed_by": "scenario1/all-k3-start",
            "budget_source": doc.get("reference"),
            "note": ("Per-layer K4 cardinality copied from the human-built "
                     "3.42bpw Coder quant so the only free variable is WHICH "
                     "experts get promoted. Membership deliberately starts "
                     "uniform-K3; it is not seeded from the reference."),
        },
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
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    doc = build(a.reference, a.num_experts, a.manifest, a.mode)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(doc, indent=1))

    b = doc["budget"]["n_k4_per_layer"]
    total = sum(b.values())
    print(f"layers        : {len(b)}")
    print(f"K4 slots total: {total}")
    print(f"per-layer     : min={min(b.values())} max={max(b.values())}")
    print(f"mode          : {a.mode}")
    print(f"start state   : all {a.num_experts} experts at K3 in every layer")
    print(f"manifest      : {doc['manifest'][:16]}...")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
