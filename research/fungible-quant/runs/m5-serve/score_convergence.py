#!/usr/bin/env python3
"""Score how well runtime-observed routing rediscovers a human's quant.

The claim under test: a human built `GLM-5.2-EXL3-TR3-3.42bpw` by choosing, per
layer, which experts deserve K4 instead of K3. If a fungible-quant loop watching
live routing promotes *the same* experts, then that human's offline allocation
work is reproducible at runtime — and shipping a separate full checkpoint per
bitrate is moving bytes nobody needed to move.

This scores that as a number, against two reference points that make the number
mean something:

* a **chance floor** — Jaccard under uniform random selection of the same
  cardinality. Any method beats zero; beating chance is the real bar.
* a **human ceiling** — the same overlap measured between the 3.42bpw Coder
  build and its 3.40bpw non-coder sibling, built by the same person from the
  same calibration data. Two competent humans agree ~0.66, so that is the
  honest target, not 1.0.

Deliberately NOT used as the ranking signal: `expert_rel_rt_mse`. That is the
*achieved* reconstruction error at the bitrate each expert was already shipped
at, so ranking by it would rediscover the reference from a field derived from
the reference — circular, and it would produce a spectacular fake result.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    u = len(a | b)
    return len(a & b) / u if u else 0.0


def load_reference(path: Path) -> dict:
    doc = json.loads(path.read_text())
    sets = doc["per_layer_k4_sets"]
    ref = {int(l): {"k4": set(int(e) for e in v["k4_experts"]),
                    "n_k4": int(v["n_k4"]), "n_k3": int(v["n_k3"])}
           for l, v in sets.items()}
    return {"ref": ref, "doc": doc}


def load_stats(path: Path, *, signal: str = "mass",
               last_only: bool = True) -> tuple[list[int], list[list[float]]]:
    """Read the loop's VLLM_FQ_DUMP_STATS jsonl -> (layer_ids, [L][E] scores).

    Later intervals see more traffic, so by default we score the LAST record
    rather than averaging across a warming window.
    """
    recs = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    if not recs:
        raise SystemExit(f"no records in {path}")
    use = [recs[-1]] if last_only else recs
    layers = [int(x) for x in use[0]["layers"]]
    # Check BEFORE indexing: the accumulator below reads use[0][signal][0],
    # so a missing field raised a bare KeyError and the careful message never
    # printed.
    for r in use:
        if signal not in r:
            raise SystemExit(
                f"{path} has no '{signal}' field — the collector aliases mass "
                f"to count when no topk_weights getter is bound; pass "
                f"--signal count to score that explicitly rather than "
                f"silently scoring a different quantity")
    acc: list[list[float]] = [[0.0] * len(use[0][signal][0])
                              for _ in range(len(layers))]
    for r in use:
        for i, row in enumerate(r[signal]):
            for j, v in enumerate(row):
                acc[i][j] += float(v)
    return layers, acc


def select_topk(scores: list[float], n: int) -> set:
    """Top-n expert ids by score; ties broken by lowest id for determinism."""
    order = sorted(range(len(scores)), key=lambda e: (-scores[e], e))
    return set(order[:n])


def chance_jaccard(n_a: int, n_b: int, total: int) -> float:
    """E[|A n B|] = n_a*n_b/N, evaluated at the expectation."""
    inter = n_a * n_b / total if total else 0.0
    union = n_a + n_b - inter
    return inter / union if union else 0.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reference", type=Path, required=True)
    ap.add_argument("--stats", type=Path, required=True,
                    help="jsonl written by VLLM_FQ_DUMP_STATS")
    ap.add_argument("--signal", default="mass", choices=["mass", "count"])
    ap.add_argument("--all-intervals", action="store_true",
                    help="accumulate every interval instead of the last")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    R = load_reference(args.reference)
    ref, doc = R["ref"], R["doc"]
    layers, scores = load_stats(args.stats, signal=args.signal,
                                last_only=not args.all_intervals)

    rng = random.Random(args.seed)
    rows, skipped = [], []
    for i, layer in enumerate(layers):
        if layer not in ref:
            skipped.append(layer)
            continue
        n_k4 = ref[layer]["n_k4"]
        total = len(scores[i])
        if n_k4 == 0 or n_k4 >= total:
            # layer 78 is all-K3 in the reference: nothing to select, and a
            # Jaccard of 1.0 there would inflate the mean for free
            skipped.append(layer)
            continue
        ours = select_topk(scores[i], n_k4)
        theirs = ref[layer]["k4"]
        rand = set(rng.sample(range(total), n_k4))
        rows.append({
            "layer": layer,
            "n_k4": n_k4,
            "jaccard": jaccard(ours, theirs),
            "jaccard_random_draw": jaccard(rand, theirs),
            "jaccard_chance_expected": chance_jaccard(n_k4, n_k4, total),
            "overlap": len(ours & theirs),
        })

    if not rows:
        raise SystemExit("no scorable layers — check that the stats dump's "
                         "layer ids match the reference's")

    mean = sum(r["jaccard"] for r in rows) / len(rows)
    mean_rand = sum(r["jaccard_random_draw"] for r in rows) / len(rows)
    mean_chance = sum(r["jaccard_chance_expected"] for r in rows) / len(rows)
    pooled_i = sum(r["overlap"] for r in rows)
    pooled_n = sum(r["n_k4"] for r in rows)
    pooled = pooled_i / (2 * pooled_n - pooled_i) if pooled_n else 0.0

    siblings = doc.get("sibling_human_builds_for_baseline", {})
    ceiling = max((v.get("mean_per_layer_jaccard", 0)
                   for v in siblings.values()), default=None)

    result = {
        "schema": "fq-convergence/1",
        "reference": doc.get("reference"),
        "signal": args.signal,
        "intervals": "all" if args.all_intervals else "last",
        "layers_scored": len(rows),
        "layers_skipped": skipped,
        "mean_per_layer_jaccard": round(mean, 6),
        "pooled_jaccard": round(pooled, 6),
        "chance_floor_expected": round(mean_chance, 6),
        "chance_floor_sampled": round(mean_rand, 6),
        "human_human_ceiling": ceiling,
        "lift_over_chance": (round(mean / mean_chance, 3)
                             if mean_chance else None),
        "fraction_of_human_agreement": (round(mean / ceiling, 3)
                                        if ceiling else None),
        "per_layer": rows,
    }

    print(f"layers scored      : {len(rows)} (skipped {len(skipped)})")
    print(f"mean Jaccard       : {mean:.4f}")
    print(f"pooled Jaccard     : {pooled:.4f}")
    print(f"chance floor       : {mean_chance:.4f} (sampled {mean_rand:.4f})")
    if ceiling:
        print(f"human-human ceiling: {ceiling:.4f}")
        print(f"-> {mean/mean_chance:.2f}x chance, "
              f"{100*mean/ceiling:.0f}% of human-human agreement")
    if args.out:
        args.out.write_text(json.dumps(result, indent=1))
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
