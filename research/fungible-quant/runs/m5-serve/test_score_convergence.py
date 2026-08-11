"""Known-answer tests for the convergence scorer.

A metric that reports a flattering number for a signal that carries no
information is worse than no metric, so the cases that matter here are the
ones with an answer known in advance: a perfect ranking must score 1.0, a
random one must land on the chance floor, and an inverted one must score 0.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import score_convergence as sc  # noqa: E402

E = 256
REF_FILE = Path(__file__).parent / "reference-coder-quant.json"


def synth_reference(tmp_path, layers=(3, 4, 5), n_k4=108):
    per_layer, rng = {}, random.Random(0)
    for l in layers:
        k4 = sorted(rng.sample(range(E), n_k4))
        per_layer[str(l)] = {"n_k3": E - n_k4, "n_k4": n_k4,
                             "k4_experts": k4}
    doc = {"reference": {"repo": "synthetic"},
           "per_layer_k4_sets": per_layer,
           "sibling_human_builds_for_baseline": {
               "sib": {"mean_per_layer_jaccard": 0.65}}}
    p = tmp_path / "ref.json"
    p.write_text(json.dumps(doc))
    return p, doc


def write_stats(tmp_path, doc, kind, name="s.jsonl", signal="mass"):
    layers = sorted(int(l) for l in doc["per_layer_k4_sets"])
    rows = []
    for l in layers:
        k4 = set(doc["per_layer_k4_sets"][str(l)]["k4_experts"])
        if kind == "perfect":
            row = [1.0 if e in k4 else 0.0 for e in range(E)]
        elif kind == "inverted":
            row = [0.0 if e in k4 else 1.0 for e in range(E)]
        else:
            rng = random.Random(l)
            row = [rng.random() for _ in range(E)]
        rows.append(row)
    rec = {"step": 1, "interval": 1, "layers": layers,
           "tier_of": [[3] * E for _ in layers], signal: rows}
    p = tmp_path / name
    p.write_text(json.dumps(rec) + "\n")
    return p


def score(tmp_path, kind, **kw):
    ref_p, doc = synth_reference(tmp_path)
    stats_p = write_stats(tmp_path, doc, kind, **kw)
    layers, scores = sc.load_stats(stats_p, signal=kw.get("signal", "mass"))
    R = sc.load_reference(ref_p)
    out = []
    for i, l in enumerate(layers):
        n = R["ref"][l]["n_k4"]
        out.append(sc.jaccard(sc.select_topk(scores[i], n), R["ref"][l]["k4"]))
    return sum(out) / len(out)


def test_perfect_ranking_scores_one(tmp_path):
    assert score(tmp_path, "perfect") == pytest.approx(1.0)


def test_inverted_ranking_scores_zero(tmp_path):
    assert score(tmp_path, "inverted") == pytest.approx(0.0)


def test_random_ranking_lands_on_the_chance_floor(tmp_path):
    got = score(tmp_path, "random")
    expected = sc.chance_jaccard(108, 108, E)
    assert got == pytest.approx(expected, abs=0.06), (got, expected)


def test_chance_jaccard_matches_the_published_reference_value():
    """The reference file states 0.267327 for 108-of-256; our analytic
    formula must agree, or one of the two is wrong."""
    assert sc.chance_jaccard(108, 108, 256) == pytest.approx(0.267327,
                                                             abs=1e-5)
    assert sc.chance_jaccard(50, 50, 256) == pytest.approx(0.108225, abs=1e-5)


def test_topk_is_deterministic_under_ties():
    """All-equal scores must not make the result depend on dict ordering."""
    flat = [1.0] * E
    assert sc.select_topk(flat, 5) == {0, 1, 2, 3, 4}
    assert sc.select_topk(flat, 5) == sc.select_topk(flat, 5)


def test_missing_signal_fails_loudly(tmp_path):
    """The collector aliases mass to count when no weights getter is bound.
    Silently scoring a different quantity would be a wrong headline number."""
    ref_p, doc = synth_reference(tmp_path)
    p = write_stats(tmp_path, doc, "perfect", signal="count")
    with pytest.raises(SystemExit, match="no 'mass' field"):
        sc.load_stats(p, signal="mass")


def test_empty_stats_file_fails_loudly(tmp_path):
    p = tmp_path / "empty.jsonl"
    p.write_text("")
    with pytest.raises(SystemExit, match="no records"):
        sc.load_stats(p)


@pytest.mark.skipif(not REF_FILE.exists(), reason="reference not present")
def test_against_the_real_reference_file(tmp_path):
    """The real artifact must load and be internally consistent."""
    R = sc.load_reference(REF_FILE)
    ref = R["ref"]
    assert len(ref) == 76, len(ref)
    total_k4 = sum(v["n_k4"] for v in ref.values())
    assert total_k4 == 8042, total_k4          # stated in the report
    for layer, v in ref.items():
        assert len(v["k4"]) == v["n_k4"], layer
        assert v["n_k3"] + v["n_k4"] == 256, layer


@pytest.mark.skipif(not REF_FILE.exists(), reason="reference not present")
def test_all_k3_layer_is_skipped_not_scored_as_perfect(tmp_path):
    """Layer 78 has zero K4 experts. Scoring it would yield a free 1.0 on an
    empty-vs-empty comparison and inflate the mean."""
    R = sc.load_reference(REF_FILE)
    zero = [l for l, v in R["ref"].items() if v["n_k4"] == 0]
    assert zero, "expected at least one all-K3 layer in the reference"
