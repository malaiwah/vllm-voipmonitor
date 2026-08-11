"""Tests for the scenario-1 boot policy builder.

The property that matters here is not "does it produce JSON" — it is that
every K4 slot the policy declares is backed by a K4 fragment that actually
exists. A policy that names an expert with no fragment does not fail loudly:
`VLLM_FQ_K_FALLBACK=3` serves the K3 bytes instead, so the model keeps
answering while the occupancy table reports a promotion that never happened.
That is the failure these tests exist to make impossible.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import make_scenario1_policy as mk  # noqa: E402

E = 256
REF_FILE = Path(__file__).parent / "reference-coder-quant.json"


# ------------------------------------------------------------------ fixtures

def write_reference(tmp_path, per_layer):
    """per_layer: {layer: [expert ids at K4]}"""
    doc = {"reference": {"repo_id": "synthetic"},
           "per_layer_k4_sets": {
               str(l): {"n_k4": len(k4), "n_k3": E - len(k4),
                        "k4_experts": sorted(k4)}
               for l, k4 in per_layer.items()}}
    p = tmp_path / "ref.json"
    p.write_text(json.dumps(doc))
    return p


def write_segment_dir(tmp_path, name, per_layer, k=4, with_segment=True,
                      with_index=True):
    """A minimal local segment dir shaped exactly like the resolver expects."""
    base = tmp_path / name
    base.mkdir(parents=True, exist_ok=True)
    index = {}
    for layer, experts in per_layer.items():
        index[str(layer)] = {
            "file": f"layer-{layer:03d}.k{k}.safetensors",
            "sha256": "0" * 64,
            "body_offset": 0,
            "experts": {str(e): [0, 16] for e in sorted(experts)},
        }
        if with_segment:
            (base / f"layer-{layer:03d}.k{k}.safetensors").write_bytes(b"\0" * 8)
    if with_index:
        (base / f"index-k{k}.json").write_text(json.dumps(index))
    return base


# ------------------------------------------------------- coverage discovery

def test_coverage_is_per_expert_not_per_layer(tmp_path):
    """A primed segment carries only the human's chosen experts. Reporting
    the whole layer as covered is what produces unbacked K4 slots."""
    d = write_segment_dir(tmp_path, "segs", {4: [7, 9, 200]})
    cov, _ = mk.discover_k_coverage([d], num_experts=E)
    assert cov == {4: {7, 9, 200}}


def test_coverage_unions_across_dirs(tmp_path):
    a = write_segment_dir(tmp_path, "a", {4: [1, 2]})
    b = write_segment_dir(tmp_path, "b", {4: [2, 3], 5: [10]})
    cov, _ = mk.discover_k_coverage([a, b], num_experts=E)
    assert cov == {4: {1, 2, 3}, 5: {10}}


def test_index_without_segment_file_is_not_coverage(tmp_path):
    """The resolver requires index AND payload; an index alone is a miss."""
    d = write_segment_dir(tmp_path, "segs", {4: [1, 2]}, with_segment=False)
    cov, _ = mk.discover_k_coverage([d], num_experts=E)
    assert cov == {}


def test_segment_file_without_index_is_not_coverage(tmp_path):
    d = write_segment_dir(tmp_path, "segs", {4: [1, 2]}, with_index=False)
    cov, _ = mk.discover_k_coverage([d], num_experts=E)
    assert cov == {}


def test_missing_and_broken_dirs_are_recorded_not_raised(tmp_path):
    bad = tmp_path / "broken"
    bad.mkdir()
    (bad / "index-k4.json").write_text("{ not json")
    (bad / "layer-004.k4.safetensors").write_bytes(b"\0")
    cov, prov = mk.discover_k_coverage([tmp_path / "nope", bad], num_experts=E)
    assert cov == {}
    assert [p["skipped"] for p in prov] == [
        "no index-k4.json", "unreadable index: JSONDecodeError"]


def test_experts_outside_the_model_are_dropped(tmp_path):
    d = write_segment_dir(tmp_path, "segs", {4: [1, 999]})
    cov, _ = mk.discover_k_coverage([d], num_experts=E)
    assert cov == {4: {1}}


# ------------------------------------------------------------- seeded build

def test_uncovered_layers_stay_uniform_k3_with_zero_budget(tmp_path):
    ref = write_reference(tmp_path, {3: list(range(50)),
                                     4: list(range(108)),
                                     5: list(range(108))})
    d = write_segment_dir(tmp_path, "segs", {4: list(range(10, 60))})
    cov, _ = mk.discover_k_coverage([d], num_experts=E)
    doc = mk.build(ref, E, "m", "seeded", coverage=cov)
    b = doc["budget"]["n_k4_per_layer"]
    assert b["3"] == 0 and b["5"] == 0
    assert set(doc["bits_per_expert"]["3"]) == {3}
    assert set(doc["bits_per_expert"]["5"]) == {3}
    assert doc["provenance"]["uncovered_layers"] == [3, 5]
    assert doc["provenance"]["covered_layers"] == [4]


def test_every_declared_k4_slot_has_a_fragment(tmp_path):
    """The headline invariant. Seeding by lowest id overall would put K4 on
    experts 0..49, none of which are in this pool."""
    pool = list(range(200, 256))
    ref = write_reference(tmp_path, {4: list(range(108))})
    d = write_segment_dir(tmp_path, "segs", {4: pool})
    cov, _ = mk.discover_k_coverage([d], num_experts=E)
    doc = mk.build(ref, E, "m", "seeded", coverage=cov)
    named = {e for e, b in enumerate(doc["bits_per_expert"]["4"]) if b == 4}
    assert named, "expected some promotions"
    assert named <= set(pool)


def test_budget_never_exceeds_the_fragment_pool(tmp_path):
    ref = write_reference(tmp_path, {4: list(range(108))})
    d = write_segment_dir(tmp_path, "segs", {4: list(range(20))})
    cov, _ = mk.discover_k_coverage([d], num_experts=E)
    doc = mk.build(ref, E, "m", "seeded", coverage=cov)
    assert doc["budget"]["n_k4_per_layer"]["4"] == 20


def test_budget_never_exceeds_the_reference_cardinality(tmp_path):
    """A pool bigger than the human's budget must not inflate the envelope —
    that would turn a selection test into a budget test."""
    ref = write_reference(tmp_path, {4: list(range(108))})
    d = write_segment_dir(tmp_path, "segs", {4: list(range(E))})
    cov, _ = mk.discover_k_coverage([d], num_experts=E)
    doc = mk.build(ref, E, "m", "seeded", coverage=cov)
    assert doc["budget"]["n_k4_per_layer"]["4"] == 108


def test_occupancy_equals_capacity_on_every_layer(tmp_path):
    """fq-policy/2's store.validate_policy hard-refuses any other state."""
    ref = write_reference(tmp_path, {3: list(range(50)),
                                     4: list(range(108)),
                                     5: list(range(108))})
    d = write_segment_dir(tmp_path, "segs",
                          {3: list(range(30)), 4: list(range(200, 256))})
    cov, _ = mk.discover_k_coverage([d], num_experts=E)
    doc = mk.build(ref, E, "m", "seeded", coverage=cov, fill_fraction=0.5)
    for layer, bits in doc["bits_per_expert"].items():
        assert sum(b == 4 for b in bits) == \
            doc["budget"]["n_k4_per_layer"][layer], layer
        assert len(bits) == E


def test_fill_fraction_leaves_promotion_candidates(tmp_path):
    """A saturated pool has zero legal swap targets, so the loop can never
    move; the demo would show 'no swaps' and look like a bug."""
    ref = write_reference(tmp_path, {4: list(range(108))})
    d = write_segment_dir(tmp_path, "segs", {4: list(range(100))})
    cov, _ = mk.discover_k_coverage([d], num_experts=E)

    full = mk.build(ref, E, "m", "seeded", coverage=cov, fill_fraction=1.0)
    assert full["budget"]["n_k4_per_layer"]["4"] == 100
    assert full["provenance"]["saturated_layers"] == [4]

    half = mk.build(ref, E, "m", "seeded", coverage=cov, fill_fraction=0.5)
    assert half["budget"]["n_k4_per_layer"]["4"] == 50
    assert half["provenance"]["saturated_layers"] == []
    assert half["provenance"]["per_layer_coverage"][0][
        "promotion_candidates"] == 50


def test_a_zero_budget_layer_reports_no_promotion_candidates(tmp_path):
    """budget == 0 means NO K4 slab, so the pool is not a set of candidates.

    A promotion is a 1-for-1 trade inside a pre-allocated fixed-capacity K4
    slab. A layer whose budget rounds down to zero is assembled uniform K3:
    capacity 0, no slab, nothing to promote into, forever. Reporting
    `len(pool)` candidates there told run-demo1.sh's tradability gate the
    layer could swap, and told SCOPE.md that live promotion was "physically
    possible" on it — both false, and both in the artifacts written
    specifically to keep the claim honest.
    """
    ref = write_reference(tmp_path, {3: list(range(50)),
                                     4: list(range(108))})
    d = write_segment_dir(tmp_path, "segs", {3: list(range(50)),
                                             4: list(range(108))})
    cov, _ = mk.discover_k_coverage([d], num_experts=E)

    # 1% of a 50-fragment pool floors to 0 slots; 1% of 108 floors to 1.
    doc = mk.build(ref, E, "m", "seeded", coverage=cov, fill_fraction=0.01)
    rows = {r["layer"]: r for r in doc["provenance"]["per_layer_coverage"]}
    assert doc["budget"]["n_k4_per_layer"]["3"] == 0
    assert rows[3]["fragment_pool"] == 50
    assert rows[3]["promotion_candidates"] == 0, (
        "a layer with no K4 slab was reported as promotable")
    # the layer that DID get a slot keeps its real candidate count
    assert doc["budget"]["n_k4_per_layer"]["4"] == 1
    assert rows[4]["promotion_candidates"] == 107


def test_a_memory_cap_that_zeroes_a_layer_also_zeroes_its_candidates(tmp_path):
    """Same hole, reached through --max-extra-gib instead of --fill-fraction.

    cap_budget scales proportionally, so a small layer's share can floor to
    zero while larger layers keep slots.
    """
    ref = write_reference(tmp_path, {3: list(range(50)),
                                     4: list(range(108))})
    d = write_segment_dir(tmp_path, "segs", {3: list(range(50)),
                                             4: list(range(108))})
    cov, _ = mk.discover_k_coverage([d], num_experts=E)
    doc = mk.build(ref, E, "m", "seeded", coverage=cov, fill_fraction=1.0,
                   max_promotions=1)
    budget = doc["budget"]["n_k4_per_layer"]
    rows = {r["layer"]: r for r in doc["provenance"]["per_layer_coverage"]}
    zeroed = [l for l in (3, 4) if budget[str(l)] == 0]
    assert zeroed, "expected the cap to starve at least one layer"
    for layer in zeroed:
        assert rows[layer]["fragment_pool"] > 0
        assert rows[layer]["promotion_candidates"] == 0, (
            f"L{layer} has budget 0 but claims promotion candidates")


def test_bad_fill_fraction_is_rejected(tmp_path):
    ref = write_reference(tmp_path, {4: list(range(108))})
    for bad in (0.0, -0.5, 1.5):
        with pytest.raises(ValueError, match="fill_fraction"):
            mk.build(ref, E, "m", "seeded", coverage={}, fill_fraction=bad)


def test_pool_drawn_from_the_reference_is_flagged_as_circular(tmp_path):
    """Primed segments cut from the reference contain only the reference's
    own K4 experts. Any overlap score there is 1.0 by construction."""
    ref_k4 = list(range(100, 208))
    ref = write_reference(tmp_path, {4: ref_k4})
    d = write_segment_dir(tmp_path, "segs", {4: ref_k4})
    cov, _ = mk.discover_k_coverage([d], num_experts=E)
    doc = mk.build(ref, E, "m", "seeded", coverage=cov, fill_fraction=0.5)
    assert doc["provenance"]["circular_layers"] == [4]
    assert "circularity_warning" in doc["provenance"]


def test_pool_with_off_reference_experts_is_not_flagged(tmp_path):
    ref = write_reference(tmp_path, {4: list(range(100, 208))})
    d = write_segment_dir(tmp_path, "segs", {4: list(range(90, 190))})
    cov, _ = mk.discover_k_coverage([d], num_experts=E)
    doc = mk.build(ref, E, "m", "seeded", coverage=cov, fill_fraction=0.5)
    assert doc["provenance"]["circular_layers"] == []
    assert "circularity_warning" not in doc["provenance"]


# ------------------------------------------------------------- memory cap

def test_promotions_for_gib_matches_the_published_envelope():
    """convergence-demo-plan.md: 8.835 GiB/rank of headroom buys exactly the
    reference's 8,042 promotions at tp=4. If this drifts, the envelope
    arithmetic and the policy disagree."""
    assert mk.PROMOTION_BYTES == 4_718_592
    assert mk.promotions_for_gib(8042 * 1_179_648 / (1 << 30), 4) == 8042


def test_memory_cap_scales_every_layer_not_just_the_tail(tmp_path):
    ref = write_reference(tmp_path, {3: list(range(50)),
                                     4: list(range(108)),
                                     5: list(range(108))})
    d = write_segment_dir(tmp_path, "segs",
                          {3: list(range(50)), 4: list(range(108)),
                           5: list(range(108))})
    cov, _ = mk.discover_k_coverage([d], num_experts=E)
    doc = mk.build(ref, E, "m", "seeded", coverage=cov, max_promotions=133)
    b = doc["budget"]["n_k4_per_layer"]
    assert sum(b.values()) == 133
    assert all(v > 0 for v in b.values()), b   # no layer starved to zero


def test_cap_budget_is_a_noop_below_the_limit():
    raw = {3: 5, 4: 10}
    assert mk.cap_budget(raw, 100) == raw
    assert mk.cap_budget(raw, None) == raw


def test_cap_budget_zero_empties_every_layer():
    assert mk.cap_budget({3: 5, 4: 10}, 0) == {3: 0, 4: 0}


def test_cap_budget_is_deterministic():
    raw = {l: 7 for l in range(3, 20)}
    first = mk.cap_budget(raw, 50)
    assert first == mk.cap_budget(raw, 50)
    assert sum(first.values()) == 50


# ----------------------------------------------------------- observe mode

def test_observe_mode_is_unchanged_by_coverage(tmp_path):
    """Observe mode must stay a pure all-K3 zero-budget policy even when K4
    fragments happen to be lying around, or the 'truthful on a checkpoint
    with no K4 weights' guarantee in its docstring is void."""
    ref = write_reference(tmp_path, {3: list(range(50)), 4: list(range(108))})
    d = write_segment_dir(tmp_path, "segs", {4: list(range(108))})
    cov, _ = mk.discover_k_coverage([d], num_experts=E)
    with_cov = mk.build(ref, E, None, "observe", coverage=cov)
    without = mk.build(ref, E, None, "observe", coverage={})
    assert with_cov == without
    assert set(sum(with_cov["bits_per_expert"].values(), [])) == {3}
    assert sum(with_cov["budget"]["n_k4_per_layer"].values()) == 0


def test_seeded_manifest_tracks_membership_observe_tracks_budget(tmp_path):
    """Two seeded policies with the same cardinality but different members
    are different physical checkpoints; sharing a manifest would make the
    policy store rehydrate the wrong one."""
    ref = write_reference(tmp_path, {4: list(range(108))})
    a = write_segment_dir(tmp_path, "a", {4: list(range(50))})
    b = write_segment_dir(tmp_path, "b", {4: list(range(100, 150))})
    cov_a, _ = mk.discover_k_coverage([a], num_experts=E)
    cov_b, _ = mk.discover_k_coverage([b], num_experts=E)
    doc_a = mk.build(ref, E, None, "seeded", coverage=cov_a)
    doc_b = mk.build(ref, E, None, "seeded", coverage=cov_b)
    assert doc_a["budget"] == doc_b["budget"]
    assert doc_a["manifest"] != doc_b["manifest"]


def test_excluded_layer_never_appears(tmp_path):
    ref = write_reference(tmp_path, {4: list(range(108)), 78: []})
    doc = mk.build(ref, E, "m", "observe", exclude={78})
    assert "78" not in doc["bits_per_expert"]
    assert "78" not in doc["budget"]["n_k4_per_layer"]


# --------------------------------------------------- against the real files

@pytest.mark.skipif(not REF_FILE.exists(), reason="reference not present")
def test_real_reference_seeded_with_no_coverage_is_all_k3():
    """The honest degenerate case: no K4 fragments anywhere means seeded
    mode must produce something identical in effect to observe mode rather
    than a policy the loader cannot satisfy."""
    doc = mk.build(REF_FILE, E, "m", "seeded", exclude={78}, coverage={})
    assert sum(doc["budget"]["n_k4_per_layer"].values()) == 0
    assert set(sum(doc["bits_per_expert"].values(), [])) == {3}
    assert len(doc["provenance"]["uncovered_layers"]) == 75
