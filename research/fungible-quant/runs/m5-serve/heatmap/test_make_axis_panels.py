"""Tests for the 4-panel axis figure.

The four things that can silently make this figure lie, and are therefore
tested with answers known in advance:

1. **Panel arithmetic** — share, log2-relative-to-uniform, clipping, dead
   cells. A uniform layer must land dead centre; a layer where one expert
   takes everything must clip at the top and floor the rest.
2. **The shared scale** — the same value must produce the same colour band in
   every panel regardless of that panel's volume, and the bounds must be
   printed. A per-panel scale would make the whole comparison meaningless
   while still looking beautiful.
3. **The overlap matrix** — identical traffic must read 1.0 and disjoint top
   sets must read 0.0, at the reference's own per-layer cardinality.
4. **The synthetic watermark** — it must be impossible to emit a placeholder
   figure that does not say so, including by blanking the constant.
"""
from __future__ import annotations

import json
import random
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_axis_panels as M  # noqa: E402

E = 256
LAYERS = [3, 4, 5, 6]


# ------------------------------------------------------------- fixtures

def synth_reference(tmp_path: Path, layers=LAYERS, n_k4=50) -> Path:
    per_layer, rng = {}, random.Random(7)
    for l in layers:
        per_layer[str(l)] = {"n_k3": E - n_k4, "n_k4": n_k4,
                             "k4_experts": sorted(rng.sample(range(E), n_k4))}
    doc = {"reference": {"repo": "synthetic/ref"},
           "per_layer_k4_sets": per_layer,
           "sibling_human_builds_for_baseline": {
               "sib": {"mean_per_layer_jaccard": 0.65}}}
    p = tmp_path / "ref.json"
    p.write_text(json.dumps(doc))
    return p


def write_dump(tmp_path: Path, name: str, rows: list[list[float]],
               *, layers=LAYERS, mass_is_real=None, records=1) -> Path:
    p = tmp_path / name
    lines = []
    for i in range(records):
        rec = {"step": 100 * (i + 1), "interval": i + 1, "layers": layers,
               "tier_of": [[3] * E for _ in layers],
               "count": rows, "mass": rows}
        if mass_is_real is not None:
            rec["mass_is_real"] = mass_is_real
        lines.append(json.dumps(rec))
    p.write_text("\n".join(lines) + "\n")
    return p


def hot_rows(hot: range, hot_w=10.0, cold_w=1.0) -> list[list[float]]:
    """One matrix whose hottest experts are exactly ``hot``, every layer."""
    row = [hot_w if e in hot else cold_w for e in range(E)]
    return [list(row) for _ in LAYERS]


def panel_from(tmp_path, name, rows, scale=1.0, **kw) -> M.Panel:
    scaled = [[v * scale for v in r] for r in rows]
    return M.load_axis(name, write_dump(tmp_path, f"{name}.jsonl", scaled, **kw),
                       signal="count", all_intervals=False)


# ------------------------------------------------------- 1. panel arithmetic

def test_uniform_layer_is_exactly_one_times_uniform():
    rel = M.rel_uniform([4.0] * E)
    assert rel == [pytest.approx(1.0)] * E
    assert M.to_scale(rel[0]) == pytest.approx(0.0)
    # ... and 1x must sit in the middle band, not at a band edge, so "about
    # its fair share" is one quiet colour rather than a boundary the eye
    # reads as a threshold.
    assert M.band_of(0.0) == M.N_BANDS // 2


def test_share_is_volume_invariant():
    row = [random.Random(3).random() for _ in range(E)]
    a = M.rel_uniform(row)
    b = M.rel_uniform([v * 1e6 for v in row])
    assert a == pytest.approx(b)


def test_one_expert_takes_everything():
    row = [0.0] * E
    row[17] = 5.0
    rel = M.rel_uniform(row)
    assert rel[17] == pytest.approx(E)                # 256x uniform
    assert M.to_scale(rel[17]) == pytest.approx(M.DOMAIN[1])   # clipped
    assert M.cell_flags(rel[17]) == "hi"
    assert M.cell_flags(rel[0]) == "dead"             # never routed != cold
    assert M.to_scale(rel[0]) == pytest.approx(M.DOMAIN[0])
    assert M.band_of(M.to_scale(rel[17])) == M.N_BANDS - 1
    assert M.band_of(M.to_scale(rel[0])) == 0


def test_dead_layer_does_not_divide_by_zero():
    assert M.rel_uniform([0.0] * E) == [0.0] * E
    assert M.layer_entropy([0.0] * E) == 0.0


def test_entropy_bounds():
    assert M.layer_entropy([1.0] * E) == pytest.approx(1.0)
    spike = [0.0] * E
    spike[0] = 1.0
    assert M.layer_entropy(spike) == pytest.approx(0.0)


def test_band_edges_span_the_domain_monotonically():
    edges = M.band_edges()
    assert edges[0] == M.DOMAIN[0] and edges[-1] == M.DOMAIN[1]
    assert edges == sorted(edges)
    assert len(edges) == M.N_BANDS + 1
    vals = [(edges[i] + edges[i + 1]) / 2 for i in range(M.N_BANDS)]
    assert [M.band_of(v) for v in vals] == list(range(M.N_BANDS))


def test_runs_reproduce_the_row_exactly():
    row = [0, 0, 3, 3, 3, 1, 0]
    runs = M._runs(row)
    assert runs == [(0, 2, 0), (2, 3, 3), (5, 1, 1), (6, 1, 0)]
    rebuilt = []
    for start, length, b in runs:
        assert start == len(rebuilt)
        rebuilt += [b] * length
    assert rebuilt == row


def test_panel_census_counts_clipped_and_dead(tmp_path):
    rows = []
    for _ in LAYERS:
        row = [1.0] * E
        row[0] = 0.0            # never routed
        row[1] = 1e6            # way above 16x uniform
        rows.append(row)
    p = panel_from(tmp_path, "census", rows)
    assert p.dead == len(LAYERS)
    assert p.clipped_hi == len(LAYERS)
    # one expert holding ~all the mass pushes the other 254 below 1/16x, and
    # that clipping is reported rather than quietly rendered as "coldest".
    assert p.clipped_lo == 254 * len(LAYERS)
    assert p.total == pytest.approx(sum(sum(r) for r in rows))
    assert p.rowsum_dev == pytest.approx(0.0)


# --------------------------------------------------------- 2. shared scale

def test_same_value_gets_the_same_band_in_every_panel(tmp_path):
    """A panel seeing 1000x the traffic must not get its own scale."""
    rows = hot_rows(range(16))
    small = panel_from(tmp_path, "small", rows, scale=1.0)
    huge = panel_from(tmp_path, "huge", rows, scale=1000.0)
    assert huge.total > 100 * small.total
    a = [M.band_of(v) for v in small.scale_rows()[0]]
    b = [M.band_of(v) for v in huge.scale_rows()[0]]
    assert a == b


def test_domain_is_a_constant_not_derived_from_data(tmp_path):
    before = M.DOMAIN
    flat = panel_from(tmp_path, "flat", [[1.0] * E for _ in LAYERS])
    spiky = panel_from(tmp_path, "spiky", hot_rows(range(4), 900.0, 0.01))
    M.build_permutation([flat, spiky])
    assert M.DOMAIN == before == (-4.0, 4.0)


def test_figure_prints_the_scale_bounds(tmp_path):
    fig = _figure(tmp_path, ["a", "b"], synthetic=False)
    svg = M.svg_bytes(fig)
    assert "never auto-scaled" in svg
    assert "[-4.000, +4.000]" in svg
    assert "0.0625" in svg and "16" in svg
    assert "log2( share" in svg


def test_all_panels_are_drawn_with_one_permutation(tmp_path):
    fig = _figure(tmp_path, ["a", "b", "c"], synthetic=False)
    M.svg_bytes(fig)
    hashes = {p.perm_hash for p in fig.panels}
    assert hashes == {fig.perm_hash}


def test_a_panel_carrying_a_foreign_permutation_is_refused(tmp_path):
    fig = _figure(tmp_path, ["a", "b"], synthetic=False)
    fig.panels[1].perm_hash = "deadbeefcafe"
    with pytest.raises(RuntimeError, match="permutation"):
        M.svg_bytes(fig)


def test_permutation_is_shared_and_not_per_panel(tmp_path):
    """Per-panel sorting would make every panel a gradient by construction."""
    a = panel_from(tmp_path, "a", hot_rows(range(0, 20)))
    b = panel_from(tmp_path, "b", hot_rows(range(200, 220)))
    perm, _h = M.build_permutation([a, b])
    assert sorted(perm[0]) == list(range(E))          # a real permutation

    def nonincreasing(v):
        return all(x >= y - 1e-12 for x, y in zip(v, v[1:]))

    along_a = [a.rel[0][e] for e in perm[0]]
    along_b = [b.rel[0][e] for e in perm[0]]
    # Under per-panel sorting BOTH would descend monotonically and the
    # figure would show two identical gradients over disjoint expert sets.
    assert nonincreasing(along_a)
    assert not nonincreasing(along_b)


def test_native_order_is_identity(tmp_path):
    a = panel_from(tmp_path, "a", hot_rows(range(0, 20)))
    perm, _h = M.build_permutation([a], "native")
    assert perm[0] == list(range(E))


def test_order_by_named_axis(tmp_path):
    a = panel_from(tmp_path, "a", hot_rows(range(0, 8)))
    b = panel_from(tmp_path, "b", hot_rows(range(100, 108)))
    perm, _h = M.build_permutation([a, b], "axis:b")
    assert set(perm[0][:8]) == set(range(100, 108))
    with pytest.raises(SystemExit):
        M.build_permutation([a, b], "axis:nope")


# ------------------------------------------------------- 3. overlap matrix

def test_identical_dumps_overlap_exactly_one(tmp_path):
    ref = M.__dict__["sc"].load_reference(synth_reference(tmp_path))["ref"]
    rows = [[random.Random(l).random() for _ in range(E)] for l in LAYERS]
    a = panel_from(tmp_path, "a", rows)
    b = panel_from(tmp_path, "b", rows)
    ov = M.pairwise_overlap([a, b], ref)
    assert ov["matrix_mean_jaccard"][0][1] == pytest.approx(1.0)
    assert ov["matrix_pooled_jaccard"][0][1] == pytest.approx(1.0)
    assert ov["mean_offdiagonal"] == pytest.approx(1.0)
    assert "NULL RESULT" in M.verdict(ov, ["a", "b"])


def test_scaling_a_dump_does_not_change_overlap(tmp_path):
    ref = M.__dict__["sc"].load_reference(synth_reference(tmp_path))["ref"]
    rows = [[random.Random(l).random() for _ in range(E)] for l in LAYERS]
    a = panel_from(tmp_path, "a", rows)
    b = panel_from(tmp_path, "b", rows, scale=987.0)
    ov = M.pairwise_overlap([a, b], ref)
    assert ov["matrix_mean_jaccard"][0][1] == pytest.approx(1.0)


def test_disjoint_top_sets_overlap_exactly_zero(tmp_path):
    """N is the reference's n_k4 (50 here), so 0..49 vs 50..99 is disjoint."""
    ref = M.__dict__["sc"].load_reference(synth_reference(tmp_path))["ref"]
    a = panel_from(tmp_path, "a", hot_rows(range(0, 50)))
    b = panel_from(tmp_path, "b", hot_rows(range(50, 100)))
    ov = M.pairwise_overlap([a, b], ref)
    assert ov["matrix_mean_jaccard"][0][1] == pytest.approx(0.0)
    assert ov["matrix_pooled_jaccard"][0][1] == pytest.approx(0.0)
    assert "DIFFERENT" in M.verdict(ov, ["a", "b"])


def test_half_overlap_reads_as_a_third(tmp_path):
    """25 of 50 shared -> |A n B| / |A u B| = 25/75."""
    ref = M.__dict__["sc"].load_reference(synth_reference(tmp_path))["ref"]
    a = panel_from(tmp_path, "a", hot_rows(range(0, 50)))
    b = panel_from(tmp_path, "b", hot_rows(range(25, 75)))
    ov = M.pairwise_overlap([a, b], ref)
    assert ov["matrix_mean_jaccard"][0][1] == pytest.approx(25 / 75)


def test_matrix_is_symmetric_with_unit_diagonal(tmp_path):
    ref = M.__dict__["sc"].load_reference(synth_reference(tmp_path))["ref"]
    ps = [panel_from(tmp_path, n, hot_rows(range(i * 30, i * 30 + 50)))
          for i, n in enumerate("abc")]
    ov = M.pairwise_overlap(ps, ref)
    m = ov["matrix_mean_jaccard"]
    for i in range(3):
        assert m[i][i] == 1.0
        for j in range(3):
            assert m[i][j] == pytest.approx(m[j][i])


def test_chance_floor_matches_the_analytic_value(tmp_path):
    ref = M.__dict__["sc"].load_reference(synth_reference(tmp_path))["ref"]
    a = panel_from(tmp_path, "a", hot_rows(range(0, 50)))
    ov = M.pairwise_overlap([a, a], ref)
    # 50 of 256, twice: E|A n B| = 50*50/256 -> J = i/(2n - i)
    inter = 50 * 50 / E
    assert ov["chance_floor"] == pytest.approx(inter / (100 - inter))


def test_layers_the_reference_cannot_score_are_skipped(tmp_path):
    """A layer whose reference set is empty would score 1.0 for free."""
    p = tmp_path / "ref-with-empty-layer.json"
    doc = json.loads(synth_reference(tmp_path).read_text())
    doc["per_layer_k4_sets"]["6"] = {"n_k3": E, "n_k4": 0, "k4_experts": []}
    p.write_text(json.dumps(doc))
    ref = M.__dict__["sc"].load_reference(p)["ref"]
    a = panel_from(tmp_path, "a", hot_rows(range(0, 50)))
    b = panel_from(tmp_path, "b", hot_rows(range(50, 100)))
    ov = M.pairwise_overlap([a, b], ref)
    assert ov["layers_scored"] == len(LAYERS) - 1
    assert 6 in ov["layers_skipped"]
    assert ov["matrix_mean_jaccard"][0][1] == pytest.approx(0.0)


def test_overlap_uses_the_reference_cardinality_not_a_fixed_n(tmp_path):
    """Top-N tracks n_k4: with n_k4=8, sharing experts 0..7 is a perfect
    match even though the panels differ wildly further down the ranking."""
    ref8 = M.__dict__["sc"].load_reference(
        synth_reference(tmp_path, n_k4=8))["ref"]
    rows_a = hot_rows(range(0, 8), 100.0, 1.0)
    rows_b = [list(r) for r in rows_a]
    for r in rows_b:                      # differ only outside the top 8
        for e in range(8, E):
            r[e] = 1.0 + (e % 7) * 0.1
    a = panel_from(tmp_path, "a", rows_a)
    b = panel_from(tmp_path, "b", rows_b)
    assert M.pairwise_overlap([a, b], ref8)["matrix_mean_jaccard"][0][1] \
        == pytest.approx(1.0)


# ------------------------------------------------------ 4. synthetic fencing

def _figure(tmp_path, names, synthetic: bool) -> M.Figure:
    ref_p = synth_reference(tmp_path)
    R = M.__dict__["sc"].load_reference(ref_p)
    panels = []
    for i, n in enumerate(names):
        if synthetic and i:
            panels.append(M.synth_axis(n, LAYERS, E))
        else:
            panels.append(panel_from(tmp_path, n,
                                     hot_rows(range(i * 10, i * 10 + 40))))
    perm, h = M.build_permutation(panels)
    ov = M.pairwise_overlap(panels, R["ref"])
    return M.Figure(panels=panels, ref=R["ref"], ref_name="synthetic/ref",
                    overlap=ov, perm=perm, perm_hash=h, order_mode="pooled",
                    signal="count", title="t", subtitle="s")


def test_real_figure_carries_no_watermark(tmp_path):
    fig = _figure(tmp_path, ["a", "b"], synthetic=False)
    assert fig.is_synthetic is False
    assert M.WATERMARK not in M.svg_bytes(fig)


def test_any_synthetic_panel_watermarks_the_whole_figure(tmp_path):
    fig = _figure(tmp_path, ["a", "b", "c"], synthetic=True)
    assert fig.is_synthetic is True
    svg = M.svg_bytes(fig)
    assert svg.count(M.WATERMARK) >= 3      # banner + repeated diagonals
    assert "SYNTHETIC PLACEHOLDER" in svg   # per-panel tag
    assert "fabricated" in svg.lower()


def test_watermark_cannot_be_removed_by_blanking_the_constant(tmp_path,
                                                              monkeypatch):
    """The writer's guard compares against a literal, so emptying the
    constant fails closed instead of producing an unmarked preview."""
    fig = _figure(tmp_path, ["a", "b"], synthetic=True)
    monkeypatch.setattr(M, "WATERMARK", "")
    with pytest.raises(RuntimeError, match="watermark"):
        M.svg_bytes(fig)


def test_watermark_survives_a_stubbed_out_watermark_helper(tmp_path,
                                                           monkeypatch):
    fig = _figure(tmp_path, ["a", "b"], synthetic=True)
    monkeypatch.setattr(M, "_watermark", lambda W, H: [])
    # the banner alone still marks it; if BOTH were removed the guard fires
    assert M.WATERMARK in M.svg_bytes(fig)
    monkeypatch.setattr(M, "render", lambda fig: "<svg/>")
    with pytest.raises(RuntimeError, match="watermark"):
        M.svg_bytes(fig)


def test_no_cli_flag_can_switch_the_watermark_off():
    ap_src = Path(M.__file__).read_text()
    for forbidden in ("--no-watermark", "--no-mark", "--clean",
                      "--suppress-watermark", "--unmarked"):
        assert forbidden not in ap_src
    with pytest.raises(SystemExit):
        M.main(["--no-watermark", "--out", "/dev/null"])


def test_synthetic_output_is_renamed(tmp_path):
    assert M.synthetic_path(Path("/x/panels.svg")).name == "panels.SYNTHETIC.svg"
    # already marked -> left alone, so repeated runs stay on one path
    p = Path("/x/panels.SYNTHETIC.svg")
    assert M.synthetic_path(p) == p


def test_synthetic_panels_are_hatched(tmp_path):
    fig = _figure(tmp_path, ["a", "b"], synthetic=True)
    assert 'class="hatch"' in M.svg_bytes(fig)
    real = _figure(tmp_path, ["a", "b"], synthetic=False)
    assert 'class="hatch"' not in M.svg_bytes(real)


def test_synth_axis_is_deterministic_and_flagged():
    a = M.synth_axis("axis1_general", LAYERS, E)
    b = M.synth_axis("axis1_general", LAYERS, E)
    assert a.rel == b.rel
    assert a.synthetic and "SYNTHETIC" in a.source
    c = M.synth_axis("axis2_legal", LAYERS, E)
    assert c.rel != a.rel


# ------------------------------------------------------------ 5. end to end

def test_cli_refuses_to_fill_gaps_without_the_flag(tmp_path):
    ref = synth_reference(tmp_path)
    dump = write_dump(tmp_path, "a.jsonl", hot_rows(range(0, 40)))
    with pytest.raises(SystemExit) as e:
        M.main(["--axis", f"axis1_general={dump}", "--reference", str(ref),
                "--out", str(tmp_path / "o.svg")])
    assert "allow-synthetic" in str(e.value)
    assert not (tmp_path / "o.svg").exists()


def test_cli_end_to_end_is_valid_svg_and_writes_json(tmp_path, capsys):
    ref = synth_reference(tmp_path)
    dumps = {n: write_dump(tmp_path, f"{n}.jsonl",
                           hot_rows(range(i * 20, i * 20 + 60)))
             for i, n in enumerate(M.CORPUS_AXES)}
    out = tmp_path / "panels.svg"
    rc = M.main(["--reference", str(ref), "--out", str(out),
                 "--json", str(tmp_path / "panels.json")]
                + [f"--axis={n}={p}" for n, p in dumps.items()])
    assert rc == 0
    assert out.exists()                      # real data -> name untouched
    root = ET.parse(out).getroot()
    assert root.tag.endswith("svg")
    txt = out.read_text()
    assert M.WATERMARK not in txt
    doc = json.loads((tmp_path / "panels.json").read_text())
    assert doc["synthetic"] is False
    assert doc["domain_log2_rel_uniform"] == [-4.0, 4.0]
    assert len(doc["permutation"]) == len(LAYERS)
    assert [p["name"] for p in doc["panels"]] == list(M.CORPUS_AXES)
    printed = capsys.readouterr().out
    assert "domain = [-4.000, +4.000]" in printed
    assert "chance floor" in printed


def test_cli_preview_path_is_renamed_and_marked(tmp_path):
    ref = synth_reference(tmp_path)
    dump = write_dump(tmp_path, "a.jsonl", hot_rows(range(0, 40)))
    out = tmp_path / "panels.svg"
    M.main(["--axis", f"axis3_code_agentic={dump}", "--reference", str(ref),
            "--out", str(out), "--allow-synthetic",
            "--json", str(tmp_path / "p.json")])
    assert not out.exists()
    marked = tmp_path / "panels.SYNTHETIC.svg"
    assert M.WATERMARK in marked.read_text()
    assert not (tmp_path / "p.json").exists()   # sidecar is renamed too
    doc = json.loads((tmp_path / "p.SYNTHETIC.json").read_text())
    assert doc["synthetic"] is True
    assert doc["synthetic_panels"] == ["axis1_general", "axis2_legal",
                                       "axis4_reasoning_termination"]


def test_panels_must_share_one_row_axis(tmp_path):
    ref = synth_reference(tmp_path)
    a = write_dump(tmp_path, "a.jsonl", hot_rows(range(0, 40)))
    short = write_dump(tmp_path, "b.jsonl",
                       [[1.0] * E for _ in LAYERS[:2]], layers=LAYERS[:2])
    with pytest.raises(SystemExit, match="row axis"):
        M.main(["--axis", f"axis1_general={a}", "--axis", f"axis2_legal={short}",
                "--reference", str(ref), "--out", str(tmp_path / "o.svg"),
                "--allow-synthetic"])


def test_mass_provenance_is_read_from_the_flag_never_inferred(tmp_path):
    """count == mass in these fixtures; only the FLAG may decide."""
    rows = hot_rows(range(0, 40))
    unknown = M.load_axis("u", write_dump(tmp_path, "u.jsonl", rows),
                          signal="mass", all_intervals=False)
    assert "UNKNOWN" in unknown.mass_state
    aliased = M.load_axis("a", write_dump(tmp_path, "a2.jsonl", rows,
                                          mass_is_real=False),
                          signal="mass", all_intervals=False)
    assert aliased.mass_state == "ALIASED TO COUNT"
    real = M.load_axis("r", write_dump(tmp_path, "r.jsonl", rows,
                                       mass_is_real=True),
                       signal="mass", all_intervals=False)
    assert real.mass_state == "REAL gate mass"


def test_last_interval_is_the_default_window(tmp_path):
    """The collector's count is a decayed window; accumulating every dumped
    interval is a different quantity and must be asked for."""
    p = write_dump(tmp_path, "multi.jsonl", hot_rows(range(0, 40)), records=4)
    last = M.load_axis("l", p, signal="count", all_intervals=False)
    allv = M.load_axis("a", p, signal="count", all_intervals=True)
    assert allv.total == pytest.approx(4 * last.total)
    assert last.rel == allv.rel        # share is unchanged; volume is not
    assert last.records == 4


def test_reference_layers_absent_from_the_dump_are_left_blank(tmp_path):
    """Reference has layers 3-6 plus 78; the dump has 3-6. Layer 78 must not
    shift the strip by a row (DESIGN hazard 2)."""
    doc = json.loads(synth_reference(tmp_path).read_text())
    doc["per_layer_k4_sets"]["78"] = {"n_k3": E, "n_k4": 0, "k4_experts": []}
    ref_p = tmp_path / "ref78.json"
    ref_p.write_text(json.dumps(doc))
    R = M.__dict__["sc"].load_reference(ref_p)["ref"]
    out: list[str] = []
    perm = [list(range(E)) for _ in LAYERS]
    h, missing = M.draw_reference_strip(out, R, LAYERS, perm, 0, 0, 3)
    assert missing == 0 and h == len(LAYERS) * 3


def test_reference_strip_blanks_a_layer_the_reference_lacks(tmp_path):
    R = M.__dict__["sc"].load_reference(synth_reference(tmp_path))["ref"]
    out: list[str] = []
    perm = [list(range(E)) for _ in LAYERS + [99]]
    _h, missing = M.draw_reference_strip(out, R, LAYERS + [99], perm, 0, 0, 3)
    assert missing == 1
    assert any('class="gap"' in s for s in out)


def test_figure_geometry_stays_inside_the_viewbox(tmp_path):
    fig = _figure(tmp_path, ["a", "b"], synthetic=False)
    root = ET.fromstring(M.svg_bytes(fig))
    W, H = (int(v) for v in root.get("viewBox").split()[2:])
    for el in root.iter():
        if el.tag.endswith("text"):
            assert 0 <= float(el.get("y")) <= H
        if el.tag.endswith("rect") and el.get("width") and el.get("x"):
            assert float(el.get("x")) + float(el.get("width")) <= W + 1


def test_wrap_never_drops_a_word():
    s = ("a caveat that must not be truncated because a truncated caveat "
         "quietly stops being a caveat at all " * 3)
    got = " ".join(M._wrap(s, 400))
    assert got.split() == s.split()


def test_human_readable_magnitudes():
    assert M.human(5.05e8) == "505.00M"
    assert M.human(0) == "0"
    assert M.human(float("nan")) == "n/a"
