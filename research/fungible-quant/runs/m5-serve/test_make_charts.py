"""Structural tests for the evidence charts.

There is no rasterizer on this box (no libcairo), so "look at it" cannot be an
image diff here. These tests stand in for the parts of looking that can be
mechanised: the SVG is well-formed, every drawn coordinate is finite and
inside the viewBox, both themes are defined, and the series identity is not
carried by colour alone. A human still eyeballs the final PNG/SVG in the PR.
"""
from __future__ import annotations

import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import make_charts as mc  # noqa: E402

W, H = 900, 560


def timeline(n=30, with_k4=False, swaps=True):
    rows, t, sw, occ5 = [], 1_000_000.0, 0.0, 0.0
    rows.append({"t": t, "kind": "phase", "event": "phase_start",
                 "family": "math"})
    for i in range(n):
        t += 5.0
        if swaps and i == n // 2:
            sw, occ5 = 12.0, 64.0
        occ = {"3": {"3": 256 - occ5, "5": occ5}}
        if with_k4:
            occ["3"]["4"] = 8.0
        rows.append({"t": t, "kind": "sample", "decode_tok_s": 600.0 + i,
                     "fq": {"swaps_total": sw, "tier_occupancy": occ}})
    return rows


def render_rows(rows):
    samples = [r for r in rows if r.get("kind") != "phase"]
    phases = [r for r in rows if r.get("kind") == "phase"]
    return mc.render(samples, phases, "T", "sub")


def test_output_is_wellformed_xml():
    ET.fromstring(render_rows(timeline()))


def test_no_nan_or_inf_coordinates():
    svg = render_rows(timeline())
    assert "nan" not in svg.lower()
    assert "inf" not in svg.lower().replace("font", "")


def test_all_path_points_inside_viewbox():
    svg = render_rows(timeline())
    for d in re.findall(r'<path d="([^"]+)"', svg):
        for x, y in re.findall(r"(-?\d+\.?\d*),(-?\d+\.?\d*)", d):
            assert -1 <= float(x) <= W + 1, f"x {x} outside viewBox"
            assert -1 <= float(y) <= H + 1, f"y {y} outside viewBox"


def test_both_light_and_dark_surfaces_defined():
    svg = render_rows(timeline())
    assert "prefers-color-scheme: dark" in svg
    # the light surface is defined on a bare rule, not only inside the media
    # query — otherwise a light viewer gets an unpainted background
    assert re.search(r"\.surf\s*\{\s*fill:\s*#fcfcfb", svg)
    assert "#1a1a19" in svg


def test_legend_labels_every_plotted_series():
    svg = render_rows(timeline(with_k4=True))
    for label in ("decode tok/s", "cumulative swaps", "experts at K5",
                  "experts at K4"):
        assert label in svg, f"missing legend entry: {label}"


def test_k4_series_omitted_when_absent_but_legend_stays_consistent():
    svg = render_rows(timeline(with_k4=False))
    assert "experts at K4" not in svg
    assert "experts at K5" in svg


def test_series_colors_are_the_fixed_theme_order():
    svg = render_rows(timeline())
    # slots 1..3 in order — never re-picked per chart
    assert svg.index("#2a78d6") < svg.index("#eb6834") < svg.index("#1baf7a")


def test_phase_boundaries_are_drawn_and_labelled():
    rows = timeline()
    rows.append({"t": 1_000_080.0, "kind": "phase", "event": "phase_start",
                 "family": "code"})
    svg = render_rows(rows)
    assert ">math<" in svg and ">code<" in svg
    assert "stroke-dasharray" in svg


def test_accessible_role_and_label():
    svg = render_rows(timeline())
    root = ET.fromstring(svg)
    assert root.get("role") == "img"
    assert root.get("aria-label")


def test_empty_timeline_fails_loudly_not_silently():
    with pytest.raises(SystemExit):
        mc.render([], [], "t", "s")


def test_missing_throughput_samples_do_not_break_the_line():
    """A scrape error row has no decode_tok_s; the chart must skip it."""
    rows = timeline()
    rows[5].pop("decode_tok_s")
    rows[5]["error"] = "URLError: refused"
    svg = render_rows(rows)
    ET.fromstring(svg)
    assert "<path" in svg


def test_load_splits_samples_from_phase_rows(tmp_path):
    p = tmp_path / "t.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in timeline()) + "\n")
    samples, phases = mc.load(p)
    assert len(phases) == 1 and len(samples) == 30
