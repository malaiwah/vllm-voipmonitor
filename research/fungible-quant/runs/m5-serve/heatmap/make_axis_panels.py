#!/usr/bin/env python3
"""Render the flagship 4-panel expert-activation figure: one panel per corpus axis.

THE CLAIM UNDER TEST
--------------------
The MTP78 calibration corpus has four equal axes (general / legal /
code-agentic / reasoning-termination, 3,057 rows each). Fungible quant only
pays for itself if *different traffic lights up different experts* — if all
four axes route to the same experts, one static allocation serves everything
and there is nothing to re-tier at runtime.

This tool renders that comparison and, crucially, **measures it**: beneath the
panels it prints the pairwise overlap of each axis pair's top-N hottest
experts, at the reference quant's own per-layer cardinality. If the four axes
route nearly identically the matrix says so in numbers and the verdict line
says so in words. A null result rendered honestly is the correct output.

DESIGN
------
Follows `heatmap/DESIGN.md` (same directory), which derived these choices from
the real dumps and from CVD-simulated colour measurements. The parts that are
load-bearing here:

* **ONE shared colour scale, never auto-scaled** (§5.2). Value is
  ``log2(share / uniform share)`` on a fixed ±4 domain, so a 7 M-assignment
  replay and a 506 M-assignment replay are directly comparable — raw counts
  across our dumps span 71x, and a raw-count rendering is a rendering of
  transcript length. The bounds are printed on the figure and on stdout.
* **ONE shared column permutation** (§4.3, hazard §7.1). Per-panel sorting
  would make every panel a monotone gradient by construction and destroy the
  comparison. The permutation is computed once from the pooled mean of all
  panels, hashed, threaded through every panel including the reference strip,
  and asserted equal at draw time.
* **Purples-9 for magnitude, Greens-4 for K tier** (§3.1, §3.2), single-hue so
  hue carries no information inside a panel.
* Dependency-free SVG, light/dark via ``prefers-color-scheme``, in the house
  style of ``../make_charts.py``.

SYNTHETIC PLACEHOLDERS
----------------------
There is no live per-axis data yet (see CAPTURE-RECIPE.md for how to collect
it). Missing axes can be synthesised so the layout can be reviewed, but that
output is fenced off hard: ``--allow-synthetic`` is required, every synthetic
panel is hatched and tagged, the whole figure carries a repeated
"SYNTHETIC LAYOUT PREVIEW" watermark, the output filename gains a
``.SYNTHETIC`` infix, and the writer *refuses* to emit a file whose bytes do
not contain the watermark. There is deliberately no flag to turn any of that
off.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_M5 = _HERE.parent
sys.path.insert(0, str(_M5))
import score_convergence as sc  # noqa: E402  — ONE implementation of Jaccard

# The four corpus axes. Imported from the corpus loader when it is reachable
# so the names cannot drift from the thing that actually filters the rows;
# the literal is only a fallback for a checkout without the harness.
try:  # pragma: no cover - trivial import shim
    sys.path.insert(0, str(_M5 / "harness"))
    from load_mtp78_corpus import AXES as CORPUS_AXES  # type: ignore
except Exception:  # noqa: BLE001
    CORPUS_AXES = ("axis1_general", "axis2_legal", "axis3_code_agentic",
                   "axis4_reasoning_termination")

# ---------------------------------------------------------------- constants

#: Fixed colour domain, in log2 multiples of uniform share. NEVER derived from
#: the data: auto-scaling per panel is exactly the bug this figure exists to
#: avoid, and auto-scaling per figure would make two figures incomparable.
DOMAIN = (-4.0, 4.0)
N_BANDS = 9

#: The watermark. The guard in :func:`svg_bytes` compares against a *literal*
#: copy of this string, so blanking this constant makes the writer refuse
#: rather than silently emit an unmarked preview.
WATERMARK = "SYNTHETIC LAYOUT PREVIEW"

#: ColorBrewer Purples-9 (DESIGN §3.1): monotone L* 98.7 -> 18.1, single hue,
#: valid as a grayscale ramp under every CVD.
PURPLES = ["#FCFBFD", "#EFEDF5", "#DADAEB", "#BCBDDC", "#9E9AC8",
           "#807DBA", "#6A51A3", "#54278F", "#3F007D"]
#: Dark surface: the SAME measured ramp, reversed, so "hot" is always the pole
#: furthest from the page. The legend swatches use the same classes, so the
#: reading stays self-consistent in either theme.
PURPLES_DARK = list(reversed(PURPLES))

#: K tier (DESIGN §3.2) — fixed by K value, never reassigned by frequency.
TIER_FILL = {2: "#BBEFC7", 3: "#73C189", 4: "#289352", 5: "#096231"}

#: "Never routed" is OFF the scale, so it is painted a NEUTRAL gray — the
#: magnitude ramp is strictly single-hue purple, which makes a gray cell
#: unambiguous. Painting it the surface colour (DESIGN §5.2's suggestion) is
#: wrong at 3 px: the coldest purple band is #FCFBFD, and "never routed"
#: would be invisible against "rarely routed".
LIGHT = {"surf": "#fcfcfb", "tx": "#0b0b0b", "tx2": "#52514e",
         "axis": "#b8b7b2", "grid": "#e6e5e1", "band": "#0b0b0b",
         "warn": "#eb6834", "warntx": "#ffffff", "dead": "#b0b0aa"}
DARK = {"surf": "#1a1a19", "tx": "#ffffff", "tx2": "#c3c2b7",
        "axis": "#56554f", "grid": "#33322f", "band": "#ffffff",
        "warn": "#d95926", "warntx": "#ffffff", "dead": "#6e6e66"}


def esc(s) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


def human(n: float) -> str:
    """Compact magnitude for provenance lines (assignments run to 5e8)."""
    if not math.isfinite(n):
        return "n/a"
    for unit, div in (("G", 1e9), ("M", 1e6), ("k", 1e3)):
        if abs(n) >= div:
            return f"{n / div:.2f}{unit}"
    return f"{n:.0f}"


# ------------------------------------------------------------- panel maths

def rel_uniform(row) -> list[float]:
    """Per-layer counts -> share in multiples of uniform share (1/E).

    A layer that never routed (total 0) yields all zeros rather than a
    ZeroDivisionError; those cells are then counted as dead, not as cold.
    """
    total = float(sum(row))
    e = len(row)
    if total <= 0.0:
        return [0.0] * e
    return [e * float(v) / total for v in row]


def to_scale(rel: float) -> float:
    """multiples-of-uniform -> the plotted value, log2, clipped to DOMAIN.

    A never-routed cell (rel == 0) floors at the bottom of the domain instead
    of going to -inf; it is *also* reported separately (see :func:`cell_flags`)
    and drawn in the surface colour, because "never routed" and "rarely
    routed" must not look the same.
    """
    if rel <= 0.0:
        return DOMAIN[0]
    return max(DOMAIN[0], min(DOMAIN[1], math.log2(rel)))


def cell_flags(rel: float) -> str:
    """'dead' | 'lo' (clipped below) | 'hi' (clipped above) | 'ok'."""
    if rel <= 0.0:
        return "dead"
    v = math.log2(rel)
    if v < DOMAIN[0]:
        return "lo"
    if v > DOMAIN[1]:
        return "hi"
    return "ok"


def band_of(value: float) -> int:
    """Plotted value -> colour band index in [0, N_BANDS).

    Equal-width bands across the FIXED domain. With an odd band count the
    centre band straddles log2 = 0, so "about its fair share" is one quiet
    band with four hotter and four colder bands either side.
    """
    lo, hi = DOMAIN
    if value <= lo:
        return 0
    if value >= hi:
        return N_BANDS - 1
    frac = (value - lo) / (hi - lo)
    return min(N_BANDS - 1, int(frac * N_BANDS))


def band_edges() -> list[float]:
    lo, hi = DOMAIN
    return [lo + (hi - lo) * i / N_BANDS for i in range(N_BANDS + 1)]


def layer_entropy(rel) -> float:
    """Normalised routing entropy of one layer, from multiples-of-uniform."""
    e = len(rel)
    if e <= 1:
        return 1.0
    tot = sum(rel)
    if tot <= 0:
        return 0.0
    h = 0.0
    for v in rel:
        p = v / tot
        if p > 0:
            h -= p * math.log(p)
    return h / math.log(e)


# ------------------------------------------------------------------ panels

@dataclass
class Panel:
    """One corpus axis: its heat matrix plus everything needed to caption it."""
    name: str
    source: str
    synthetic: bool
    layers: list[int]
    rel: list[list[float]]            # [L][E] share in multiples of uniform
    signal: str = "count"
    mass_state: str = "n/a"
    step: int | None = None
    interval: int | None = None
    records: int | None = None
    total: float = 0.0
    dead: int = 0
    clipped_lo: int = 0
    clipped_hi: int = 0
    rowsum_dev: float = 0.0
    perm_hash: str | None = None      # set at draw time; asserted equal

    @property
    def n_experts(self) -> int:
        return len(self.rel[0]) if self.rel else 0

    @property
    def max_rel(self) -> float:
        return max((max(r) for r in self.rel), default=0.0)

    @property
    def entropies(self) -> list[float]:
        return [layer_entropy(r) for r in self.rel]

    def scale_rows(self) -> list[list[float]]:
        return [[to_scale(v) for v in row] for row in self.rel]


def _tail_record(path: Path, chunk: int = 4 << 20) -> dict:
    """Parse the LAST json line without holding the whole dump in memory.

    One record is ~800 kB and the biggest dump on disk is 93 MB; the score
    arrays already come from ``sc.load_stats``, so this only exists to read
    the small provenance fields (``step``, ``interval``, ``mass_is_real``).
    """
    size = path.stat().st_size
    want = min(chunk, size)
    with path.open("rb") as fh:
        while True:
            fh.seek(max(0, size - want))
            parts = [p for p in fh.read().split(b"\n") if p.strip()]
            # parts[-1] is complete whenever it began inside the window,
            # which is guaranteed once a newline precedes it (>= 2 parts) or
            # the whole file is in hand.
            if parts and (len(parts) >= 2 or want >= size):
                return json.loads(parts[-1].decode())
            if want >= size:
                break
            want = min(size, want * 4)
    raise SystemExit(f"{path}: no records")


def _count_records(path: Path) -> int:
    n = 0
    with path.open("rb") as fh:
        for line in fh:
            if line.strip():
                n += 1
    return n


def load_axis(name: str, path: Path, *, signal: str,
              all_intervals: bool) -> Panel:
    """Read one VLLM_FQ_DUMP_STATS jsonl into a Panel.

    The score arrays come from ``score_convergence.load_stats`` — one reader,
    one set of semantics (last interval by default, since the collector's
    ``count`` is an exponentially-decayed window and later intervals have seen
    more traffic).
    """
    layers, acc = sc.load_stats(path, signal=signal, last_only=not all_intervals)
    meta = _tail_record(path)
    rel = [rel_uniform(row) for row in acc]
    p = Panel(name=name, source=str(path), synthetic=False, layers=layers,
              rel=rel, signal=signal,
              step=meta.get("step"), interval=meta.get("interval"),
              records=_count_records(path))
    # mass provenance: read the FLAG, never infer it by comparing arrays (a
    # uniform router produces equal arrays legitimately). The archived dumps
    # predate the field entirely -> "unknown", not a guess either way.
    if signal != "mass":
        p.mass_state = "n/a (signal=count)"
    elif "mass_is_real" not in meta:
        p.mass_state = "UNKNOWN (mass_is_real absent from dump)"
    else:
        p.mass_state = ("REAL gate mass" if meta["mass_is_real"]
                        else "ALIASED TO COUNT")
    _finish(p, acc)
    return p


def _finish(p: Panel, acc: list[list[float]]) -> None:
    """Totals, clip/dead census, and the per-layer-total invariant check."""
    p.total = sum(sum(r) for r in acc)
    sums = [sum(r) for r in acc]
    base = sums[0] if sums and sums[0] else 0.0
    p.rowsum_dev = (max(abs(s - base) for s in sums) / base) if base else 0.0
    for row in p.rel:
        for v in row:
            f = cell_flags(v)
            if f == "dead":
                p.dead += 1
            elif f == "lo":
                p.clipped_lo += 1
            elif f == "hi":
                p.clipped_hi += 1


def synth_axis(name: str, layers: list[int], n_experts: int, *,
               seed: int = 20260811, mix: float = 0.55) -> Panel:
    """Fabricate a plausibly-shaped placeholder panel. LAYOUT ONLY.

    Shape only has to be plausible enough to judge the composition: a Zipf-ish
    per-layer profile, partly shared between axes (``mix``) so the pairwise
    matrix lands somewhere between "identical" and "unrelated" and the matrix
    block can be reviewed too. Nothing here is measured, which is why every
    consumer of the result is forced to say so.
    """
    rows: list[list[float]] = []
    # str.__hash__ is salted per process; a placeholder that changes shape
    # between runs would make "is this the same preview?" unanswerable.
    stable = int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)
    scale = 1.0 + (stable % 97)          # per-axis volume, to prove the
    for li, layer in enumerate(layers):  # normalisation kills run length
        shared = random.Random(f"shared:{seed}:{layer}")
        own = random.Random(f"{name}:{seed}:{layer}")
        s_order = list(range(n_experts))
        o_order = list(range(n_experts))
        shared.shuffle(s_order)
        own.shuffle(o_order)
        s_rank = [0] * n_experts
        o_rank = [0] * n_experts
        for r, e in enumerate(s_order):
            s_rank[e] = r
        for r, e in enumerate(o_order):
            o_rank[e] = r
        row = []
        for e in range(n_experts):
            zs = 1.0 / ((s_rank[e] + 1) ** 0.8)
            zo = 1.0 / ((o_rank[e] + 1) ** 0.8)
            v = (zs ** (1.0 - mix)) * (zo ** mix)
            row.append(v * (0.85 + 0.3 * own.random()))
        tot = sum(row)
        rows.append([v / tot * 1e5 * scale for v in row])
    p = Panel(name=name, source="SYNTHETIC — no dump", synthetic=True,
              layers=list(layers), rel=[rel_uniform(r) for r in rows],
              signal="synthetic", mass_state="n/a (synthetic)",
              records=0)
    _finish(p, rows)
    return p


# ------------------------------------------------------- shared permutation

def build_permutation(panels: list[Panel], mode: str = "pooled"
                      ) -> tuple[list[list[int]], str]:
    """ONE per-layer column order for the whole figure, plus its hash.

    ``pooled`` ranks experts by the mean plotted value across every panel, so
    no single axis is privileged; ``axis:NAME`` ranks by one named baseline
    (which makes the figure baseline-dependent — say so in the caption);
    ``native`` keeps expert id order (what an operator hunting a specific
    expert wants). Per-panel sorting is deliberately NOT on offer: it makes
    every panel a monotone gradient by construction and destroys the
    comparison.
    """
    n_layers = len(panels[0].layers)
    n_experts = panels[0].n_experts
    if mode.startswith("axis:"):
        want = mode[5:]
        chosen = [p for p in panels if p.name == want]
        if not chosen:
            raise SystemExit(f"--order {mode}: no panel named {want!r} "
                             f"(have {[p.name for p in panels]})")
        src = chosen
    else:
        src = panels
    perm: list[list[int]] = []
    for li in range(n_layers):
        if mode == "native":
            perm.append(list(range(n_experts)))
            continue
        pooled = [0.0] * n_experts
        for p in src:
            row = p.rel[li]
            for e in range(n_experts):
                pooled[e] += to_scale(row[e])
        order = sorted(range(n_experts), key=lambda e: (-pooled[e], e))
        perm.append(order)
    h = hashlib.sha256(
        json.dumps(perm, separators=(",", ":")).encode()).hexdigest()[:12]
    return perm, h


# ------------------------------------------------------- pairwise overlap

def _scorable(layers: list[int], ref: dict, n_experts: int
              ) -> list[tuple[int, int, int]]:
    """(row index, layer id, n_k4) for layers the reference can score.

    Same skip rule as score_convergence: a layer whose reference set is empty
    or everything is not a selection test, and scoring it 1.0 for free would
    inflate every number in the matrix.
    """
    out = []
    for i, layer in enumerate(layers):
        if layer not in ref:
            continue
        n = ref[layer]["n_k4"]
        if n == 0 or n >= n_experts:
            continue
        out.append((i, layer, n))
    return out


def pairwise_overlap(panels: list[Panel], ref: dict) -> dict:
    """Top-N overlap between every pair of axes, N = reference cardinality.

    "Hottest N experts" is taken at the *reference's own per-layer* K4 count,
    which is what makes this comparable to the convergence score: same
    cardinality, same selection rule (``sc.select_topk``), same metric
    (``sc.jaccard``). Identical dumps therefore read 1.0 and disjoint top sets
    read 0.0, exactly.
    """
    layers = panels[0].layers
    n_experts = panels[0].n_experts
    rows = _scorable(layers, ref, n_experts)
    if not rows:
        raise SystemExit("no scorable layers — do the dump's layer ids match "
                         "the reference's?")
    tops = []
    for p in panels:
        tops.append([sc.select_topk(p.rel[i], n) for i, _l, n in rows])
    ref_sets = [ref[layer]["k4"] for _i, layer, _n in rows]

    def mean_j(a, b):
        return sum(sc.jaccard(x, y) for x, y in zip(a, b)) / len(rows)

    def pooled_j(a, b):
        inter = sum(len(x & y) for x, y in zip(a, b))
        union = sum(len(x | y) for x, y in zip(a, b))
        return inter / union if union else 0.0

    n = len(panels)
    matrix = [[1.0 if i == j else mean_j(tops[i], tops[j])
               for j in range(n)] for i in range(n)]
    pooled = [[1.0 if i == j else pooled_j(tops[i], tops[j])
               for j in range(n)] for i in range(n)]
    chance = sum(sc.chance_jaccard(k, k, n_experts) for _i, _l, k in rows) / len(rows)
    off = [matrix[i][j] for i in range(n) for j in range(n) if i < j]
    return {
        "layers_scored": len(rows),
        "layers_skipped": [l for l in layers
                           if l not in [x[1] for x in rows]],
        "cardinality": "reference per-layer n_k4",
        "n_k4_min_med_max": _mmm([k for _i, _l, k in rows]),
        "matrix_mean_jaccard": matrix,
        "matrix_pooled_jaccard": pooled,
        "vs_reference_mean_jaccard": [mean_j(t, ref_sets) for t in tops],
        "chance_floor": chance,
        "mean_offdiagonal": (sum(off) / len(off)) if off else float("nan"),
        "min_offdiagonal": min(off) if off else float("nan"),
        "max_offdiagonal": max(off) if off else float("nan"),
    }


def _mmm(v: list[int]) -> list[float]:
    s = sorted(v)
    return [s[0], s[len(s) // 2], s[-1]]


def verdict(ov: dict, names: list[str]) -> str:
    """One computed sentence. Says "no effect" when there is no effect."""
    if len(names) < 2:
        return "only one panel — no comparison possible"
    m = ov["mean_offdiagonal"]
    c = ov["chance_floor"]
    lift = (m / c) if c else float("nan")
    if m >= 0.90:
        return (f"NULL RESULT: the axes select nearly the same experts "
                f"(mean pairwise top-N Jaccard {m:.3f}, chance {c:.3f}) — "
                f"one static allocation would serve all of this traffic")
    if m >= 0.70:
        return (f"WEAK: axes overlap heavily (mean {m:.3f} vs chance {c:.3f}, "
                f"{lift:.2f}x) — differences exist but are a minority of the "
                f"selected set")
    return (f"DIFFERENT: axes select substantially different experts "
            f"(mean pairwise {m:.3f} vs chance {c:.3f}, {lift:.2f}x chance; "
            f"range {ov['min_offdiagonal']:.3f}-{ov['max_offdiagonal']:.3f})")


# ------------------------------------------------------------------ drawing

def _css(cell: int) -> str:
    out = [f"  .surf {{ fill: {LIGHT['surf']}; }}",
           "  .tx  { fill: %s; font-family: ui-sans-serif, system-ui, sans-serif; }" % LIGHT["tx"],
           "  .tx2 { fill: %s; font-family: ui-sans-serif, system-ui, sans-serif; }" % LIGHT["tx2"],
           "  .mono { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }",
           f"  .axis {{ stroke: {LIGHT['axis']}; stroke-width: 1; fill: none; }}",
           f"  .grid {{ stroke: {LIGHT['grid']}; stroke-width: 1; }}",
           f"  .band {{ fill: {LIGHT['band']}; opacity: 0.035; }}",
           f"  .warn {{ fill: {LIGHT['warn']}; }}",
           f"  .warntx {{ fill: {LIGHT['warntx']}; font-family: ui-sans-serif, system-ui, sans-serif; }}",
           f"  .mz {{ fill: {LIGHT['dead']}; }}",
           f"  .gap {{ fill: {LIGHT['surf']}; }}",
           "  .hatch { stroke: %s; stroke-width: 1; opacity: 0.35; }" % LIGHT["warn"],
           "  .mark { fill: %s; opacity: 0.13; font-family: ui-sans-serif, system-ui, sans-serif; font-weight: 700; }" % LIGHT["warn"]]
    for i, c in enumerate(PURPLES):
        out.append(f"  .m{i} {{ fill: {c}; }}")
    for k, c in TIER_FILL.items():
        out.append(f"  .k{k} {{ fill: {c}; }}")
    out.append("  @media (prefers-color-scheme: dark) {")
    out.append(f"    .surf {{ fill: {DARK['surf']}; }}")
    out.append(f"    .tx {{ fill: {DARK['tx']}; }} .tx2 {{ fill: {DARK['tx2']}; }}")
    out.append(f"    .axis {{ stroke: {DARK['axis']}; }} .grid {{ stroke: {DARK['grid']}; }}")
    out.append(f"    .band {{ fill: {DARK['band']}; opacity: 0.06; }}")
    out.append(f"    .warn {{ fill: {DARK['warn']}; }}")
    out.append(f"    .mz {{ fill: {DARK['dead']}; }}")
    out.append(f"    .gap {{ fill: {DARK['surf']}; }}")
    out.append(f"    .hatch {{ stroke: {DARK['warn']}; }}")
    out.append(f"    .mark {{ fill: {DARK['warn']}; opacity: 0.20; }}")
    for i, c in enumerate(PURPLES_DARK):
        out.append(f"    .m{i} {{ fill: {c}; }}")
    out.append("  }")
    return "\n".join(out)


def _runs(bands: list[int]) -> list[tuple[int, int, int]]:
    """Run-length encode one row: [(start, length, band), ...].

    19,200 cells x N panels as one <rect> each is 4-6x past where DOM
    rendering degrades and produces a multi-megabyte file. Under the shared
    sorted permutation neighbouring columns usually share a band, so this
    collapses a row of 256 cells into a couple of dozen rectangles with no
    loss whatsoever — the geometry is identical, just fewer nodes.
    """
    out, start = [], 0
    for i in range(1, len(bands) + 1):
        if i == len(bands) or bands[i] != bands[start]:
            out.append((start, i - start, bands[start]))
            start = i
    return out


def _cellpath(x: int, y: int, w: int, h: int) -> str:
    """One rectangle as path data — a third of the bytes of a <rect>.

    76,800 cells is a big enough figure that the encoding matters: as <rect>
    elements the flagship is 2.5 MB, as path subpaths grouped by colour it is
    under 1 MB, with byte-identical geometry.
    """
    return f"M{x} {y}h{w}v{h}h-{w}z"


def _hatch(out: list[str], x0: float, y0: float, w: float, h: float,
           step: int = 11) -> None:
    """Diagonal hatch over a rect, geometry clipped by hand.

    No <pattern>/<clipPath>: GitHub sanitises SVG it renders in markdown and
    the safest thing to hand it is plain <line> elements with real endpoints.
    """
    c = -h
    while c < w:
        ax, ay = x0 + c, y0
        bx, by = x0 + c + h, y0 + h
        if ax < x0:                      # clip on the left edge
            ay += (x0 - ax)
            ax = x0
        if bx > x0 + w:                  # clip on the right edge
            by -= (bx - (x0 + w))
            bx = x0 + w
        if bx > ax and by > ay:
            out.append(f'<line x1="{ax:.0f}" y1="{ay:.0f}" x2="{bx:.0f}" '
                       f'y2="{by:.0f}" class="hatch"/>')
        c += step


def draw_matrix(out: list[str], panel: Panel, perm: list[list[int]],
                perm_hash: str, x0: int, y0: int, cell: int) -> int:
    """Draw one heat panel. Returns its height in px."""
    if panel.perm_hash not in (None, perm_hash):
        raise RuntimeError(
            f"panel {panel.name} was prepared with permutation "
            f"{panel.perm_hash} but the figure uses {perm_hash} — a "
            f"permutation mismatch between panels is silent and "
            f"catastrophic (DESIGN §7.1); refusing to draw")
    panel.perm_hash = perm_hash
    n_layers, n_experts = len(panel.layers), panel.n_experts
    w, h = n_experts * cell, n_layers * cell
    scaled = panel.scale_rows()
    buckets: dict[int, list[str]] = {}
    dead: list[str] = []
    for li in range(n_layers):
        order = perm[li]
        row_rel = panel.rel[li]
        row_val = scaled[li]
        bands = [band_of(row_val[e]) for e in order]
        y = y0 + li * cell
        for start, length, b in _runs(bands):
            buckets.setdefault(b, []).append(
                _cellpath(x0 + start * cell, y, length * cell, cell))
        for ci, e in enumerate(order):   # dead cells overpaint their band
            if row_rel[e] <= 0.0:
                dead.append(_cellpath(x0 + ci * cell, y, cell, cell))
    for b in sorted(buckets):
        out.append(f'<path class="m{b}" d="' + "".join(buckets[b]) + '"/>')
    if dead:
        out.append('<path class="mz" d="' + "".join(dead) + '"/>')
    if panel.synthetic:
        _hatch(out, x0, y0, w, h)
    out.append(f'<rect x="{x0}" y="{y0}" width="{w}" height="{h}" class="axis"/>')
    return h


def draw_reference_strip(out: list[str], ref: dict, layers: list[int],
                         perm: list[list[int]], x0: int, y0: int,
                         cell: int) -> tuple[int, int]:
    """The human's K4 bitmap under the SAME permutation. (height, missing)."""
    n_experts = len(perm[0])
    k4_rects, k3_rects, gap_rects = [], [], []
    missing = 0
    for li, layer in enumerate(layers):
        y = y0 + li * cell
        if layer not in ref:
            missing += 1
            gap_rects.append(_cellpath(x0, y, n_experts * cell, cell))
            continue
        k4 = ref[layer]["k4"]
        bands = [1 if e in k4 else 0 for e in perm[li]]
        for start, length, b in _runs(bands):
            r = _cellpath(x0 + start * cell, y, length * cell, cell)
            (k4_rects if b else k3_rects).append(r)
    out.append('<path class="k3" d="' + "".join(k3_rects) + '"/>')
    out.append('<path class="k4" d="' + "".join(k4_rects) + '"/>')
    if gap_rects:
        out.append('<path class="gap" d="' + "".join(gap_rects) + '"/>')
    h = len(layers) * cell
    out.append(f'<rect x="{x0}" y="{y0}" width="{n_experts * cell}" '
               f'height="{h}" class="axis"/>')
    return h, missing


def _layer_ticks(out: list[str], layers: list[int], x0: int, y0: int,
                 cell: int, every: int = 12) -> None:
    last = len(layers) - 1
    # The final row is always labelled — a reader needs to see where the
    # stack ends — so a regular tick too close to it is dropped instead
    # (75 rows every 12 would otherwise print L75 and L77 two pixels apart).
    ticks = [i for i in range(0, last, every) if last - i >= 4] + [last]
    for li in ticks:
        layer = layers[li]
        y = y0 + li * cell + cell
        out.append(f'<text x="{x0 - 6}" y="{y:.0f}" text-anchor="end" '
                   f'class="tx2 mono" font-size="9">L{layer}</text>')


# ----------------------------------------------------------------- figure

@dataclass
class Figure:
    panels: list[Panel]
    ref: dict
    ref_name: str
    overlap: dict
    perm: list[list[int]]
    perm_hash: str
    order_mode: str
    signal: str
    title: str
    subtitle: str
    cell: int = 3
    notes: list[str] = field(default_factory=list)

    @property
    def is_synthetic(self) -> bool:
        return any(p.synthetic for p in self.panels)


def render(fig: Figure) -> str:
    cell = fig.cell
    n_experts = fig.panels[0].n_experts
    pw = n_experts * cell
    left, right = 62, 34
    W = left + pw + right
    o: list[str] = []
    y = 26

    o.append(f'<text x="{left}" y="{y}" class="tx" font-size="17" '
             f'font-weight="600">{esc(fig.title)}</text>')
    y += 19
    o.append(f'<text x="{left}" y="{y}" class="tx2" font-size="11">'
             f'{esc(fig.subtitle)}</text>')
    y += 15
    lo, hi = DOMAIN
    for line in (
        f"ONE colour scale, shared by every panel, never auto-scaled:",
        f"    value  = log2( share ÷ uniform share 1/{n_experts} )",
        f"    domain = [{lo:+.3f}, {hi:+.3f}]  =  [{2 ** lo:g}×, "
        f"{2 ** hi:g}×] of uniform share, in {N_BANDS} equal bands",
        f"    columns = {fig.order_mode} order, perm sha256 "
        f"{fig.perm_hash} — the SAME permutation in every panel"
        f"   ·   signal = {fig.signal}",
    ):
        o.append(f'<text x="{left}" y="{y}" class="tx2 mono" font-size="9.5">'
                 f'{esc(line)}</text>')
        y += 12
    y += 8

    if fig.is_synthetic:
        o.append(f'<rect x="{left}" y="{y - 2}" width="{pw}" height="20" '
                 f'class="warn"/>')
        o.append(f'<text x="{left + 8}" y="{y + 12}" class="warntx" '
                 f'font-size="11" font-weight="700">{esc(WATERMARK)} '
                 f'&#183; hatched panels are FABRICATED placeholder data, not '
                 f'measured routing</text>')
        y += 28

    letters = "ABCDEFGH"
    for idx, p in enumerate(fig.panels):
        ent = p.entropies
        ent_s = sorted(ent)
        tag = "   — SYNTHETIC PLACEHOLDER, NOT MEASURED" if p.synthetic else ""
        head = (f"{letters[idx % len(letters)]}  {p.name}{tag}")
        o.append(f'<text x="{left}" y="{y + 10}" class="tx" font-size="12" '
                 f'font-weight="600">{esc(head)}</text>')
        src = (p.source if p.synthetic else
               f"{Path(p.source).name} · interval {p.interval} "
               f"· {p.records} records")
        o.append(f'<text x="{left + pw}" y="{y + 10}" text-anchor="end" '
                 f'class="tx2 mono" font-size="9">{esc(src)}</text>')
        y += 13
        stat = (f"Σ {human(p.total)} assignments"
                f" · max {p.max_rel:.1f}× uniform"
                f" · layer entropy min/med {ent_s[0]:.3f}/"
                f"{ent_s[len(ent_s) // 2]:.3f}"
                f" · never routed {p.dead}"
                f" · clipped below {p.clipped_lo} / above {p.clipped_hi}")
        o.append(f'<text x="{left}" y="{y + 9}" class="tx2 mono" '
                 f'font-size="9">{esc(stat)}</text>')
        y += 13
        h = draw_matrix(o, p, fig.perm, fig.perm_hash, left, y, cell)
        _layer_ticks(o, p.layers, left, y, cell)
        y += h + 14

    # Reference bitmap: the human's K4 choice, same columns, same rows.
    o.append(f'<text x="{left}" y="{y + 10}" class="tx" font-size="12" '
             f'font-weight="600">R  reference quant — the experts a human '
             f'promoted to K4</text>')
    o.append(f'<text x="{left + pw}" y="{y + 10}" text-anchor="end" '
             f'class="tx2 mono" font-size="9">{esc(fig.ref_name)}</text>')
    y += 13
    o.append(f'<text x="{left}" y="{y + 9}" class="tx2 mono" font-size="9">'
             f'{esc("same rows, same columns as the panels above — read it "
                    "against them, not on its own")}</text>')
    y += 13
    h, missing = draw_reference_strip(o, fig.ref, fig.panels[0].layers,
                                      fig.perm, left, y, cell)
    _layer_ticks(o, fig.panels[0].layers, left, y, cell)
    y += h + 20

    y = _legend(o, left, y, pw, cell)
    y = _matrix_block(o, fig, left, y, pw)
    y = _caption(o, fig, left, y, pw, missing)

    H = int(y + 12)
    head = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
            f'width="{W}" height="{H}" role="img" '
            f'aria-label="{esc(fig.title)}">',
            f"<style>\n{_css(cell)}\n</style>",
            f'<rect width="{W}" height="{H}" class="surf"/>']
    body = head + o
    if fig.is_synthetic:
        body += _watermark(W, H)
    body.append("</svg>")
    return "\n".join(body)


def _watermark(W: int, H: int) -> list[str]:
    """Repeated diagonal watermark. Emitted whenever ANY panel is synthetic;
    there is no parameter, no flag and no attribute that suppresses it."""
    out = []
    step = 300
    for i in range(max(2, H // step) + 1):
        cy = 120 + i * step
        if cy > H - 40:          # a mark drawn off the canvas marks nothing
            break
        out.append(f'<text x="{W / 2:.0f}" y="{cy}" text-anchor="middle" '
                   f'class="mark" font-size="46" '
                   f'transform="rotate(-24 {W / 2:.0f} {cy})">{esc(WATERMARK)}'
                   f'</text>')
    return out


def _legend(o: list[str], x: int, y: int, pw: int, cell: int) -> float:
    sw, sh = 34, 12
    o.append(f'<text x="{x}" y="{y + 10}" class="tx2" font-size="10">'
             f'magnitude — share of the layer’s routing in multiples of '
             f'uniform (1× = its fair share of the 256)</text>')
    y += 16
    for i in range(N_BANDS):
        o.append(f'<rect x="{x + i * sw}" y="{y}" width="{sw}" height="{sh}" '
                 f'class="m{i}"/>')
    o.append(f'<rect x="{x}" y="{y}" width="{N_BANDS * sw}" height="{sh}" '
             f'class="axis"/>')
    # Ticks sit at round multiples of uniform, positioned by VALUE — the band
    # boundaries fall on 8/9ths and labelling those gives "1/4.66612x".
    lo, hi = DOMAIN
    ramp = N_BANDS * sw
    for v in range(int(lo), int(hi) + 1):
        if v % 2:
            continue
        xx = x + (v - lo) / (hi - lo) * ramp
        mult = 2.0 ** v
        lab = f"{mult:g}×" if mult >= 1 else f"1/{1 / mult:g}×"
        o.append(f'<text x="{xx:.1f}" y="{y + sh + 11}" text-anchor="middle" '
                 f'class="tx2 mono" font-size="9">{esc(lab)}</text>')
        o.append(f'<line x1="{xx:.1f}" y1="{y + sh}" x2="{xx:.1f}" '
                 f'y2="{y + sh + 3}" class="axis"/>')
    xk = x + N_BANDS * sw + 26
    o.append(f'<rect x="{xk}" y="{y}" width="{sh}" height="{sh}" class="mz"/>')
    o.append(f'<rect x="{xk}" y="{y}" width="{sh}" height="{sh}" class="axis"/>')
    o.append(f'<text x="{xk + sh + 5}" y="{y + 10}" class="tx2" font-size="9">'
             f'never routed (off scale)</text>')
    xk += 150
    for k in (3, 4):
        o.append(f'<rect x="{xk}" y="{y}" width="{sh}" height="{sh}" '
                 f'class="k{k}"/>')
        o.append(f'<text x="{xk + sh + 4}" y="{y + 10}" class="tx2" '
                 f'font-size="9">reference K{k}</text>')
        xk += 88
    return y + sh + 26


def _matrix_block(o: list[str], fig: Figure, x: int, y: int, pw: int) -> float:
    ov = fig.overlap
    names = [p.name for p in fig.panels]
    o.append(f'<text x="{x}" y="{y + 11}" class="tx" font-size="12" '
             f'font-weight="600">Pairwise overlap of the hottest experts '
             f'(mean per-layer Jaccard)</text>')
    y += 18
    sub = (f"top-N per layer at the reference’s own cardinality "
           f"(n_k4 min/median/max {ov['n_k4_min_med_max'][0]}/"
           f"{ov['n_k4_min_med_max'][1]}/{ov['n_k4_min_med_max'][2]}), "
           f"{ov['layers_scored']} layers · chance floor "
           f"{ov['chance_floor']:.4f} · identical traffic reads 1.000")
    o.append(f'<text x="{x}" y="{y + 10}" class="tx2" font-size="10">'
             f'{esc(sub)}</text>')
    y += 20

    label_w, cw, rh = 190, 78, 19
    for j, n in enumerate(names):
        o.append(f'<text x="{x + label_w + cw * j + cw / 2:.0f}" y="{y + 10}" '
                 f'text-anchor="middle" class="tx2 mono" font-size="9">'
                 f'{esc(_short(n))}</text>')
    o.append(f'<text x="{x + label_w + cw * len(names) + 34:.0f}" y="{y + 10}" '
             f'text-anchor="middle" class="tx2 mono" font-size="9">vs ref'
             f'</text>')
    y += 6
    for i, n in enumerate(names):
        row_y = y + i * rh
        if i % 2:
            o.append(f'<rect x="{x}" y="{row_y}" '
                     f'width="{label_w + cw * len(names) + 70}" '
                     f'height="{rh}" class="band"/>')
        tag = " [SYN]" if fig.panels[i].synthetic else ""
        o.append(f'<text x="{x + 4}" y="{row_y + 13}" class="tx2 mono" '
                 f'font-size="10">{esc(n + tag)}</text>')
        for j in range(len(names)):
            v = ov["matrix_mean_jaccard"][i][j]
            txt = "—" if i == j else f"{v:.3f}"
            weight = "600" if (i != j and v >= 0.9) else "400"
            o.append(f'<text x="{x + label_w + cw * j + cw / 2:.0f}" '
                     f'y="{row_y + 13}" text-anchor="middle" '
                     f'class="tx mono" font-size="10" '
                     f'font-weight="{weight}">{txt}</text>')
        rv = ov["vs_reference_mean_jaccard"][i]
        o.append(f'<text x="{x + label_w + cw * len(names) + 34:.0f}" '
                 f'y="{row_y + 13}" text-anchor="middle" class="tx2 mono" '
                 f'font-size="10">{rv:.3f}</text>')
    y += rh * len(names) + 16
    for piece in _wrap(verdict(ov, names), pw, cw=6.5):
        o.append(f'<text x="{x}" y="{y + 10}" class="tx" font-size="11" '
                 f'font-weight="600">{esc(piece)}</text>')
        y += 14
    return y + 12


def _short(n: str) -> str:
    return n.replace("axis", "a").replace("_", " ")[:13]


def _caption(o: list[str], fig: Figure, x: int, y: int, pw: int,
             missing: int) -> float:
    lines = [
        "What this hides, stated so it cannot be read wrong: the panels show "
        "SHARE, not volume — a burst and a soak look the same, so the "
        "per-panel Σ is printed above each panel.",
        f"The columns are SORTED ({fig.order_mode}), so the left-to-right "
        f"gradient is the sort, not a finding; what is readable is how far "
        f"each panel departs from that order. Same column = same expert in "
        f"every panel, per layer — expert id is not on the x axis, the "
        f"permutation is in the JSON sidecar.",
        f"{fig.signal} is the collector’s exponentially-decayed window "
        f"(defaults λ=0.95 over 64 slots of 32 steps ≈ the last 2,048 engine "
        f"steps), not a lifetime total. Gate-mass provenance, read from the "
        f"dump’s flag and never inferred from the arrays: "
        + "; ".join(sorted({p.mass_state for p in fig.panels
                            if not p.synthetic})) + ".",
    ]
    if missing:
        lines.append(f"{missing} layer(s) in the dumps have no entry in the "
                     f"reference and are blank in strip R.")
    for note in fig.notes:
        lines.append(note)
    if fig.is_synthetic:
        syn = [p.name for p in fig.panels if p.synthetic]
        lines.append("PLACEHOLDER PANELS (fabricated, layout review only): "
                     + ", ".join(syn) + ". Every number above that involves "
                     "them is meaningless. Collect the real dumps with "
                     "heatmap/CAPTURE-RECIPE.md.")
    for ln in lines:
        for piece in _wrap(ln, pw):
            y += 12
            o.append(f'<text x="{x}" y="{y}" class="tx2" font-size="9.5">'
                     f'{esc(piece)}</text>')
    return y + 8


def _wrap(s: str, pw: int, cw: float = 5.0) -> list[str]:
    """Greedy word wrap at an assumed character width.

    SVG <text> does not wrap and this renderer cannot measure a font, so the
    width is assumed conservatively (9.5 px body text ~5.0 px/char, 11 px
    semibold ~6.5). Erring wide costs a line break; erring narrow puts a
    caveat off the edge of the figure, which is how a caveat quietly stops
    being read.
    """
    limit = max(20, int(pw / cw))
    out, line = [], ""
    for word in s.split():
        cand = word if not line else f"{line} {word}"
        if len(cand) > limit:
            out.append(line)
            line = word
        else:
            line = cand
    if line:
        out.append(line)
    return out or [""]


def svg_bytes(fig: Figure) -> str:
    """Render, then refuse to hand back an unmarked synthetic figure.

    The comparison is against a LITERAL, so blanking ``WATERMARK`` (or any
    other clever way of emptying the mark) fails closed here instead of
    quietly producing a preview that looks like a measurement.
    """
    svg = render(fig)
    if fig.is_synthetic and "SYNTHETIC LAYOUT PREVIEW" not in svg:
        raise RuntimeError(
            "refusing to emit a figure that contains synthetic panels but "
            "not the synthetic watermark")
    return svg


def synthetic_path(out: Path) -> Path:
    """Force '.SYNTHETIC' into the filename of a placeholder figure."""
    if "SYNTHETIC" in out.name:
        return out
    return out.with_name(out.stem + ".SYNTHETIC" + out.suffix)


# --------------------------------------------------------------------- cli

def parse_axis_args(specs: list[str]) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for s in specs:
        if "=" not in s:
            raise SystemExit(f"--axis expects NAME=PATH, got {s!r}")
        name, path = s.split("=", 1)
        p = Path(path)
        if not p.exists():
            raise SystemExit(f"--axis {name}: no such dump {p}")
        out[name.strip()] = p
    return out


def print_report(fig: Figure) -> None:
    ov = fig.overlap
    lo, hi = DOMAIN
    print("=" * 78)
    if fig.is_synthetic:
        print(f"*** {WATERMARK} — some panels are FABRICATED ***")
    print(f"colour scale (shared by every panel, never auto-scaled):")
    print(f"  value  = log2(share / uniform share 1/{fig.panels[0].n_experts})")
    print(f"  domain = [{lo:+.3f}, {hi:+.3f}]  =  "
          f"[{2 ** lo:g}x, {2 ** hi:g}x] of uniform, {N_BANDS} equal bands")
    print(f"  column order = {fig.order_mode}, perm sha256 {fig.perm_hash} "
          f"(identical in every panel)")
    print("-" * 78)
    for p in fig.panels:
        flag = "SYNTHETIC" if p.synthetic else "real"
        print(f"  {p.name:<30} {flag:<9} sum={human(p.total):>8} "
              f"max={p.max_rel:6.2f}x dead={p.dead:<4} "
              f"clip_lo={p.clipped_lo:<4} clip_hi={p.clipped_hi:<4} "
              f"src={Path(p.source).name}")
        if not p.synthetic:
            print(f"  {'':<30} records={p.records} last interval={p.interval} "
                  f"step={p.step} mass={p.mass_state} "
                  f"per-layer-total dev={p.rowsum_dev:.2e}")
    print("-" * 78)
    names = [p.name for p in fig.panels]
    w = max(len(n) for n in names) + 2
    print(f"pairwise top-N overlap (mean per-layer Jaccard, N = reference "
          f"n_k4, {ov['layers_scored']} layers)")
    print(" " * w + "".join(f"{_short(n):>15}" for n in names) + f"{'vs ref':>10}")
    for i, n in enumerate(names):
        cells = "".join(
            f"{'--':>15}" if i == j else f"{ov['matrix_mean_jaccard'][i][j]:>15.4f}"
            for j in range(len(names)))
        print(f"{n:<{w}}{cells}{ov['vs_reference_mean_jaccard'][i]:>10.4f}")
    print(f"chance floor {ov['chance_floor']:.4f} · off-diagonal mean "
          f"{ov['mean_offdiagonal']:.4f} "
          f"(min {ov['min_offdiagonal']:.4f}, max {ov['max_offdiagonal']:.4f})")
    print(verdict(ov, names))
    print("=" * 78)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--axis", action="append", default=[], metavar="NAME=PATH",
                    help="a stats dump for one corpus axis; repeatable")
    ap.add_argument("--reference", type=Path,
                    default=_M5 / "reference-coder-quant.json")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--json", type=Path, help="write the numbers alongside")
    ap.add_argument("--signal", default="count", choices=["count", "mass"],
                    help="count is the default: the archived dumps predate "
                         "the mass_is_real field, so 'mass' cannot be shown "
                         "to be real for them")
    ap.add_argument("--all-intervals", action="store_true",
                    help="accumulate every interval instead of the last")
    ap.add_argument("--order", default="pooled", metavar="MODE",
                    help="column order: 'pooled' (mean of all panels, the "
                         "default), 'native' (expert id), or 'axis:NAME' to "
                         "rank by one named baseline")
    ap.add_argument("--cell", type=int, default=3, help="px per cell")
    ap.add_argument("--allow-synthetic", action="store_true",
                    help="permit fabricated placeholder panels for axes with "
                         "no dump; the result is watermarked and renamed")
    ap.add_argument("--axes", default=",".join(CORPUS_AXES),
                    help="the axis set the figure must show")
    ap.add_argument("--title", default="Which experts does each kind of "
                                       "traffic light up?")
    ap.add_argument("--subtitle",
                    default="GLM-5.2, 75 instrumented MoE layers (3-77) x 256 "
                            "experts, live TP4 serve; MTP78 calibration "
                            "corpus replayed one axis at a time")
    a = ap.parse_args(argv)

    given = parse_axis_args(a.axis)
    wanted = [x for x in a.axes.split(",") if x.strip()]
    for name in given:
        if name not in wanted:
            wanted.append(name)

    real = {n: load_axis(n, p, signal=a.signal,
                         all_intervals=a.all_intervals)
            for n, p in given.items()}
    if not real:
        raise SystemExit("no real dumps given — at least one --axis NAME=PATH "
                         "is required so the layer/expert geometry and the "
                         "signal are taken from measured data")
    probe = next(iter(real.values()))
    for p in real.values():
        if p.layers != probe.layers:
            raise SystemExit(
                f"{p.name} covers layers {p.layers[0]}-{p.layers[-1]} "
                f"({len(p.layers)}) but {probe.name} covers "
                f"{probe.layers[0]}-{probe.layers[-1]} ({len(probe.layers)}) "
                f"— panels must share one row axis")
        if p.n_experts != probe.n_experts:
            raise SystemExit(f"{p.name} has {p.n_experts} experts, "
                             f"{probe.name} has {probe.n_experts}")

    missing = [n for n in wanted if n not in real]
    if missing and not a.allow_synthetic:
        raise SystemExit(
            "no dump for: " + ", ".join(missing) + "\n"
            "Collect them with heatmap/CAPTURE-RECIPE.md, or pass "
            "--allow-synthetic to render a WATERMARKED layout preview with "
            "fabricated data in their place.")
    panels = []
    for n in wanted:
        panels.append(real[n] if n in real
                      else synth_axis(n, probe.layers, probe.n_experts))

    R = sc.load_reference(a.reference)
    ref = R["ref"]
    rdoc = R["doc"].get("reference") or {}
    ref_name = (rdoc.get("repo_id") or rdoc.get("repo")
                or Path(a.reference).name)
    perm, perm_hash = build_permutation(panels, a.order)
    ov = pairwise_overlap(panels, ref)

    notes = []
    bad = [p.name for p in panels if p.rowsum_dev > 1e-9]
    if bad:
        notes.append("per-layer totals are NOT equal across layers in "
                     + ", ".join(bad) + " (top-k routing should make them "
                     "identical) — per-layer share is used, so the panels "
                     "stay honest, but check the collector.")
    fig = Figure(panels=panels, ref=ref, ref_name=str(ref_name), overlap=ov,
                 perm=perm, perm_hash=perm_hash, order_mode=a.order,
                 signal=a.signal, title=a.title, subtitle=a.subtitle,
                 cell=a.cell, notes=notes)

    out = synthetic_path(a.out) if fig.is_synthetic else a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(svg_bytes(fig))
    print_report(fig)
    print(f"wrote {out} ({out.stat().st_size / 1024:.0f} kB)")
    if a.json:
        payload = {
            "schema": "fq-axis-panels/1",
            "synthetic": fig.is_synthetic,
            "synthetic_panels": [p.name for p in panels if p.synthetic],
            "signal": a.signal,
            "intervals": "all" if a.all_intervals else "last",
            "domain_log2_rel_uniform": list(DOMAIN),
            "bands": N_BANDS,
            "order_mode": a.order,
            "permutation_sha256_12": perm_hash,
            "permutation": perm,
            "reference": ref_name,
            "panels": [{"name": p.name, "source": p.source,
                        "synthetic": p.synthetic, "total": p.total,
                        "max_rel_uniform": p.max_rel, "dead_cells": p.dead,
                        "clipped": p.clipped_lo + p.clipped_hi,
                        "records": p.records, "interval": p.interval,
                        "step": p.step, "mass_state": p.mass_state,
                        "rowsum_rel_dev": p.rowsum_dev} for p in panels],
            "overlap": ov,
            "verdict": verdict(ov, [p.name for p in panels]),
            "notes": notes,
        }
        jout = synthetic_path(a.json) if fig.is_synthetic else a.json
        jout.parent.mkdir(parents=True, exist_ok=True)
        jout.write_text(json.dumps(payload, indent=1))
        print(f"wrote {jout}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
