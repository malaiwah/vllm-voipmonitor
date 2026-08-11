"""Maths tests for heatmap.html.

There is no browser and no rasteriser on this box, so the pixels cannot be
checked. What CAN be checked, and is checked here, is everything that decides
what those pixels would be:

  * the HTML parses, is self-contained, and carries no external reference;
  * the 64 KB of inline JavaScript parses under a real ECMAScript parser;
  * and -- the point of this file -- the page's OWN arithmetic, sliced out of
    the HTML at the ``==FQ-HEATMAP-PURE-END==`` marker and executed in QuickJS,
    agrees cell-for-cell with an independent NumPy implementation of DESIGN.md
    on the real dumps in ``results/k3-fq/``.

So the maths is tested even though the pixels are not.

Run::

    /home/mbelleau/venvs/fq/bin/python -m pytest test_heatmap_math.py -q
    /home/mbelleau/venvs/fq/bin/python test_heatmap_math.py          # same, no pytest

Optional deps, each skipped individually when missing::

    uv pip install --python /home/mbelleau/venvs/fq/bin/python esprima quickjs
"""

from __future__ import annotations

import hashlib
import html.parser
import json
import math
import os
import re
import struct

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
PAGE = os.path.join(HERE, "heatmap.html")
DUMPS = os.path.abspath(os.path.join(HERE, "..", "results", "k3-fq"))

DUMP_FILES = [
    "stats-code-axis.jsonl",
    "stats.jsonl",
    "stats-synthetic.jsonl",
    "stats-INVALID-truncated-corpus.jsonl",
]

# --------------------------------------------------------------------------
# DESIGN.md sections 3.1 / 3.2 / 3.3. These literals are duplicated on purpose:
# test_constants_block_matches_design asserts the page agrees with them, so a
# ramp cannot drift in the HTML without this file failing.
# --------------------------------------------------------------------------
RAMP_MAG = ["#FCFBFD", "#EFEDF5", "#DADAEB", "#BCBDDC", "#9E9AC8",
            "#807DBA", "#6A51A3", "#54278F", "#3F007D"]
RAMP_DIV = ["#005F9D", "#2D81C0", "#7DA4CE", "#C1C1C1",
            "#D89067", "#C85B37", "#9B3925"]
TIER_COLOR = {2: "#BBEFC7", 3: "#73C189", 4: "#289352", 5: "#096231"}
MAG_DOMAIN = 4.0
LUT_N = 257

# Measured by DESIGN.md section 5.2 on the final interval of stats-code-axis.jsonl,
# and independently reproduced here. If this changes, either the dump changed or
# the normalisation did.
GOLDEN_BANDS_CODE_AXIS = [2028, 2739, 3995, 4802, 3501, 1470, 487, 153, 25]


# ==========================================================================
# Independent reference implementation (NumPy / plain Python), written from
# DESIGN.md rather than transcribed from the page.
# ==========================================================================
XN, YN, ZN = 0.95047, 1.0, 1.08883
LEPS = 216.0 / 24389.0
LKAPPA = 24389.0 / 27.0


def hex_rgb(h):
    h = h.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def _s2l(c):
    c = c / 255.0
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def _l2s(c):
    v = 12.92 * c if c <= 0.0031308 else 1.055 * (c ** (1.0 / 2.4)) - 0.055
    # floor(x+0.5), matching JS Math.floor -- NOT Python's banker's round()
    v = math.floor(v * 255.0 + 0.5)
    return 0 if v < 0 else (255 if v > 255 else v)


def srgb_to_lab(rgb):
    r, g, b = _s2l(rgb[0]), _s2l(rgb[1]), _s2l(rgb[2])
    x = (0.4124564 * r + 0.3575761 * g + 0.1804375 * b) / XN
    y = (0.2126729 * r + 0.7151522 * g + 0.0721750 * b) / YN
    z = (0.0193339 * r + 0.1191920 * g + 0.9503041 * b) / ZN

    def f(t):
        return t ** (1.0 / 3.0) if t > LEPS else (LKAPPA * t + 16.0) / 116.0

    fx, fy, fz = f(x), f(y), f(z)
    return (116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz))


def lab_to_srgb(lab):
    L, a, b = lab
    fy = (L + 16.0) / 116.0
    fx, fz = fy + a / 500.0, fy - b / 200.0

    def fi(t):
        t3 = t * t * t
        return t3 if t3 > LEPS else (116.0 * t - 16.0) / LKAPPA

    x = fi(fx) * XN
    y = (((L + 16.0) / 116.0) ** 3 if L > LKAPPA * LEPS else L / LKAPPA) * YN
    z = fi(fz) * ZN
    return (_l2s(3.2404542 * x - 1.5371385 * y - 0.4985314 * z),
            _l2s(-0.9692660 * x + 1.8760108 * y + 0.0415560 * z),
            _l2s(0.0556434 * x - 0.2040259 * y + 1.0572252 * z))


def make_lut(stops, n=LUT_N):
    labs = [srgb_to_lab(hex_rgb(s)) for s in stops]
    m = len(labs) - 1
    out = []
    for i in range(n):
        t = (i / (n - 1.0)) * m
        k = int(math.floor(t))
        if k >= m:
            k, f = m - 1, 1.0
        else:
            f = t - k
        A, B = labs[k], labs[k + 1]
        out.append(lab_to_srgb((A[0] + (B[0] - A[0]) * f,
                                A[1] + (B[1] - A[1]) * f,
                                A[2] + (B[2] - A[2]) * f)))
    return out


def lut_index(v, lo, hi, n=LUT_N):
    i = int(math.floor(((v - lo) / (hi - lo)) * (n - 1) + 0.5))
    return 0 if i < 0 else (n - 1 if i > n - 1 else i)


def band9(v):
    """Nine unit-wide bands over [-4,+4]: band 0 is < -3, band 8 is >= +4."""
    if not math.isfinite(v):
        return 0
    b = math.floor(v) + 4
    return 0 if b < 0 else (8 if b > 8 else int(b))


def rank01(vals):
    """Rank in [0,1] with ties averaged."""
    n = len(vals)
    order = sorted(range(n), key=lambda i: vals[i])
    out = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and vals[order[j + 1]] == vals[order[i]]:
            j += 1
        r = ((i + j) / 2.0) / (n - 1.0) if n > 1 else 0.5
        for k in range(i, j + 1):
            out[order[k]] = r
        i = j + 1
    return out


# ==========================================================================
# fixtures / helpers
# ==========================================================================
def read_page():
    with open(PAGE, encoding="utf-8") as fh:
        return fh.read()


def last_record(path):
    prev = None
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                prev = line
    assert prev, "empty dump " + path
    return prev


def have(mod):
    try:
        __import__(mod)
        return True
    except ImportError:
        return False


need_dumps = pytest.mark.skipif(
    not os.path.isdir(DUMPS), reason="real dumps not present at " + DUMPS)
need_np = pytest.mark.skipif(not have("numpy"), reason="numpy not installed")
need_js = pytest.mark.skipif(
    not have("quickjs"),
    reason="quickjs not installed (uv pip install quickjs) -- the JS/Python "
           "cross-check is the strongest test here; do not ship without it")


_SCRIPTS = None


def inline_scripts(src=None):
    """Return the contents of every inline <script> in the page."""
    global _SCRIPTS
    if _SCRIPTS is not None and src is None:
        return _SCRIPTS
    text = src if src is not None else read_page()

    class Grab(html.parser.HTMLParser):
        def __init__(self):
            super().__init__(convert_charrefs=False)
            self.out, self._cap = [], None

        def handle_starttag(self, tag, attrs):
            if tag == "script":
                self._cap = []

        def handle_endtag(self, tag):
            if tag == "script" and self._cap is not None:
                self.out.append("".join(self._cap))
                self._cap = None

        def handle_data(self, data):
            if self._cap is not None:
                self._cap.append(data)

        def handle_entityref(self, name):
            if self._cap is not None:
                self._cap.append("&" + name + ";")

        def handle_charref(self, name):
            if self._cap is not None:
                self._cap.append("&#" + name + ";")

    g = Grab()
    g.feed(text)
    g.close()
    if src is None:
        _SCRIPTS = g.out
    return g.out


PURE_MARKER = "==FQ-HEATMAP-PURE-END=="


def pure_js():
    """The DOM-free prefix of the page's script, plus a tiny atob polyfill."""
    js = inline_scripts()[0]
    idx = js.find(PURE_MARKER)
    assert idx > 0, "the " + PURE_MARKER + " marker is missing from heatmap.html"
    # cut back to the start of the comment that carries the marker
    cut = js.rfind("/*", 0, idx)
    prefix = js[:cut]
    polyfill = r"""
    var _B64="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    function atob(s){
      s=String(s).replace(/=+$/,"");
      var out="",bits=0,acc=0,i,c;
      for(i=0;i<s.length;i++){
        c=_B64.indexOf(s.charAt(i));
        if(c<0) continue;
        acc=(acc<<6)|c; bits+=6;
        if(bits>=8){ bits-=8; out+=String.fromCharCode((acc>>bits)&255); }
      }
      return out;
    }
    """
    return polyfill + "\n" + prefix


_CTX = None


def js_ctx():
    """A QuickJS context with the page's own arithmetic loaded."""
    global _CTX
    if _CTX is not None:
        return _CTX
    import quickjs
    ctx = quickjs.Context()
    ctx.set_memory_limit(-1)
    ctx.set_time_limit(-1)
    ctx.set_max_stack_size(8 * 1024 * 1024)
    ctx.eval(pure_js())
    ctx.eval(r"""
    /* test-side helpers, built on the page's own functions */
    function _loadRec(rec, label){ return parseSample(rec, label||"t", "file"); }
    function _lutDump(which){
      var L = which==="mag" ? LUT_MAG : LUT_DIV, out=[], i;
      for(i=0;i<LUT_N;i++) out.push([L[i*3],L[i*3+1],L[i*3+2]]);
      return JSON.stringify(out);
    }
    function _identity(f){
      var a=[],l,e;
      for(l=0;l<f.L;l++){ var r=new Int32Array(f.E); for(e=0;e<f.E;e++) r[e]=e; a.push(r); }
      return a;
    }
    /* the exact colour index every cell of a panel would receive */
    function _panelIndices(f, kind, a, b){
      var p={kind:kind, frame:f, a:a, b:b};
      var pv=panelValues(p, _identity(f));
      var out=new Array(f.L*f.E), i;
      for(i=0;i<out.length;i++){
        if(pv.tier) out[i]=-pv.vals[i];                       /* tier: -K */
        else if(pv.dead && pv.dead[i]) out[i]=-1000;          /* dead cell */
        else out[i]=lutIndex(pv.vals[i], pv.lo, pv.hi);
      }
      return JSON.stringify(out);
    }
    function _panelValues(f, kind, a, b){
      var p={kind:kind, frame:f, a:a, b:b};
      var pv=panelValues(p, _identity(f));
      return JSON.stringify(Array.prototype.slice.call(pv.vals));
    }
    function _derived(f, metric){
      var d=derive(f, metric);
      return JSON.stringify({rowSum:Array.prototype.slice.call(d.rowSum),
        entropy:Array.prototype.slice.call(d.entropy),
        maxRel:Array.prototype.slice.call(d.maxRel),
        dead:d.dead, clipLo:d.clipLo, clipHi:d.clipHi, total:d.total,
        rowSumDev:d.rowSumDev, rowSumOk:d.rowSumOk, metricUsed:d.metricUsed});
    }
    function _rel(f, metric){
      return JSON.stringify(Array.prototype.slice.call(derive(f,metric).rel));
    }
    """)
    _CTX = ctx
    return ctx


def js_json(ctx, expr):
    return json.loads(ctx.eval(expr))


def js_load(ctx, raw_line, var="F", label="t"):
    """Load a raw JSONL line into the QuickJS context as a page Frame."""
    ctx.eval("var _REC = " + raw_line.strip() + ";")
    ctx.eval("var " + var + " = _loadRec(_REC, " + json.dumps(label) + ");")


# ==========================================================================
# 1. the page itself
# ==========================================================================
def test_html_parses_and_tags_balance():
    src = read_page()
    void = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link",
            "meta", "param", "source", "track", "wbr"}

    class Check(html.parser.HTMLParser):
        def __init__(self):
            super().__init__(convert_charrefs=True)
            self.stack, self.errs = [], []

        def handle_starttag(self, tag, attrs):
            if tag not in void:
                self.stack.append((tag, self.getpos()))

        def handle_endtag(self, tag):
            if tag in void:
                return
            if not self.stack:
                self.errs.append("stray </%s> at %s" % (tag, self.getpos()))
                return
            top = self.stack.pop()
            if top[0] != tag:
                self.errs.append("</%s> at %s closes <%s> opened at %s"
                                 % (tag, self.getpos(), top[0], top[1]))

    c = Check()
    c.feed(src)
    c.close()
    assert not c.errs, c.errs
    assert not c.stack, "unclosed tags: %s" % [t[0] for t in c.stack]


def test_page_is_self_contained():
    """Zero external requests: no CDN script, font, stylesheet or image."""
    src = read_page()
    offenders = []
    for m in re.finditer(r"""(?:src|href)\s*=\s*["']([^"']*)["']""", src):
        u = m.group(1)
        if not (u.startswith("#") or u.startswith("data:")):
            offenders.append("src/href -> " + u)
    if re.search(r"@import", src):
        offenders.append("@import in CSS")
    if re.search(r"<link\b", src, re.I):
        offenders.append("<link> element")
    for m in re.finditer(r"""url\(\s*["']?(?!data:)([^)"']+)""", src):
        offenders.append("css url() -> " + m.group(1))
    # a literal remote host anywhere in fetch()/XHR
    for m in re.finditer(r"""(?:fetch|open)\(\s*["'](https?://[^"']+)""", src):
        offenders.append("hardcoded fetch -> " + m.group(1))
    assert not offenders, offenders


def test_theme_tokens_are_defined_in_all_three_places():
    src = read_page()
    assert re.search(r"^:root\{", src, re.M), "bare :root palette missing"
    assert '@media (prefers-color-scheme: dark)' in src
    assert ':root:not([data-theme="light"])' in src
    assert ':root[data-theme="dark"]' in src
    assert re.search(r"body\{[^}]*background:var\(--bg\)", src), \
        "body needs an explicit background token"


def test_wide_content_scrolls_in_its_own_container():
    src = read_page()
    assert re.search(r"\.figwrap\{[^}]*overflow-x:auto", src, re.S)
    assert re.search(r"body\{[^}]*overflow-x:hidden", src, re.S)
    assert re.search(r"\.tblwrap\{[^}]*overflow-x:auto", src)


def test_required_controls_exist():
    src = read_page()
    for ident in ["endpoint", "poll", "btn-pause", "btn-reset", "btn-png",
                  "btn-mass", "metric-seg", "order-seg", "p-compare", "file",
                  "record", "cell", "fig", "overlay", "tip"]:
        assert 'id="%s"' % ident in src, "missing control id=%s" % ident


@pytest.mark.skipif(not have("esprima"), reason="esprima not installed")
def test_inline_javascript_parses():
    import esprima
    scripts = inline_scripts()
    assert scripts, "no inline <script> found"
    for i, s in enumerate(scripts):
        esprima.parseScript(s, {"tolerant": False})


def test_constants_block_matches_design():
    """The ramps in the page are the ramps DESIGN.md specifies."""
    src = read_page()
    m = re.search(r"==FQ-HEATMAP-CONSTANTS-BEGIN==(.*?)==FQ-HEATMAP-CONSTANTS-END==",
                  src, re.S)
    assert m, "constants block markers missing from heatmap.html"
    block = m.group(1)

    def hexes(var):
        mm = re.search(re.escape(var) + r"\s*=\s*(\[.*?\]|\{.*?\})", block, re.S)
        assert mm, "no %s in the constants block" % var
        return [h.upper() for h in re.findall(r"#([0-9A-Fa-f]{6})", mm.group(1))]

    assert hexes("RAMP_MAG") == [c.lstrip("#").upper() for c in RAMP_MAG]
    assert hexes("RAMP_DIV") == [c.lstrip("#").upper() for c in RAMP_DIV]
    assert hexes("TIER_COLOR") == [TIER_COLOR[k].lstrip("#").upper()
                                   for k in sorted(TIER_COLOR)]
    assert re.search(r"MAG_DOMAIN\s*=\s*4", block)
    assert re.search(r"LUT_N\s*=\s*257", block)


# ==========================================================================
# 2. colour scale properties DESIGN.md asserts
# ==========================================================================
def test_sequential_ramp_is_single_hue_and_monotone_in_lightness():
    ls = [srgb_to_lab(hex_rgb(c))[0] for c in RAMP_MAG]
    assert ls == sorted(ls, reverse=True), ls
    # the measured L* table in DESIGN.md section 3.1
    assert [round(x, 1) for x in ls] == [98.7, 94.1, 87.5, 77.4, 65.4, 54.9,
                                         40.6, 27.6, 18.1]


def test_diverging_midpoint_is_a_true_neutral():
    L, a, b = srgb_to_lab(hex_rgb(RAMP_DIV[3]))
    assert abs(a) < 0.05 and abs(b) < 0.05, (a, b)
    assert abs(L - 78.1) < 0.15, L
    ls = [round(srgb_to_lab(hex_rgb(c))[0], 1) for c in RAMP_DIV]
    # symmetric and monotone outward: no bright halo around zero
    assert ls == [38.9, 51.9, 66.0, 78.1, 66.2, 51.9, 37.9], ls
    for k in range(3):
        assert ls[k] < ls[k + 1] and ls[6 - k] < ls[5 - k]


def test_zero_lands_exactly_on_the_neutral_swatch():
    """An odd LUT is the only way v=0 maps to the specified neutral."""
    lut = make_lut(RAMP_DIV)
    for dom in (1.0, 2.0, 4.0, 8.0):
        assert lut[lut_index(0.0, -dom, dom)] == hex_rgb("#C1C1C1")


def test_tier_ramp_is_evenly_spaced_in_lightness():
    ls = [srgb_to_lab(hex_rgb(TIER_COLOR[k]))[0] for k in (2, 3, 4, 5)]
    assert ls == sorted(ls, reverse=True)
    steps = [ls[i] - ls[i + 1] for i in range(3)]
    for s in steps:
        assert 16.0 < s < 20.0, steps


@need_js
def test_js_lut_is_identical_to_the_python_lut():
    """The page's Lab interpolation and this file's agree on all 257x2 stops."""
    ctx = js_ctx()
    for which, stops in (("mag", RAMP_MAG), ("div", RAMP_DIV)):
        js = js_json(ctx, '_lutDump("%s")' % which)
        py = [list(c) for c in make_lut(stops)]
        assert js == py, "LUT drift in %s at %s" % (
            which, [i for i in range(LUT_N) if js[i] != py[i]][:6])


@need_js
def test_js_lut_index_matches_python():
    ctx = js_ctx()
    vals = [-9, -4, -4.0001, -3.5, -1, -1e-9, 0, 1e-9, 0.5, 2.5, 4, 4.0001, 9]
    for v in vals:
        got = ctx.eval("lutIndex(%r, -4, 4)" % v)
        assert got == lut_index(v, -4, 4), (v, got)


# ==========================================================================
# 3. normalisation, on the real dumps
# ==========================================================================
@need_dumps
@need_np
def test_normalisation_reproduces_the_design_measurements():
    import numpy as np
    rec = json.loads(last_record(os.path.join(DUMPS, "stats-code-axis.jsonl")))
    c = np.array(rec["count"], dtype=np.float64)
    L, E = c.shape
    assert (L, E) == (75, 256)
    rows = c.sum(axis=1)
    # DESIGN.md section 5.1: top-k routing makes every per-layer total identical
    assert float(np.max(np.abs(rows - rows[0])) / rows[0]) < 1e-9
    with np.errstate(divide="ignore"):
        v = np.log2(E * c / rows[:, None])
    bands = [0] * 9
    for x in v.ravel():
        bands[band9(float(x))] += 1
    assert bands == GOLDEN_BANDS_CODE_AXIS
    assert sum(bands) == L * E


@need_dumps
@need_np
def test_clipping_is_counted_at_both_edges():
    """DESIGN.md quotes upper-tail coverage only; the LOW tail is the big one.

    Measured here: 752 cells (3.92%) of the code-axis dump fall below 1/16x
    uniform and are clipped at the floor, against 25 (0.13%) above 16x. The
    page must report both, which is why the caption prints two numbers.
    """
    import numpy as np
    rec = json.loads(last_record(os.path.join(DUMPS, "stats-code-axis.jsonl")))
    c = np.array(rec["count"], dtype=np.float64)
    E = c.shape[1]
    with np.errstate(divide="ignore"):
        v = np.log2(E * c / c.sum(axis=1)[:, None])
    lo = int((v < -MAG_DOMAIN).sum())
    hi = int((v > MAG_DOMAIN).sum())
    assert (lo, hi) == (752, 25), (lo, hi)
    assert lo > 20 * hi, "the low tail dominates; a one-sided coverage claim hides it"


@need_dumps
@need_js
@need_np
def test_js_derive_matches_numpy_cell_for_cell():
    """The page's derive() against an independent NumPy implementation."""
    import numpy as np
    ctx = js_ctx()
    js_load(ctx, last_record(os.path.join(DUMPS, "stats-code-axis.jsonl")), "F")
    d = js_json(ctx, '_derived(F, "count")')
    rel_js = np.array(js_json(ctx, '_rel(F, "count")'), dtype=np.float64)

    rec = json.loads(last_record(os.path.join(DUMPS, "stats-code-axis.jsonl")))
    c = np.array(rec["count"], dtype=np.float64)
    L, E = c.shape
    rows = c.sum(axis=1)
    rel = (E * c / rows[:, None]).ravel()
    with np.errstate(divide="ignore"):
        v = np.log2(rel)

    assert np.allclose(rel_js, rel, rtol=1e-12, atol=0)
    assert np.allclose(np.array(d["rowSum"]), rows, rtol=1e-12)
    assert d["metricUsed"] == "count"
    assert d["rowSumOk"] is True
    assert d["dead"] == int((c == 0).sum())
    assert d["clipLo"] == int((v < -MAG_DOMAIN).sum())
    assert d["clipHi"] == int((v > MAG_DOMAIN).sum())

    share = c / rows[:, None]
    p = np.where(share > 0, share, 1.0)
    ent = -(np.where(share > 0, share * np.log(p), 0.0)).sum(axis=1) / math.log(E)
    assert np.allclose(np.array(d["entropy"]), ent, rtol=1e-12)
    assert np.allclose(np.array(d["maxRel"]), (E * share).max(axis=1), rtol=1e-12)


# ==========================================================================
# 4. THE colour-bucket test: every one of the 19,200 cells
# ==========================================================================
@need_dumps
@need_js
@need_np
def test_colour_bucket_assignment_on_a_real_dump():
    """Feed a real dump through the page's renderer maths and assert the
    colour index -- and therefore the RGB -- of every cell."""
    import numpy as np
    ctx = js_ctx()
    raw = last_record(os.path.join(DUMPS, "stats-code-axis.jsonl"))
    js_load(ctx, raw, "F")
    idx_js = js_json(ctx, '_panelIndices(F, "mag")')

    rec = json.loads(raw)
    c = np.array(rec["count"], dtype=np.float64)
    L, E = c.shape
    with np.errstate(divide="ignore"):
        v = np.log2(E * c / c.sum(axis=1)[:, None]).ravel()
    lut = make_lut(RAMP_MAG)

    idx_py = []
    for x in v:
        if not math.isfinite(x) and x < 0:
            idx_py.append(-1000)                       # dead cell, off-ramp
        else:
            idx_py.append(lut_index(max(-MAG_DOMAIN, min(MAG_DOMAIN, float(x))),
                                    -MAG_DOMAIN, MAG_DOMAIN))
    assert len(idx_js) == L * E == 19200
    bad = [i for i in range(L * E) if idx_js[i] != idx_py[i]]
    assert not bad, "colour index differs at %d cells, first: %s" % (
        len(bad), [(i // E, i % E, idx_js[i], idx_py[i]) for i in bad[:5]])

    # golden: pin the whole field, so any drift in ramp, domain or rounding fails
    digest = hashlib.sha256(
        b",".join(str(i).encode() for i in idx_js)).hexdigest()[:16]
    assert digest == "d890132bcb32bd65", digest

    # and a few cells by hand, with their RGB
    for (row, e, want_idx, want_rgb) in [
        (0, 0, 123, (163, 159, 203)),
        (0, 1, 86, (197, 198, 225)),
        (0, 3, 195, (104, 77, 161)),
        (10, 100, 84, (199, 200, 226)),
        (37, 128, 124, (162, 158, 202)),
        (74, 255, 120, (165, 163, 205)),
    ]:
        got = idx_js[row * E + e]
        assert got == want_idx, ("cell L%d E%d" % (row, e), got, want_idx)
        assert lut[got] == want_rgb


@need_dumps
@need_js
def test_domain_is_fixed_never_autoscaled():
    """Two dumps 71x apart in total traffic must share one colour domain."""
    ctx = js_ctx()
    js_load(ctx, last_record(os.path.join(DUMPS, "stats-synthetic.jsonl")), "FS")
    js_load(ctx, last_record(os.path.join(DUMPS, "stats.jsonl")), "FF")
    for var in ("FS", "FF"):
        lo = ctx.eval('panelValues({kind:"mag",frame:%s}, _identity(%s)).lo' % (var, var))
        hi = ctx.eval('panelValues({kind:"mag",frame:%s}, _identity(%s)).hi' % (var, var))
        assert (lo, hi) == (-4.0, 4.0), (var, lo, hi)


@need_dumps
@need_js
def test_dead_cells_are_off_ramp_not_the_palest_purple():
    """The synthetic dump has 42 never-routed cells; they must not be drawn as
    'rarely routed'."""
    ctx = js_ctx()
    js_load(ctx, last_record(os.path.join(DUMPS, "stats-synthetic.jsonl")), "FS")
    idx = js_json(ctx, '_panelIndices(FS, "mag")')
    assert idx.count(-1000) == 42, idx.count(-1000)
    dead_rgb = js_json(ctx, "JSON.stringify(DEAD_RGB)")
    assert dead_rgb == [255, 255, 255]
    assert tuple(dead_rgb) != make_lut(RAMP_MAG)[0]      # != #FCFBFD


# ==========================================================================
# 5. compare mode
# ==========================================================================
@need_dumps
@need_js
@need_np
def test_compare_delta_conserves_within_every_layer():
    """DESIGN.md section 5.4: rows of delta sum to zero, so the panel reads as
    traffic moving between experts. No other candidate metric does this."""
    import numpy as np
    ctx = js_ctx()
    js_load(ctx, last_record(os.path.join(DUMPS, "stats-code-axis.jsonl")), "A")
    js_load(ctx, last_record(os.path.join(DUMPS, "stats-synthetic.jsonl")), "B")
    ctx.eval('S.cmpMetric="delta"; S.cmpDomain=4;')
    vals = np.array(js_json(ctx, '_panelValues(B, "diff", A, B)'), dtype=np.float64)
    vals = vals.reshape(75, 256)
    assert float(np.max(np.abs(vals.sum(axis=1)))) < 1e-12

    ra = np.array(json.loads(last_record(os.path.join(DUMPS, "stats-code-axis.jsonl")))["count"])
    rb = np.array(json.loads(last_record(os.path.join(DUMPS, "stats-synthetic.jsonl")))["count"])
    E = ra.shape[1]
    delta = (E * rb / rb.sum(axis=1)[:, None]) - (E * ra / ra.sum(axis=1)[:, None])
    assert np.allclose(vals, delta, rtol=1e-11, atol=1e-12)


@need_dumps
@need_js
def test_symlog_is_offered_but_does_not_conserve():
    import numpy as np
    ctx = js_ctx()
    js_load(ctx, last_record(os.path.join(DUMPS, "stats-code-axis.jsonl")), "A")
    js_load(ctx, last_record(os.path.join(DUMPS, "stats-synthetic.jsonl")), "B")
    ctx.eval('S.cmpMetric="symlog";')
    vals = np.array(js_json(ctx, '_panelValues(B, "diff", A, B)')).reshape(75, 256)
    ctx.eval('S.cmpMetric="delta";')
    assert float(np.max(np.abs(vals.sum(axis=1)))) > 1e-6, \
        "symlog must NOT conserve -- the page warns about exactly this"


@need_js
def test_compare_zero_difference_renders_neutral_gray():
    """A vs A is not 'no data': it is a fully neutral field."""
    ctx = js_ctx()
    rec = json.dumps({"layers": [3, 4], "tier_of": [[3, 3], [3, 3]],
                      "count": [[1.0, 3.0], [2.0, 2.0]],
                      "mass": [[1.0, 3.0], [2.0, 2.0]]})
    ctx.eval("var R=" + rec + "; var X=_loadRec(R,'x');")
    idx = js_json(ctx, '_panelIndices(X, "diff", X, X)')
    lut = make_lut(RAMP_DIV)
    assert set(idx) == {128}
    assert lut[128] == hex_rgb("#C1C1C1")


# ==========================================================================
# 6. mismatch panel
# ==========================================================================
def test_rank01_averages_ties():
    assert rank01([5, 5, 5, 5]) == [0.5, 0.5, 0.5, 0.5]
    assert rank01([1, 2, 3]) == [0.0, 0.5, 1.0]
    assert rank01([9, 1, 1, 9]) == [pytest.approx(5 / 6), pytest.approx(1 / 6),
                                    pytest.approx(1 / 6), pytest.approx(5 / 6)]


@need_js
def test_js_rank01_matches_python():
    ctx = js_ctx()
    vals = [3.0, 1.0, 1.0, 7.0, 7.0, 7.0, 0.5]
    got = js_json(ctx, """(function(){
      var a=new Float64Array(%s), o=new Float64Array(a.length);
      rank01(a,0,a.length,o);
      return JSON.stringify(Array.prototype.slice.call(o));
    })()""" % json.dumps(vals))
    assert got == pytest.approx(rank01(vals))


@need_dumps
@need_js
def test_uniform_tier_collapses_mismatch_to_heat_rank_minus_half():
    """Every archived run is K3-uniform, so ztier is 0.5 everywhere and the
    mismatch panel carries NO tier information. The page must say so."""
    import numpy as np
    ctx = js_ctx()
    js_load(ctx, last_record(os.path.join(DUMPS, "stats-code-axis.jsonl")), "F")
    mm = np.array(js_json(ctx, '_panelValues(F, "mismatch")')).reshape(75, 256)
    rel = np.array(js_json(ctx, '_rel(F, "count")')).reshape(75, 256)
    for l in range(0, 75, 17):
        zh = np.array(rank01(list(rel[l])))
        assert np.allclose(mm[l], zh - 0.5, atol=1e-12)
    assert "carries no information" in read_page()


@need_js
def test_mismatch_signs_point_the_right_way():
    """positive = hot but under-tiered (promote); negative = cold, over-tiered."""
    ctx = js_ctx()
    rec = json.dumps({"layers": [3],
                      "count": [[100.0, 1.0]],
                      "tier_of": [[2, 5]]})     # hottest expert on the LOWEST K
    ctx.eval("var R2=" + rec + "; var Y=_loadRec(R2,'y');")
    mm = js_json(ctx, '_panelValues(Y, "mismatch")')
    assert mm[0] > 0, "hot expert at K2 must read as under-tiered"
    assert mm[1] < 0, "cold expert at K5 must read as over-tiered"


# ==========================================================================
# 7. wire format
# ==========================================================================
def _bf16_bytes(vals, round_to_nearest=True):
    """Encode float32 -> bf16, little-endian.

    bf16 is the top half of a float32. torch's cast rounds to nearest even,
    which is what a real endpoint would put on the wire and what gives the
    0.389% worst-case quantum the specs quote; plain truncation doubles that
    to 2**-7. Both are exercised.
    """
    out = bytearray()
    for v in vals:
        u = struct.unpack("<I", struct.pack("<f", float(v)))[0]
        if round_to_nearest:
            u = (u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFFFFFF
        out += struct.pack("<H", (u >> 16) & 0xFFFF)
    return bytes(out)


@need_dumps
@need_js
def test_bf16_decode_roundtrips_within_the_measured_quantum():
    import base64
    ctx = js_ctx()
    rec = json.loads(last_record(os.path.join(DUMPS, "stats.jsonl")))
    row = rec["count"][0]
    blob = base64.b64encode(_bf16_bytes(row)).decode()
    got = js_json(ctx, "JSON.stringify(Array.prototype.slice.call(decodeBf16(%s, %d)))"
                  % (json.dumps(blob), len(row)))
    worst = max(abs(g - r) / abs(r) for g, r in zip(got, row) if r != 0)
    assert worst < 0.005, worst          # DESIGN/ENDPOINT-SPEC measured 0.389%
    assert worst > 1e-4, "suspiciously exact -- is this really bf16?"

    # truncation instead of round-to-nearest doubles the quantum; the decoder
    # is the same shift either way, which is the point.
    blob_t = base64.b64encode(_bf16_bytes(row, round_to_nearest=False)).decode()
    got_t = js_json(ctx, "JSON.stringify(Array.prototype.slice.call(decodeBf16(%s, %d)))"
                    % (json.dumps(blob_t), len(row)))
    worst_t = max(abs(g - r) / abs(r) for g, r in zip(got_t, row) if r != 0)
    assert worst < worst_t < 2 ** -7 * 1.01, (worst, worst_t)


@need_js
def test_truncated_array_is_refused_not_silently_shifted():
    """A short base64 blob must never render as a heatmap shifted by a cell."""
    import base64
    ctx = js_ctx()
    blob = base64.b64encode(_bf16_bytes([1.0] * 19199)).decode()
    with pytest.raises(Exception) as ei:
        ctx.eval("decodeBf16(%s, 19200)" % json.dumps(blob))
    assert "19199" in str(ei.value) or "expected" in str(ei.value)


@need_js
def test_envelope_and_raw_dump_shapes_both_parse():
    import base64
    ctx = js_ctx()
    count = [1.0, 2.0, 4.0, 8.0, 3.0, 5.0]
    env = {
        "schema": "fq-heatmap/1", "layers": [3, 4, 5], "num_layers": 3,
        "num_experts": 2, "cells": 6, "mass_is_real": False,
        "encoding": {"layout": "layer-major", "count": "bf16", "tier": "u8"},
        "count": base64.b64encode(_bf16_bytes(count)).decode(),
        "mass": None,
        "tier": base64.b64encode(bytes([3, 3, 3, 5, 5, 5])).decode(),
        "step": 100, "interval": 1,
    }
    ctx.eval("var EV=" + json.dumps(env) + "; var G=_loadRec(EV,'env');")
    assert ctx.eval("G.L") == 3 and ctx.eval("G.E") == 2
    assert ctx.eval("G.massIsReal") is False
    assert ctx.eval("G.mass") is None
    assert js_json(ctx, "JSON.stringify(Array.prototype.slice.call(G.tier))") == \
        [3, 3, 3, 5, 5, 5]
    assert js_json(ctx, "JSON.stringify(Array.prototype.slice.call(G.layers))") == [3, 4, 5]


# ==========================================================================
# 8. the hazards DESIGN.md section 7 lists
# ==========================================================================
@need_dumps
@need_js
def test_mass_is_real_absent_is_unknown_not_false():
    """DESIGN.md hazard 4. All four archived dumps predate the flag."""
    ctx = js_ctx()
    for i, name in enumerate(DUMP_FILES):
        path = os.path.join(DUMPS, name)
        if not os.path.exists(path):
            continue
        raw = last_record(path)
        assert '"mass_is_real"' not in raw, name
        js_load(ctx, raw, "M%d" % i, name)
        assert ctx.eval("M%d.massIsReal" % i) is None, \
            "%s: absent mass_is_real must load as null (unknown), never false" % name
        warns = js_json(ctx, "JSON.stringify(M%d.warnings)" % i)
        assert any("mass_is_real is absent" in w for w in warns), warns


@need_js
def test_mass_is_real_is_never_inferred_from_array_equality():
    """A uniform router legitimately makes count == mass with real gate mass."""
    ctx = js_ctx()
    rec = {"layers": [3], "count": [[1.0, 1.0]], "mass": [[1.0, 1.0]],
           "tier_of": [[3, 3]], "mass_is_real": True}
    ctx.eval("var U=" + json.dumps(rec) + "; var Z=_loadRec(U,'u');")
    assert ctx.eval("Z.massIsReal") is True
    assert ctx.eval('derive(Z,"mass").metricUsed') == "mass"


@need_js
def test_tier_row_count_mismatch_is_rejected():
    """DESIGN.md hazard 2: policies carry 76 layer keys (3-78, incl. MTP);
    stats carry 75. A positional join is off by one and must not be silent."""
    ctx = js_ctx()
    rec = {"layers": [3, 4], "count": [[1.0, 2.0], [3.0, 4.0]],
           "tier_of": [[3, 3], [3, 3], [3, 3]]}       # 3 tier rows, 2 layers
    ctx.eval("var BAD=" + json.dumps(rec) + ";")
    with pytest.raises(Exception) as ei:
        ctx.eval("_loadRec(BAD,'bad')")
    assert "tier_of" in str(ei.value)


@need_js
def test_layer_ids_come_from_the_data_not_from_row_plus_three():
    ctx = js_ctx()
    rec = {"layers": [5, 9, 40], "count": [[1.0], [2.0], [3.0]],
           "tier_of": [[3], [4], [5]]}
    ctx.eval("var LR=" + json.dumps(rec) + "; var W=_loadRec(LR,'w');")
    assert js_json(ctx, "JSON.stringify(Array.prototype.slice.call(W.layers))") == [5, 9, 40]
    assert ctx.eval("W.L") == 3


@need_js
def test_row_sum_violation_is_reported_not_hidden():
    """A shared/always-on expert or a varying top_k breaks the invariant that
    makes per-layer share equal per-run share. It must be surfaced."""
    ctx = js_ctx()
    rec = {"layers": [3, 4], "count": [[1.0, 1.0], [5.0, 5.0]],
           "tier_of": [[3, 3], [3, 3]]}
    ctx.eval("var V=" + json.dumps(rec) + "; var Q=_loadRec(V,'q');")
    d = js_json(ctx, '_derived(Q, "count")')
    assert d["rowSumOk"] is False
    assert d["rowSumDev"] > 1.0
    assert "VIOLATED" in read_page()


@need_dumps
@need_js
def test_one_shared_permutation_across_panels():
    """DESIGN.md hazard 1: per-panel sorting is fitting on the test set."""
    ctx = js_ctx()
    js_load(ctx, last_record(os.path.join(DUMPS, "stats-code-axis.jsonl")), "PA")
    js_load(ctx, last_record(os.path.join(DUMPS, "stats-synthetic.jsonl")), "PB")
    ctx.eval('S.order="sorted"; S.orderRef="pooled"; S.metric="count";')
    pooled = ctx.eval("buildPerm([PA, PB]).hash")
    only_a = ctx.eval("buildPerm([PA]).hash")
    only_b = ctx.eval("buildPerm([PB]).hash")
    assert pooled and pooled != "native"
    assert len({pooled, only_a, only_b}) == 3, \
        "the reference must actually change the permutation, else 'shared' is vacuous"
    assert ctx.eval("buildPerm([PA, PB]).hash") == pooled     # deterministic
    ctx.eval('S.order="native";')
    assert ctx.eval("buildPerm([PA]).hash") == "native"


@need_dumps
@need_js
def test_native_order_is_the_identity():
    ctx = js_ctx()
    js_load(ctx, last_record(os.path.join(DUMPS, "stats-code-axis.jsonl")), "NP")
    ctx.eval('S.order="native";')
    p = js_json(ctx, "JSON.stringify(Array.prototype.slice.call(buildPerm([NP]).perm[7]))")
    assert p == list(range(256))


@need_dumps
@need_js
def test_all_four_archived_dumps_load():
    """Shape only for the INVALID one: its contents are from a broken replay."""
    ctx = js_ctx()
    for i, name in enumerate(DUMP_FILES):
        path = os.path.join(DUMPS, name)
        if not os.path.exists(path):
            pytest.skip("missing " + name)
        js_load(ctx, last_record(path), "D%d" % i, name)
        assert ctx.eval("D%d.L" % i) == 75
        assert ctx.eval("D%d.E" % i) == 256
        assert ctx.eval("D%d.count.length" % i) == 19200
        assert ctx.eval("D%d.tier.length" % i) == 19200
        assert js_json(ctx, "JSON.stringify([D%d.layers[0], D%d.layers[74]])" % (i, i)) \
            == [3, 77]


# ==========================================================================
# 9. the render path, executed headlessly
#
# No browser here, so a canvas is stubbed and the page's whole script -- not
# just its arithmetic -- is run in QuickJS. This does not prove the figure
# LOOKS right (nobody has seen it rendered), but it does prove the render path
# executes, the blit is an exact integer scale, and the bytes written into the
# ImageData are the colours the reference says they should be.
# ==========================================================================
DOM_STUB = r"""
var LOG = { fillRect:0, strokeRect:0, fillText:0, drawImage:[], putImageData:[],
            downloads:[], fetches:[] };
var _B64="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
function atob(s){
  s=String(s).replace(/=+$/,"");
  var out="",bits=0,acc=0,i,c;
  for(i=0;i<s.length;i++){
    c=_B64.indexOf(s.charAt(i));
    if(c<0) continue;
    acc=(acc<<6)|c; bits+=6;
    if(bits>=8){ bits-=8; out+=String.fromCharCode((acc>>bits)&255); }
  }
  return out;
}
function Ctx2D(owner){
  this._o=owner; this.font=""; this.fillStyle=""; this.strokeStyle="";
  this.lineWidth=1; this.textAlign=""; this.textBaseline="";
  this.globalAlpha=1; this.imageSmoothingEnabled=true;
}
Ctx2D.prototype.setTransform=function(){};
Ctx2D.prototype.clearRect=function(){};
Ctx2D.prototype.fillRect=function(){ LOG.fillRect++; };
Ctx2D.prototype.strokeRect=function(){ LOG.strokeRect++; };
Ctx2D.prototype.fillText=function(){ LOG.fillText++; };
Ctx2D.prototype.measureText=function(s){ return {width:String(s).length*6.1}; };
Ctx2D.prototype.beginPath=function(){};
Ctx2D.prototype.moveTo=function(){};
Ctx2D.prototype.lineTo=function(){};
Ctx2D.prototype.stroke=function(){};
Ctx2D.prototype.createImageData=function(w,h){
  return {width:w, height:h, data:new Uint8ClampedArray(w*h*4)};
};
Ctx2D.prototype.putImageData=function(id){
  this._o._img={width:id.width, height:id.height, data:id.data};
  LOG.putImageData.push({w:id.width, h:id.height});
};
Ctx2D.prototype.drawImage=function(src,sx,sy,sw,sh,dx,dy,dw,dh){
  LOG.drawImage.push({sw:sw, sh:sh, dx:dx, dy:dy, dw:dw, dh:dh,
                      smooth:this.imageSmoothingEnabled,
                      img: src && src._img ? src._img : null});
};
function El(id, tag){
  this.id=id; this.tagName=(tag||"div").toUpperCase();
  this.value=""; this.checked=false; this.disabled=false;
  this.textContent=""; this.innerHTML=""; this.href=""; this.download="";
  this.width=0; this.height=0;
  this.style={}; this.files=[]; this._attrs={}; this._listeners={};
  this._children=[]; this._ctx=null; this._img=null;
  this.clientWidth=1200; this.offsetWidth=240; this.offsetHeight=160;
  var self=this;
  this.classList={
    add:function(c){ self._attrs["class."+c]=1; },
    remove:function(c){ delete self._attrs["class."+c]; },
    toggle:function(c,on){ if(on) self._attrs["class."+c]=1;
                           else delete self._attrs["class."+c]; },
    contains:function(c){ return !!self._attrs["class."+c]; }
  };
}
El.prototype.addEventListener=function(t,f){ (this._listeners[t]=this._listeners[t]||[]).push(f); };
El.prototype.dispatch=function(t,ev){
  var ls=this._listeners[t]||[], i;
  for(i=0;i<ls.length;i++) ls[i].call(this, ev||{target:this});
};
El.prototype.getAttribute=function(k){ return (k in this._attrs)?this._attrs[k]:null; };
El.prototype.setAttribute=function(k,v){ this._attrs[k]=v; };
El.prototype.removeAttribute=function(k){ delete this._attrs[k]; };
El.prototype.appendChild=function(c){ this._children.push(c); return c; };
El.prototype.querySelectorAll=function(){ return []; };
El.prototype.closest=function(){ return null; };
El.prototype.click=function(){ LOG.downloads.push({name:this.download,
                                 href:String(this.href).slice(0,32)}); };
El.prototype.getBoundingClientRect=function(){ return {left:0,top:0,width:this.width,height:this.height}; };
El.prototype.getContext=function(){ if(!this._ctx) this._ctx=new Ctx2D(this); return this._ctx; };
El.prototype.toDataURL=function(){ return "data:image/png;base64,iVBORw0KGgo="; };
var _els={};
var document={
  documentElement:new El("html","html"),
  getElementById:function(id){ if(!_els[id]) _els[id]=new El(id); return _els[id]; },
  createElement:function(tag){ return new El("_"+tag, tag); },
  addEventListener:function(){}
};
var window={ devicePixelRatio:2, innerWidth:1440, innerHeight:900,
             addEventListener:function(){}, confirm:function(){ return false; } };
function getComputedStyle(){ return {getPropertyValue:function(){ return "#FFFFFF"; }}; }
var localStorage={ _m:{}, getItem:function(k){ return this._m[k]||null; },
                   setItem:function(k,v){ this._m[k]=String(v); } };
function setTimeout(){ return 1; }
function clearTimeout(){}
function fetch(url){ LOG.fetches.push(String(url));
                     return Promise.reject(new Error("stub: offline")); }
function Blob(parts,o){ this.size=String(parts[0]).length; this.type=o?o.type:""; }
var URL={ createObjectURL:function(){ return "blob:stub"; },
          revokeObjectURL:function(){} };
"""

_PAGE_CTX = None


def page_ctx():
    """A QuickJS context running the ENTIRE page script against a stub DOM."""
    global _PAGE_CTX
    if _PAGE_CTX is not None:
        return _PAGE_CTX
    import quickjs
    ctx = quickjs.Context()
    ctx.set_memory_limit(-1)
    ctx.set_time_limit(-1)
    ctx.set_max_stack_size(16 * 1024 * 1024)
    ctx.eval(DOM_STUB)
    ctx.eval(inline_scripts()[0])
    # let the boot poll() settle against the offline stub, then start clean
    for _ in range(500):
        if not ctx.execute_pending_job():
            break
    ctx.eval("S.inflight=false; S.fails=0; LOG.fetches=[];")
    _PAGE_CTX = ctx
    return ctx


def load_offline(ctx, name, var="REC", label="t"):
    raw = last_record(os.path.join(DUMPS, name))
    ctx.eval("var %s = %s;" % (var, raw.strip()))
    ctx.eval("S.offline = parseSample(%s, %s, 'file');" % (var, json.dumps(label)))


@need_js
def test_full_script_boots_against_a_stub_dom():
    """Top-level init of the whole page runs without throwing."""
    ctx = page_ctx()
    assert ctx.eval("typeof render") == "function"
    assert ctx.eval("typeof LAYOUT") == "object"
    assert ctx.eval("cv.width") > 0, "canvas was never sized"


@need_dumps
@need_js
def test_render_executes_in_every_panel_configuration():
    ctx = page_ctx()
    load_offline(ctx, "stats-code-axis.jsonl", "R1", "code-axis")
    configs = [
        ('S.tierStrip=true;S.mismatch=false;S.compare=false;S.stack=false;'
         'S.order="native";S.cell="3";S.marginals=true;S.flagDead=false;'
         'S.tierNative=false;', 2),
        ('S.mismatch=true;', 3),
        ('S.order="sorted";S.orderRef="pooled";', 3),
        ('S.marginals=false;', 3),
        ('S.marginals=true;S.flagDead=true;S.tierNative=true;', 3),
        ('S.cell="6";', 3),
        ('S.cell="fit";S.tierStrip=false;S.mismatch=false;', 1),
    ]
    for setup, want_panels in configs:
        ctx.eval("LOG.drawImage=[];")
        ctx.eval(setup)
        ctx.eval("render();")
        got = ctx.eval("LOG.drawImage.length")
        assert got == want_panels, (setup, got, want_panels)
        assert ctx.eval("cv.width") > 0 and ctx.eval("cv.height") > 0


@need_dumps
@need_js
def test_blit_is_an_exact_integer_scale_with_smoothing_off():
    """DESIGN.md section 4.1, non-negotiable: fractional cells make the browser
    resample, and resampling a heatmap averages neighbouring experts into
    colours that correspond to no data point."""
    ctx = page_ctx()
    load_offline(ctx, "stats-code-axis.jsonl", "R2", "code-axis")
    for cell in ("2", "3", "4", "6", "8", "12", "fit"):
        ctx.eval('LOG.drawImage=[]; S.cell=%s; S.tierStrip=true; S.mismatch=false;'
                 'S.compare=false; S.stack=false; render();' % json.dumps(cell))
        blits = js_json(ctx, "JSON.stringify(LOG.drawImage.map(function(d){"
                             "return [d.sw,d.sh,d.dw,d.dh,d.smooth];}))")
        assert blits
        for sw, sh, dw, dh, smooth in blits:
            assert smooth is False, "imageSmoothingEnabled must be off for the blit"
            assert (sw, sh) == (256, 75), (sw, sh)
            assert dw % sw == 0 and dh % sh == 0, (cell, sw, sh, dw, dh)
            assert dw // sw == dh // sh, "non-square cells"
        dpr = ctx.eval("DPR")
        assert dpr == int(dpr) and dpr >= 1


@need_dumps
@need_js
def test_rendered_pixels_are_the_reference_colours():
    """Read back the bytes the page actually wrote into the ImageData."""
    ctx = page_ctx()
    load_offline(ctx, "stats-code-axis.jsonl", "R3", "code-axis")
    ctx.eval('S.cell="3"; S.order="native"; S.metric="count"; S.stack=false;'
             'S.compare=false; S.mismatch=false; S.tierStrip=true;'
             'LOG.drawImage=[]; render();')
    px = js_json(ctx, """(function(){
      var d=LOG.drawImage[0], out=[], pick=%s, k;
      for(k=0;k<pick.length;k++){
        var o=(pick[k][0]*d.sw+pick[k][1])*4;
        out.push([d.img.data[o],d.img.data[o+1],d.img.data[o+2],d.img.data[o+3]]);
      }
      return JSON.stringify(out);
    })()""" % json.dumps([[0, 0], [0, 1], [0, 3], [10, 100], [37, 128], [74, 255]]))
    want = [(163, 159, 203), (197, 198, 225), (104, 77, 161),
            (199, 200, 226), (162, 158, 202), (165, 163, 205)]
    for got, exp in zip(px, want):
        assert tuple(got[:3]) == exp and got[3] == 255, (got, exp)

    # the tier strip renders the fixed K-value colour, not a frequency-ranked one
    tier_px = js_json(ctx, """(function(){
      var d=LOG.drawImage[1];
      return JSON.stringify([d.img.data[0],d.img.data[1],d.img.data[2]]);
    })()""")
    assert tuple(tier_px) == hex_rgb(TIER_COLOR[3]), tier_px


@need_dumps
@need_js
def test_hover_resolves_the_right_layer_and_expert():
    ctx = page_ctx()
    load_offline(ctx, "stats-code-axis.jsonl", "R4", "code-axis")
    ctx.eval('S.cell="3"; S.order="native"; S.stack=false; S.compare=false; render();')
    h = js_json(ctx, """(function(){
      var p=LAYOUT.panels[0];
      var h=hitTest(p.x+3*7+1, p.y+3*11+1);
      return JSON.stringify(h?{row:h.row,col:h.col,e:h.e,kind:h.p.kind}:null);
    })()""")
    assert h == {"row": 11, "col": 7, "e": 7, "kind": "mag"}
    ctx.eval("""(function(){
      var p=LAYOUT.panels[0], h=hitTest(p.x+3*7+1, p.y+3*11+1);
      showTip(h,100,100); renderLayerTable(h.p.frame,h.row);
    })()""")
    tip = ctx.eval("document.getElementById('tip').innerHTML")
    assert "L14" in tip and "E7" in tip, tip[:200]        # layers[11] == 14, not 11
    assert "K3" in tip
    assert ctx.eval("document.getElementById('tbl-layer').innerHTML").startswith("<thead>")


@need_dumps
@need_js
def test_export_paths_produce_downloads():
    ctx = page_ctx()
    load_offline(ctx, "stats-code-axis.jsonl", "R5", "code-axis")
    ctx.eval('S.cell="3"; render(); LOG.downloads=[];')
    ctx.eval("document.getElementById('btn-png').dispatch('click',{});")
    ctx.eval("document.getElementById('btn-csv').dispatch('click',{});")
    ctx.eval("document.getElementById('btn-perm').dispatch('click',{});")
    got = js_json(ctx, "JSON.stringify(LOG.downloads)")
    assert len(got) == 3
    assert got[0]["name"].endswith(".png") and got[0]["href"].startswith("data:image/png")
    assert got[1]["name"].endswith(".csv")
    assert got[2]["name"].endswith(".json")
    # the PNG is rendered on a light ground, then the view is restored
    assert ctx.eval("S.exportLight") is False


@need_dumps
@need_js
def test_canvas_carries_a_text_summary():
    """Canvas is opaque to screen readers; do not pretend otherwise."""
    ctx = page_ctx()
    load_offline(ctx, "stats-code-axis.jsonl", "R6", "code-axis")
    ctx.eval("render();")
    label = ctx.eval("cv.getAttribute('aria-label')")
    assert "75 layers by 256 experts" in label
    assert "27.9" in label                 # max share x uniform, DESIGN appendix
    assert "777" in label                  # 752 low + 25 high clipped
    assert "CSV" in label


@need_dumps
@need_js
def test_mass_gate_never_shows_count_under_a_mass_label():
    ctx = page_ctx()
    load_offline(ctx, "stats-code-axis.jsonl", "R7", "code-axis")
    ctx.eval("render();")
    # archived dumps: flag absent -> selectable but explicitly UNKNOWN
    assert ctx.eval("S.offline.massIsReal") is None
    assert ctx.eval("document.getElementById('btn-mass').disabled") is False
    assert "unknown" in ctx.eval("document.getElementById('mass-note').textContent")

    # flag false -> disabled, labelled, and the metric falls back to count
    ctx.eval("S.offline.massIsReal=false; S.offline.mass=null; S.metric='mass'; render();")
    assert ctx.eval("S.metric") == "count"
    assert ctx.eval("document.getElementById('btn-mass').disabled") is True
    assert "not recorded" in ctx.eval("document.getElementById('mass-note').textContent")
    ctx.eval("S.metric='count';")


@need_dumps
@need_js
def test_unreachable_endpoint_keeps_the_last_frame():
    """Show the error, keep the last frame, keep retrying -- never blank."""
    ctx = page_ctx()
    load_offline(ctx, "stats-code-axis.jsonl", "R8", "code-axis")
    ctx.eval("S.live=parseSample(R8,'live','live'); S.offline=null;"
             "S.fails=0; S.inflight=false; render();")
    ctx.eval("var _done=false; poll().then(function(){_done=true;});")
    for _ in range(500):
        if not ctx.execute_pending_job():
            break
    assert ctx.eval("S.fails") == 1
    assert ctx.eval("S.live") is not None, "the last frame must survive a failed poll"
    assert ctx.eval("document.getElementById('figwrap').classList.contains('stale')") is True
    banners = ctx.eval("document.getElementById('banners').innerHTML")
    assert "unreachable" in banners and "keeping the last frame" in banners
    assert "CORS" in banners, "a file:// page hitting a serve fails on CORS first"
    assert ctx.eval("backoff()") > ctx.eval("S.pollMs"), "backoff must grow"
    ctx.eval("LOG.drawImage=[]; render();")
    assert ctx.eval("LOG.drawImage.length") > 0, "the figure went blank on error"
    ctx.eval("S.fails=0; S.inflight=false;")


@need_dumps
@need_js
def test_compare_statistics_reproduce_the_design_measurements():
    """DESIGN.md section 5.4 measured code vs synthetic: Spearman median 0.37,
    top-32 overlap median 28%. Recomputed here by the page's own code."""
    ctx = page_ctx()
    raw_a = last_record(os.path.join(DUMPS, "stats-code-axis.jsonl"))
    raw_b = last_record(os.path.join(DUMPS, "stats-synthetic.jsonl"))
    ctx.eval("var CA=" + raw_a.strip() + "; var CB=" + raw_b.strip() + ";")
    ctx.eval("""
      S.slots.A=snapshot(parseSample(CA,'code-axis','file'),'code-axis');
      S.slots.B=snapshot(parseSample(CB,'synthetic','file'),'synthetic');
      S.offline=S.slots.A; S.cmpA='A'; S.cmpB='B'; S.compare=true;
      S.stack=true; S.cell='4'; S.order='sorted'; S.orderRef='pooled';
      S.mismatch=true; S.tierStrip=true; S.tierNative=false;
      LOG.drawImage=[]; render();
    """)
    kinds = ctx.eval("LAYOUT.panels.map(function(p){return p.kind;}).join(',')")
    assert kinds == "mag,mag,tier,mismatch,diff", kinds
    assert ctx.eval("LOG.drawImage.length") == 5

    st = "(function(){var p=null,i;for(i=0;i<LAYOUT.panels.length;i++)" \
         "if(LAYOUT.panels[i].kind==='diff')p=LAYOUT.panels[i];return diffStats(p).%s;})()"
    assert ctx.eval(st % "conserve") < 1e-11
    assert abs(ctx.eval(st % "rho.med") - 0.37) < 0.03
    assert abs(ctx.eval(st % "ov.med") - 0.28) < 0.03
    assert ctx.eval(st % "movers.length") == 50
    assert "<tbody>" in ctx.eval("document.getElementById('tbl-movers').innerHTML")

    # every panel shares one permutation (DESIGN.md hazard 1)
    hashes = ctx.eval("LAYOUT.panels.map(function(p){return p._perm===LAYOUT.perm.perm;})"
                      ".join(',')")
    assert hashes == "true,true,true,true,true", hashes
    ctx.eval("S.stack=false; S.compare=false; S.mismatch=false; S.order='native';"
             "S.slots.A=null; S.slots.B=null;")


if __name__ == "__main__":
    raise SystemExit(pytest.main([os.path.abspath(__file__), "-q"]))
