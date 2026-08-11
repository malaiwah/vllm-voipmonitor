#!/usr/bin/env python3
"""Decode and render the live expert-activation matrix (`fq-heatmap/1`).

The endpoint ships the 75x256 matrix base64-encoded rather than as nested
JSON lists — layer-major, little-endian, `count`/`mass`/`live_count` as bf16
and `tier` as u8. That is the right call for a 19,200-cell payload polled every
few seconds, but it means a naive reader looking for `layers[i][j]` finds a
list of layer IDs, reports "0 active cells", and concludes the model is idle
while it is at 100% GPU. This decodes it properly.

bf16 is the top 16 bits of an IEEE-754 float32, so widening is a shift, not a
conversion — no numpy dependency and no precision surprise.

Renders two panels per sample:
  * activation heat  — which experts the traffic actually routes to
  * tier map         — which experts are currently K4

Read them together. The whole fungible-quant thesis is that those two pictures
should agree, and the interesting evidence is exactly where they do not.

Usage:
    render_heatmap.py <sample.json> [sample2.json ...] --out DIR
"""
from __future__ import annotations

import argparse
import base64
import json
import struct
import sys
from pathlib import Path


def _bf16(raw: bytes) -> list[float]:
    """bf16 -> float. The top 16 bits of a float32, so shift and reinterpret."""
    out: list[float] = []
    for i in range(0, len(raw) - 1, 2):
        lo, hi = raw[i], raw[i + 1]
        out.append(struct.unpack("<f", bytes((0, 0, lo, hi)))[0])
    return out


def _u8(raw: bytes) -> list[int]:
    return list(raw)


def decode(doc: dict, field: str) -> list[list[float]]:
    """Return the field as [layer][expert], or [] when absent."""
    blob = doc.get(field)
    if not blob:
        return []
    enc = doc.get("encoding") or {}
    raw = base64.b64decode(blob)
    dtype = enc.get(field, "bf16")
    flat = _u8(raw) if dtype == "u8" else _bf16(raw)
    n_l, n_e = int(doc["num_layers"]), int(doc["num_experts"])
    if len(flat) < n_l * n_e:
        raise ValueError(
            f"{field}: decoded {len(flat)} cells, expected {n_l * n_e}")
    return [flat[r * n_e:(r + 1) * n_e] for r in range(n_l)]


def summarise(doc: dict) -> dict:
    counts = decode(doc, "count")
    tiers = decode(doc, "tier")
    total = sum(sum(r) for r in counts)
    active = sum(1 for r in counts for v in r if v > 0)
    k4 = sum(1 for r in tiers for v in r if int(v) == 4)
    # Concentration is the number worth watching: if a handful of experts per
    # layer carry the traffic, a fixed K4 budget spent on the right ones is
    # worth a lot; if routing is flat, it cannot be.
    top_share = []
    for row in counts:
        s = sum(row)
        if s <= 0:
            continue
        top = sorted(row, reverse=True)[:26]   # the per-layer K4 budget here
        top_share.append(sum(top) / s)
    return {
        "step": doc.get("step"),
        "cells": len(counts) * (len(counts[0]) if counts else 0),
        "active_cells": active,
        "total_activation": total,
        "k4_cells": k4,
        "mean_top26_share": (sum(top_share) / len(top_share)) if top_share
        else 0.0,
        "layers": len(counts),
    }


def render_svg(counts, tiers, path: Path, title: str, layers=None,
               topk: int = 26) -> None:
    """Three panels: where traffic goes, where the bits are, and the gap.

    The third panel is the one worth having. Heat and tier side by side make
    you eyeball two pictures and hold the difference in your head; the gap
    panel computes it:

        green  the K4 budget is spent on a top-K expert          (hit)
        red    a top-K expert is stuck at K3                     (miss)
        grey   K4 spent on an expert outside the top-K           (waste)

    Red and grey are the tuning surface. A layer that is mostly green needs
    nothing; a layer with a red band has traffic the budget is not covering,
    and a grey band is budget that could be moved. Everything the policy is
    trying to do is visible as "make red and grey disappear".
    """
    n_l = len(counts)
    n_e = len(counts[0]) if counts else 0
    if not n_l or not n_e:
        return
    cw, ch = 4, 9
    pad, gap, top = 62, 34, 92
    pw = n_e * cw
    w = pad + pw * 3 + gap * 2 + 24
    h = top + n_l * ch + 58
    peak = max((max(r) for r in counts if r), default=1.0) or 1.0
    layers = layers or list(range(n_l))

    import math
    P = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
         f'viewBox="0 0 {w} {h}" font-family="ui-monospace,SFMono-Regular,'
         f'Menlo,monospace">',
         f'<defs><linearGradient id="hg" x1="0" x2="1">'
         f'<stop offset="0" stop-color="#0b1021"/>'
         f'<stop offset="0.35" stop-color="#3b2f6b"/>'
         f'<stop offset="0.7" stop-color="#c2410c"/>'
         f'<stop offset="1" stop-color="#fde68a"/></linearGradient></defs>',
         f'<rect width="{w}" height="{h}" fill="#0d1117"/>',
         f'<text x="{pad}" y="26" fill="#e6edf3" font-size="14" '
         f'font-weight="600">{title}</text>']

    heads = [("activation heat  (log scale)", pad),
             (f"tier map  K4={topk if tiers else 0}/layer", pad + pw + gap),
             ("budget vs traffic", pad + (pw + gap) * 2)]
    for label, x in heads:
        P.append(f'<text x="{x}" y="50" fill="#8b949e" font-size="11">'
                 f'{label}</text>')
    # legend for the gap panel
    lx = pad + (pw + gap) * 2
    for dx, col, lab in ((0, "#2ea043", "hit"), (54, "#f85149", "miss"),
                         (116, "#6e7681", "waste")):
        P.append(f'<rect x="{lx + dx}" y="60" width="9" height="9" '
                 f'fill="{col}"/>'
                 f'<text x="{lx + dx + 13}" y="69" fill="#8b949e" '
                 f'font-size="10">{lab}</text>')
    # heat colour ramp
    P.append(f'<rect x="{pad}" y="60" width="120" height="9" fill="url(#hg)"/>'
             f'<text x="{pad + 126}" y="69" fill="#8b949e" font-size="10">'
             f'0 &#8594; {peak:,.0f}</text>')

    def _runs(vals, key):
        out, start, cur = [], 0, key(vals[0])
        for i in range(1, len(vals)):
            k = key(vals[i])
            if k != cur:
                out.append((start, i - start, cur))
                start, cur = i, k
        out.append((start, len(vals) - start, cur))
        return out

    def _heat(v):
        if v <= 0:
            return None
        t = math.log1p(v) / math.log1p(peak)
        q = round(t * 20) / 20
        # dark blue -> purple -> orange -> pale yellow: perceptually ordered,
        # unlike a raw red ramp where mid values are indistinguishable.
        stops = ((0.0, (11, 16, 33)), (0.35, (59, 47, 107)),
                 (0.70, (194, 65, 12)), (1.0, (253, 230, 138)))
        for (a, ca), (b, cb) in zip(stops, stops[1:]):
            if q <= b:
                f = (q - a) / (b - a) if b > a else 0
                return "rgb(%d,%d,%d)" % tuple(
                    int(ca[i] + (cb[i] - ca[i]) * f) for i in range(3))
        return "rgb(253,230,138)"

    for r in range(n_l):
        y = top + r * ch
        if r % 5 == 0:
            P.append(f'<text x="{pad - 8}" y="{y + ch - 1}" fill="#6e7681" '
                     f'font-size="9" text-anchor="end">L{layers[r]}</text>')
        for c0, ln, col in _runs(counts[r], _heat):
            if col:
                P.append(f'<rect x="{pad + c0 * cw}" y="{y}" '
                         f'width="{ln * cw}" height="{ch - 1}" fill="{col}"/>')
        if not tiers:
            continue
        x1 = pad + pw + gap
        for c0, ln, is4 in _runs(tiers[r], lambda v: int(v) == 4):
            if is4:
                P.append(f'<rect x="{x1 + c0 * cw}" y="{y}" '
                         f'width="{ln * cw}" height="{ch - 1}" fill="#f0883e"/>')
        # gap panel
        row = counts[r]
        hot = set(sorted(range(n_e), key=lambda i: (-row[i], i))[:topk])
        x2 = pad + (pw + gap) * 2

        def _cls(i, _hot=hot, _t=tiers[r]):
            k4 = int(_t[i]) == 4
            if k4 and i in _hot:
                return "#2ea043"
            if k4:
                return "#6e7681"
            if i in _hot:
                return "#f85149"
            return None
        for c0, ln, col in _runs(list(range(n_e)), _cls):
            if col:
                P.append(f'<rect x="{x2 + c0 * cw}" y="{y}" '
                         f'width="{ln * cw}" height="{ch - 1}" fill="{col}"/>')

    if tiers:
        hits = miss = waste = 0
        for r in range(n_l):
            row = counts[r]
            hot = set(sorted(range(n_e), key=lambda i: (-row[i], i))[:topk])
            for i in range(n_e):
                k4 = int(tiers[r][i]) == 4
                if k4 and i in hot:
                    hits += 1
                elif k4:
                    waste += 1
                elif i in hot:
                    miss += 1
        tot = max(hits + waste, 1)
        P.append(f'<text x="{pad}" y="{h - 22}" fill="#8b949e" font-size="11">'
                 f'budget on target: {hits:,}/{tot:,} '
                 f'({100 * hits / tot:.1f}%)  &#183;  misplaced {waste:,}'
                 f'  &#183;  uncovered hot experts {miss:,}</text>')
    P.append("</svg>")
    path.write_text("\n".join(P))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("samples", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=Path("."))
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)

    for s in a.samples:
        doc = json.loads(s.read_text())
        info = summarise(doc)
        print(f"{s.name}: step={info['step']} "
              f"{info['active_cells']:,}/{info['cells']:,} active cells, "
              f"mass {info['total_activation']:,.0f}, "
              f"K4 {info['k4_cells']:,}, "
              f"top-26 share {info['mean_top26_share']:.3f}")
        out = a.out / (s.stem + ".svg")
        render_svg(decode(doc, "count"), decode(doc, "tier"), out,
                   f"{s.stem}   step {info['step']}   "
                   f"active {info['active_cells']:,}/{info['cells']:,}   "
                   f"top-26 mass share {info['mean_top26_share']:.3f}",
                   layers=doc.get("layers"))
        print(f"  -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
