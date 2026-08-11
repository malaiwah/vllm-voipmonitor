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


def render_svg(counts, tiers, path: Path, title: str) -> None:
    """One SVG per sample: heat on the left, tier map on the right.

    SVG rather than PNG so this needs nothing beyond the standard library —
    the box already lost time to a headless-browser dependency once.
    """
    n_l = len(counts)
    n_e = len(counts[0]) if counts else 0
    if not n_l or not n_e:
        return
    cw, ch = 3, 6           # cell size
    gap = 40
    w = n_e * cw * 2 + gap + 80
    h = n_l * ch + 70
    peak = max((max(r) for r in counts if r), default=1.0) or 1.0

    px = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
          f'viewBox="0 0 {w} {h}">',
          f'<rect width="{w}" height="{h}" fill="#0d1117"/>',
          f'<text x="10" y="20" fill="#e6edf3" font-family="monospace" '
          f'font-size="13">{title}</text>',
          f'<text x="10" y="38" fill="#8b949e" font-family="monospace" '
          f'font-size="10">activation heat (log)</text>',
          f'<text x="{n_e * cw + gap + 10}" y="38" fill="#8b949e" '
          f'font-family="monospace" font-size="10">tier map: '
          f'<tspan fill="#f0883e">K4</tspan> / '
          f'<tspan fill="#1f6feb">K3</tspan></text>']

    import math

    def _runs(vals, key):
        """Merge horizontally adjacent cells that render identically.

        One <rect> per cell is 19,200 rects and ~1.4 MB per sample. Routing is
        smooth enough along the expert axis that run-length merging cuts that
        by roughly an order of magnitude with no visual change.
        """
        out, start, cur = [], 0, key(vals[0])
        for i in range(1, len(vals)):
            k = key(vals[i])
            if k != cur:
                out.append((start, i - start, cur))
                start, cur = i, k
        out.append((start, len(vals) - start, cur))
        return out

    def _shade(v):
        if v <= 0:
            return None
        # log scale: routing is heavy-tailed and a linear ramp shows one
        # bright expert per layer and nothing else. Quantized to 24 steps so
        # neighbouring cells merge.
        t = math.log1p(v) / math.log1p(peak)
        q = round(t * 24) / 24
        return f"rgb({int(20 + 235 * q)},{int(20 + 120 * q)},40)"

    for r in range(n_l):
        y = 48 + r * ch
        for c0, ln, col in _runs(counts[r], _shade):
            if col:
                px.append(f'<rect x="{10 + c0 * cw}" y="{y}" '
                          f'width="{ln * cw}" height="{ch}" fill="{col}"/>')
        if tiers:
            for c0, ln, is4 in _runs(tiers[r], lambda v: int(v) == 4):
                if is4:
                    px.append(
                        f'<rect x="{n_e * cw + gap + 10 + c0 * cw}" y="{y}" '
                        f'width="{ln * cw}" height="{ch}" fill="#f0883e"/>')
    px.append("</svg>")
    path.write_text("\n".join(px))


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
                   f"{s.stem}  step={info['step']}  "
                   f"active={info['active_cells']}  "
                   f"top26share={info['mean_top26_share']:.3f}")
        print(f"  -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
