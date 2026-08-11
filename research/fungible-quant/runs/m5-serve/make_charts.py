#!/usr/bin/env python3
"""Render the M5 evidence charts from a swap_evidence timeline.

Output is dependency-free SVG so the PR renders it inline on GitHub without a
build step and without trusting a CDN.

Design constraints followed deliberately:
 * Categorical hues are the default theme's slots in FIXED order (blue,
   orange, aqua, violet) — never cycled, never re-picked per chart.
 * NO dual-axis charts. Throughput (tok/s) and expert counts are different
   scales, so they are two stacked panels sharing one time axis, not two
   y-scales on one plot.
 * Both light and dark surfaces are defined via CSS custom properties with a
   prefers-color-scheme block, so the figure stays readable in either GitHub
   theme instead of becoming a white slab in dark mode.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# Default categorical theme, light/dark pairs, used in fixed slot order.
SERIES = [("#2a78d6", "#3987e5"),   # 1 blue
          ("#eb6834", "#d95926"),   # 2 orange
          ("#1baf7a", "#199e70"),   # 3 aqua
          ("#4a3aa7", "#9085e9")]   # 4 violet

CSS = """
  .surf { fill: #fcfcfb; }
  .tx   { fill: #0b0b0b; font-family: ui-sans-serif, system-ui, sans-serif; }
  .tx2  { fill: #52514e; font-family: ui-sans-serif, system-ui, sans-serif; }
  .grid { stroke: #e6e5e1; stroke-width: 1; }
  .axis { stroke: #b8b7b2; stroke-width: 1; }
  .band { fill: #0b0b0b; opacity: 0.035; }
  .s1 { stroke: %s; } .s2 { stroke: %s; } .s3 { stroke: %s; } .s4 { stroke: %s; }
  .f1 { fill: %s; } .f2 { fill: %s; } .f3 { fill: %s; } .f4 { fill: %s; }
  @media (prefers-color-scheme: dark) {
    .surf { fill: #1a1a19; }
    .tx  { fill: #ffffff; } .tx2 { fill: #c3c2b7; }
    .grid { stroke: #33322f; } .axis { stroke: #56554f; }
    .band { fill: #ffffff; opacity: 0.05; }
    .s1 { stroke: %s; } .s2 { stroke: %s; } .s3 { stroke: %s; } .s4 { stroke: %s; }
    .f1 { fill: %s; } .f2 { fill: %s; } .f3 { fill: %s; } .f4 { fill: %s; }
  }
""" % tuple([c[0] for c in SERIES] * 2 + [c[1] for c in SERIES] * 2)


def esc(s: str) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


def load(path: Path) -> tuple[list[dict], list[dict]]:
    samples, phases = [], []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        (phases if row.get("kind") == "phase" else samples).append(row)
    return samples, phases


def nice_ticks(lo: float, hi: float, n: int = 5) -> list[float]:
    if hi <= lo:
        return [lo]
    raw = (hi - lo) / n
    mag = 10 ** (len(str(int(abs(raw)))) - 1) if abs(raw) >= 1 else 0.1
    step = next((m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw), 10 * mag)
    start = (int(lo / step)) * step
    out, v = [], start
    while v <= hi + step * 0.5:
        if v >= lo - step * 0.5:
            out.append(round(v, 6))
        v += step
    return out or [lo, hi]


class Panel:
    """One plot area with a shared x (time) domain."""

    def __init__(self, x0, y0, w, h, t0, t1, ylabel):
        self.x0, self.y0, self.w, self.h = x0, y0, w, h
        self.t0, self.t1 = t0, max(t1, t0 + 1e-6)
        self.ylabel = ylabel
        self.ymin, self.ymax = 0.0, 1.0

    def fit(self, values):
        vals = [v for v in values if v is not None]
        if vals:
            self.ymax = max(vals) * 1.15 or 1.0
        return self

    def px(self, t):
        return self.x0 + (t - self.t0) / (self.t1 - self.t0) * self.w

    def py(self, v):
        span = (self.ymax - self.ymin) or 1.0
        return self.y0 + self.h - (v - self.ymin) / span * self.h

    def frame(self, out, xticks_fmt=None, xticks=None):
        out.append(f'<rect x="{self.x0}" y="{self.y0}" width="{self.w}" '
                   f'height="{self.h}" fill="none" class="axis"/>')
        for t in nice_ticks(self.ymin, self.ymax):
            y = self.py(t)
            if not (self.y0 - 1 <= y <= self.y0 + self.h + 1):
                continue
            out.append(f'<line x1="{self.x0}" y1="{y:.1f}" x2="{self.x0+self.w}" '
                       f'y2="{y:.1f}" class="grid"/>')
            lab = f"{t:.0f}" if abs(t) >= 10 else f"{t:g}"
            out.append(f'<text x="{self.x0-8}" y="{y+4:.1f}" text-anchor="end" '
                       f'class="tx2" font-size="11">{lab}</text>')
        out.append(f'<text x="{self.x0-46}" y="{self.y0+self.h/2}" class="tx2" '
                   f'font-size="11" text-anchor="middle" '
                   f'transform="rotate(-90 {self.x0-46} {self.y0+self.h/2})">'
                   f'{esc(self.ylabel)}</text>')
        if xticks:
            for t, lab in xticks:
                x = self.px(t)
                out.append(f'<text x="{x:.1f}" y="{self.y0+self.h+16}" '
                           f'text-anchor="middle" class="tx2" font-size="11">'
                           f'{esc(lab)}</text>')

    def line(self, out, pts, cls, width=2):
        pts = [(self.px(t), self.py(v)) for t, v in pts if v is not None]
        if len(pts) < 2:
            return
        d = "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        out.append(f'<path d="{d}" fill="none" class="{cls}" stroke-width="{width}" '
                   f'stroke-linejoin="round" stroke-linecap="round"/>')

    def phase_bands(self, out, phases, label=True):
        starts = [p for p in phases if p.get("event") == "phase_start"]
        for i, p in enumerate(starts):
            a = self.px(p["t"])
            b = (self.px(starts[i + 1]["t"]) if i + 1 < len(starts)
                 else self.x0 + self.w)
            if i % 2 == 1:
                out.append(f'<rect x="{a:.1f}" y="{self.y0}" '
                           f'width="{max(0,b-a):.1f}" height="{self.h}" class="band"/>')
            out.append(f'<line x1="{a:.1f}" y1="{self.y0}" x2="{a:.1f}" '
                       f'y2="{self.y0+self.h}" class="axis" stroke-dasharray="3 3"/>')
            if label:
                out.append(f'<text x="{a+6:.1f}" y="{self.y0+14}" class="tx2" '
                           f'font-size="11">{esc(p["family"])}</text>')


def render(samples: list[dict], phases: list[dict], title: str,
           subtitle: str) -> str:
    if not samples:
        raise SystemExit("no samples in timeline")
    t0 = min(s["t"] for s in samples)
    t1 = max(s["t"] for s in samples)
    W, H = 900, 560
    L, R = 70, 24
    pw = W - L - R

    tput = [(s["t"], s.get("decode_tok_s")) for s in samples]
    swaps = [(s["t"], (s.get("fq") or {}).get("swaps_total")) for s in samples]

    def occ_at(s, tier):
        occ = (s.get("fq") or {}).get("tier_occupancy") or {}
        tot = 0.0
        for layer_tiers in occ.values():
            tot += float(layer_tiers.get(str(tier), 0) or 0)
        return tot

    k5 = [(s["t"], occ_at(s, 5)) for s in samples]
    k4 = [(s["t"], occ_at(s, 4)) for s in samples]

    top = Panel(L, 66, pw, 180, t0, t1, "decode tok/s").fit(v for _, v in tput)
    bot = Panel(L, 320, pw, 180, t0, t1, "experts").fit(
        [v for _, v in swaps] + [v for _, v in k5] + [v for _, v in k4])

    xt = []
    for frac in (0, 0.25, 0.5, 0.75, 1.0):
        t = t0 + (t1 - t0) * frac
        xt.append((t, f"{(t - t0) / 60:.0f}m"))

    o: list[str] = []
    o.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
             f'width="{W}" height="{H}" role="img" '
             f'aria-label="{esc(title)}">')
    o.append(f"<style>{CSS}</style>")
    o.append(f'<rect width="{W}" height="{H}" class="surf"/>')
    o.append(f'<text x="{L}" y="30" class="tx" font-size="17" '
             f'font-weight="600">{esc(title)}</text>')
    o.append(f'<text x="{L}" y="50" class="tx2" font-size="12">'
             f'{esc(subtitle)}</text>')

    top.phase_bands(o, phases)
    top.frame(o)
    top.line(o, tput, "s1")
    o.append(f'<text x="{L}" y="{top.y0-6}" class="tx2" font-size="12">'
             f'Serving throughput</text>')

    bot.phase_bands(o, phases, label=False)
    bot.frame(o, xticks=xt)
    bot.line(o, swaps, "s2")
    bot.line(o, k5, "s3")
    if any(v for _, v in k4):
        bot.line(o, k4, "s4")
    o.append(f'<text x="{L}" y="{bot.y0-6}" class="tx2" font-size="12">'
             f'Expert re-tiering</text>')
    o.append(f'<text x="{L+pw/2}" y="{H-14}" text-anchor="middle" class="tx2" '
             f'font-size="11">minutes since start of run</text>')

    # Legend: identity is never carried by color alone.
    lx = L
    for cls, label in (("f1", "decode tok/s"), ("f2", "cumulative swaps"),
                       ("f3", "experts at K5"),
                       *( (("f4", "experts at K4"),) if any(v for _, v in k4) else ())):
        o.append(f'<rect x="{lx}" y="{H-42}" width="10" height="10" rx="2" '
                 f'class="{cls}"/>')
        o.append(f'<text x="{lx+15}" y="{H-33}" class="tx2" font-size="11">'
                 f'{esc(label)}</text>')
        lx += 22 + 7.2 * len(label)
    o.append("</svg>")
    return "\n".join(o)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("timeline", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--title", default="Live expert re-tiering under load")
    ap.add_argument("--subtitle", default="GLM-5.2 3.0bpw assembled from "
                                          "Progressive Tensors segments, TP4")
    a = ap.parse_args()
    samples, phases = load(a.timeline)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(render(samples, phases, a.title, a.subtitle))
    print(f"wrote {a.out} ({len(samples)} samples, {len(phases)} phase rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
