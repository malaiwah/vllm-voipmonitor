#!/usr/bin/env python3
"""Trace the routing signal's stability over a run.

The loop holds its swaps whenever consecutive intervals disagree about which
experts *should* be K4:

    FQ interval step=200: jaccard 0.551 < floor 0.950 — holding 64 proposed swaps

That guard is right — do not chase a moving target — but it raises a question
the guard itself cannot answer: **does the signal ever settle?** If Jaccard
climbs toward the floor as counts accumulate, the guard is a warm-up delay and
convergence just needs time. If it plateaus below the floor, no amount of
patience helps and either the window or the floor is wrong.

Those two cases look identical for the first few minutes, and telling them
apart by watching a log scroll is how people end up "tuning until something
moves". So: plot the trajectory, then decide.

Usage:
    jaccard_trace.py <serve.log> [--floor 0.95]
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

RE_HELD = re.compile(
    r"FQ interval step=(\d+): jaccard ([\d.]+) < floor ([\d.]+)")
RE_APPLIED = re.compile(
    r"FQ interval step=(\d+): (\d+) swaps across (\d+) layers")


def trace(path: Path) -> tuple[list[tuple[int, float]], list[tuple[int, int]]]:
    held: list[tuple[int, float]] = []
    applied: list[tuple[int, int]] = []
    seen: set[int] = set()
    for line in path.read_text(errors="replace").splitlines():
        if m := RE_HELD.search(line):
            step = int(m.group(1))
            # Four ranks log the same interval; count it once.
            if step not in seen:
                seen.add(step)
                held.append((step, float(m.group(2))))
        elif m := RE_APPLIED.search(line):
            step, n = int(m.group(1)), int(m.group(2))
            if (step, n) not in applied:
                applied.append((step, n))
    return held, applied


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("log", type=Path)
    ap.add_argument("--floor", type=float, default=0.95)
    a = ap.parse_args(argv)

    held, applied = trace(a.log)
    if not held and not applied:
        print("no interval decisions in this log yet")
        return 0

    print(f"{'step':>8}  {'jaccard':>8}  trend")
    for step, jac in held:
        bar = "#" * int(jac * 40)
        print(f"{step:>8}  {jac:>8.3f}  {bar}")

    if held:
        first, last = held[0][1], held[-1][1]
        best = max(j for _, j in held)
        print(f"\nfirst {first:.3f}  last {last:.3f}  best {best:.3f}  "
              f"floor {a.floor:.3f}  intervals held {len(held)}")
        # The verdict this tool exists for. A rising trend that has not yet
        # reached the floor is a warm-up; a flat one is a design mismatch.
        window = held[-min(8, len(held)):]
        rising = window[-1][1] > window[0][1] + 0.01
        if best >= a.floor:
            print("VERDICT: the signal cleared the floor at least once — the "
                  "guard is a warm-up delay, not a block.")
        elif rising:
            print("VERDICT: still rising. Give it more of the SAME domain "
                  "before concluding anything about the floor.")
        else:
            print("VERDICT: plateaued below the floor. More time will not "
                  "help — either the collector window is too short for the "
                  "set to stabilise, or the floor is stricter than this "
                  "model's routing can ever satisfy. Fix the mismatch; do "
                  "NOT lower the floor to manufacture swaps.")
    n_applied = sum(n for _, n in applied if n)
    print(f"\nswaps actually applied: {n_applied} across {len(applied)} "
          f"intervals")
    return 0


if __name__ == "__main__":
    sys.exit(main())
