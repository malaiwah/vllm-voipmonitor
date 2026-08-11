#!/usr/bin/env python3
"""Extract the battle-test metrics from a progressive-boot serve log.

Why a parser rather than eyeballing the log: BT-2's claim is "a hot restart
does not re-download". The tempting evidence is wall-clock -- and wall-clock
cannot support that claim. A restart can be faster because of page cache, a
warmer JIT, a quieter Hub, or fewer competing jobs on the box. Only the BYTES
FETCHED counter distinguishes "the cache was used" from "the network happened
to be fast today", so that is what `compare` asserts on.

The same reasoning runs through the rest: assert on artifacts (byte counters,
segment origin counts, tier digests, KV size), never on liveness or elapsed
time. This pipeline has produced nine distinct failures that looked exactly
like success.

Usage:
    bt_metrics.py extract <serve.log> [--json]
    bt_metrics.py compare <cold.log> <warm.log>     # BT-2 / BT-8 gate
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

# "FQ downloads: 5 in flight, 244.6 GiB delivered, 188 MiB/s avg (recent 140 MiB/s)"
RE_DL = re.compile(
    r"FQ downloads:.*?([\d.]+)\s*GiB delivered,\s*([\d.]+)\s*MiB/s avg")
RE_LAYER = re.compile(r"FQ progressive layer (\d+):")
RE_TIERS = re.compile(r"FQ progressive layer (\d+): tiers=(\S+)")
RE_DIGEST = re.compile(r"bits_digest=([0-9a-f]+)")
RE_LOCAL = re.compile(r"FQ progressive L\d+: local .*\(no fetch\)")
RE_PREFETCH = re.compile(r"FQ progressive L\d+: prefetched (\S+)")
RE_SHARED = re.compile(r"FQ progressive L\d+: shared (\S+)")
# "cached <file>" is the warm-restart signal: the segment was already in the
# fragment cache from a previous boot, so neither the network nor the local
# segment dir was consulted. Without this counter a warm boot looks like it
# did nothing at all -- zero local, zero prefetched -- which is exactly how a
# broken cache would also look.
RE_CACHED = re.compile(r"FQ progressive L\d+: cached (\S+)")
RE_KV = re.compile(r"Available KV cache memory:\s*(-?[\d.]+)\s*GiB")
RE_MODELLOAD = re.compile(
    r"Model loading took ([\d.]+) GiB memory and ([\d.]+) seconds")
RE_RECLAIM = re.compile(
    r"post-load reclaim: reserved ([\d.]+) -> ([\d.]+) GiB")
RE_SUBST = re.compile(r"bits_digest=\S+ tensors=\d+ substituted=(\S+)")
RE_TS = re.compile(r"(\d\d)-(\d\d) (\d\d):(\d\d):(\d\d)")
RE_RANK = re.compile(r"Worker_TP(\d+)")
READY = "Application startup complete"


def _ts(line: str) -> datetime | None:
    m = RE_TS.search(line)
    if not m:
        return None
    mo, d, h, mi, s = (int(x) for x in m.groups())
    # Year is absent from vLLM log lines; only DIFFERENCES are ever used, and
    # a fixed year keeps that arithmetic well-defined.
    return datetime(2026, mo, d, h, mi, s)


def extract(path: Path) -> dict:
    first_ts = ready_ts = None
    ready_seen = False
    gib = mibs = 0.0
    layers: set[int] = set()
    digests: dict[int, str] = {}
    tiers: dict[int, str] = {}
    local = shared = 0
    cached: set[str] = set()
    prefetched: set[str] = set()
    kv = model_gib = load_s = None
    reclaimed = 0.0
    substitutions: list[str] = []

    for line in path.read_text(errors="replace").splitlines():
        t = _ts(line)
        if t and first_ts is None:
            first_ts = t
        if READY in line:
            ready_seen = True
            ready_ts = t or ready_ts

        if m := RE_DL.search(line):
            # Per-rank counters; the LAST value of the busiest rank is the
            # run's total, so take the max rather than the last line seen.
            gib = max(gib, float(m.group(1)))
            mibs = max(mibs, float(m.group(2)))
        # Layer bookkeeping is rank-0 only: every rank logs its own copy and
        # summing them inflates the count fourfold.
        rank = RE_RANK.search(line)
        rank0 = rank is None or rank.group(1) == "0"
        if rank0:
            if m := RE_TIERS.search(line):
                layers.add(int(m.group(1)))
                tiers[int(m.group(1))] = m.group(2)
                if d := RE_DIGEST.search(line):
                    digests[int(m.group(1))] = d.group(1)
            elif m := RE_LAYER.search(line):
                layers.add(int(m.group(1)))
            if RE_LOCAL.search(line):
                local += 1
            if m := RE_PREFETCH.search(line):
                prefetched.add(m.group(1))
            if RE_SHARED.search(line):
                shared += 1
            if m := RE_CACHED.search(line):
                cached.add(m.group(1))
            if m := RE_SUBST.search(line):
                substitutions.append(m.group(1))
        if m := RE_KV.search(line):
            kv = float(m.group(1))
        if m := RE_MODELLOAD.search(line):
            model_gib, load_s = float(m.group(1)), float(m.group(2))
        if m := RE_RECLAIM.search(line):
            reclaimed = max(reclaimed, float(m.group(1)) - float(m.group(2)))

    ttfs = None
    if first_ts and ready_ts:
        ttfs = (ready_ts - first_ts).total_seconds()

    return {
        "log": str(path),
        "served": ready_seen,
        "time_to_serve_s": ttfs,
        "gib_fetched": gib,
        "mib_s_avg": mibs,
        "layers_loaded": len(layers),
        "segments_local_no_fetch": local,
        "segments_prefetched_remote": len(prefetched),
        "segments_shared_across_ranks": shared,
        "segments_from_cache": len(cached),
        "kv_cache_gib": kv,
        "model_gib": model_gib,
        "model_load_s": load_s,
        "allocator_reclaimed_gib": round(reclaimed, 2),
        "substitutions": substitutions,
        "tier_digest": _posture_digest(digests),
        "tiers_by_layer": tiers,
    }


def _posture_digest(digests: dict[int, str]) -> str | None:
    """One digest standing for the whole loaded posture.

    Comparing postures layer-by-layer is noisy; comparing one digest answers
    "did it come back at the SAME tiers?" -- which is what a restart claim
    actually rests on.
    """
    if not digests:
        return None
    import hashlib
    h = hashlib.blake2b(digest_size=8)
    for layer in sorted(digests):
        h.update(f"{layer}:{digests[layer]}".encode())
    return h.hexdigest()


def compare(cold: dict, warm: dict, *, tolerance_gib: float = 1.0) -> tuple[bool, list[str]]:
    """BT-2 / BT-8 gate. Returns (passed, report lines)."""
    out, ok = [], True
    out.append(f"{'metric':32s} {'cold':>14s} {'warm':>14s}")
    out.append("-" * 64)

    def row(label, a, b, fmt="{}"):
        out.append(f"{label:32s} {fmt.format(a):>14s} {fmt.format(b):>14s}"
                   if a is not None and b is not None
                   else f"{label:32s} {str(a):>14s} {str(b):>14s}")

    row("time to serve (s)", cold["time_to_serve_s"], warm["time_to_serve_s"],
        "{:.0f}")
    row("GiB fetched", cold["gib_fetched"], warm["gib_fetched"], "{:.1f}")
    row("layers loaded", cold["layers_loaded"], warm["layers_loaded"])
    row("local segments (no fetch)", cold["segments_local_no_fetch"],
        warm["segments_local_no_fetch"])
    row("segments served from cache", cold["segments_from_cache"],
        warm["segments_from_cache"])
    row("KV cache (GiB)", cold["kv_cache_gib"], warm["kv_cache_gib"], "{:.2f}")
    out.append("")

    if not warm["served"]:
        ok = False
        out.append("FAIL: warm restart never reached a served state.")

    # THE assertion. Not wall-clock: a faster restart proves nothing about the
    # cache, and a cache that silently re-downloads still finishes eventually.
    if warm["gib_fetched"] > tolerance_gib:
        ok = False
        out.append(
            f"FAIL: warm restart fetched {warm['gib_fetched']:.1f} GiB "
            f"(tolerance {tolerance_gib} GiB). The cache was NOT used; any "
            f"speedup here is page cache or a quiet network, not warm state.")
    else:
        out.append(
            f"PASS: warm restart fetched {warm['gib_fetched']:.1f} GiB "
            f"vs {cold['gib_fetched']:.1f} GiB cold.")

    # A restart that comes back at a DIFFERENT posture has lost state, however
    # fast it was.
    if cold["tier_digest"] and warm["tier_digest"]:
        if cold["tier_digest"] != warm["tier_digest"]:
            ok = False
            out.append(
                f"FAIL: posture changed across restart "
                f"({cold['tier_digest']} -> {warm['tier_digest']}). Same "
                f"policy must reload the same experts at the same K.")
        else:
            out.append(f"PASS: posture identical ({cold['tier_digest']}).")
    else:
        out.append("WARN: no tier digests -- posture equality unverified.")

    if cold["time_to_serve_s"] and warm["time_to_serve_s"]:
        out.append(
            f"speedup: {cold['time_to_serve_s'] / warm['time_to_serve_s']:.1f}x "
            f"(reported, NOT asserted on)")
    return ok, out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("log", type=Path)
    e.add_argument("--json", action="store_true")
    c = sub.add_parser("compare")
    c.add_argument("cold", type=Path)
    c.add_argument("warm", type=Path)
    c.add_argument("--tolerance-gib", type=float, default=1.0)
    a = ap.parse_args(argv)

    if a.cmd == "extract":
        m = extract(a.log)
        if a.json:
            print(json.dumps(m, indent=2, sort_keys=True))
        else:
            for k, v in m.items():
                if k != "tiers_by_layer":
                    print(f"  {k:32s} {v}")
        return 0

    ok, lines = compare(extract(a.cold), extract(a.warm),
                        tolerance_gib=a.tolerance_gib)
    print("\n".join(lines))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
