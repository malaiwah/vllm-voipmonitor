#!/usr/bin/env python3
"""Record what the fungible-quant loop actually does to a live serve.

Two jobs, deliberately kept in one file so the timeline and the traffic that
caused it cannot drift apart:

``scrape``    poll ``/metrics`` and append one JSONL row per interval, mixing
              vLLM's own throughput counters with the ``fq_*`` gauges the loop
              exports (swaps per layer, tier occupancy, jaccard, policy age).
              This is the series the swap-timeline chart is drawn from.

``workload``  drive sustained concurrent load whose *domain* changes on a
              schedule.  This matters: the M2 dryrun proposed zero swaps
              because a fixed synthetic prompt mix never shifts the routing
              distribution, so the incumbent expert set always wins on
              hysteresis.  Real upgrades need real router movement, and the
              honest way to get it is to change what the model is asked to do
              rather than to lower the guards until something moves.

The phase boundaries are written into the same JSONL as ``phase`` rows, so a
reader can line up "traffic changed" against "experts were re-tiered" without
trusting a separate clock.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

# Prompt families chosen to route differently from one another.  Each is a
# (name, builder) pair; builders take an index so repeated draws are distinct
# without being random noise (a fixed corpus keeps runs comparable).
FAMILIES: dict[str, list[str]] = {
    "math": [
        "A train leaves at {a}:00 travelling {b} km/h. A second leaves an hour "
        "later at {c} km/h. Show your reasoning step by step and give the "
        "distance from the station where they meet.",
        "Compute the sum of all integers between {a} and {b} that are "
        "divisible by neither 3 nor 5. Show each step.",
        "A bag holds {a} red and {b} blue marbles. Three are drawn without "
        "replacement. Derive the probability that exactly two are red.",
    ],
    "code": [
        "Write a Python function that merges {a} sorted iterators lazily "
        "without loading them into memory, and explain its complexity.",
        "Here is a bug: a Rust HashMap iteration order changes between runs "
        "and a test depends on it. Explain the root cause and give {a} "
        "distinct fixes with tradeoffs.",
        "Implement a lock-free single-producer single-consumer ring buffer of "
        "capacity {a} in C11 and justify each memory ordering you choose.",
    ],
    "prose_multiling": [
        "Écris un court essai sur l'impact de la quantification des modèles de "
        "langue sur l'accès communautaire à l'IA, en {a} paragraphes.",
        "用中文解释混合专家模型（MoE）中的路由机制，并说明为什么专家的使用频率"
        "分布通常是不均匀的。请分{a}点说明。",
        "Escribe un diálogo de {a} turnos entre un ingeniero y un escéptico "
        "sobre si los pesos cuantizados pueden considerarse reproducibles.",
    ],
    "biomed": [
        "Explain the mechanism by which {a} distinct classes of beta-lactam "
        "antibiotics are defeated by bacterial resistance, and which is most "
        "clinically worrying.",
        "A patient presents with hyponatremia of {a} mmol/L and normal volume "
        "status. Walk through the differential and the diagnostic order.",
        "Describe how CRISPR base editors differ from prime editors, and give "
        "{a} cases where one is clearly preferable.",
    ],
}


def build_prompt(family: str, i: int) -> str:
    """The i-th prompt of a family: stable everywhere, distinct by construction.

    Two bugs the first version had, both of which quietly weaken the evidence
    rather than crashing:

    * ``hash((family, i))`` seeded the parameters, and ``str.__hash__`` is
      salted per interpreter (PYTHONHASHSEED).  Every process therefore drew a
      DIFFERENT corpus, so two runs of the campaign were not comparable — the
      opposite of what the module docstring promises — and the distinctness
      test flipped between pass and fail depending on the salt.
    * three of the four families use only ``{a}`` (3..97), so their whole
      corpus was 3 templates x 95 values = 285 prompts.  A 420 s phase at
      concurrency 24 issues thousands of requests, i.e. every prompt is
      re-issued several times, and vLLM's prefix cache then serves the prefill
      from cache.  Cached prefill routes no tokens, which understates exactly
      the routing pressure the domain shift is supposed to create.  The
      leading clause varies per draw, so every request is a cache miss from
      the first token.
    """
    tpl = FAMILIES[family][i % len(FAMILIES[family])]
    seed = int.from_bytes(
        hashlib.blake2b(f"{family}:{i}".encode(), digest_size=8).digest(),
        "big")
    rng = random.Random(seed)
    body = tpl.format(a=rng.randint(3, 97), b=rng.randint(101, 499),
                      c=rng.randint(50, 150))
    return f"(request {i}, {family}) {body}"


# ------------------------------------------------------------------ scraping
# A sample line is  name[{labels}] value [timestamp] [# {exemplar} v [ts]].
# The first version required the value to be the LAST token, so any line
# carrying a timestamp or an OpenMetrics exemplar was dropped silently — a
# missing series, not a visibly broken one.  Leading whitespace is legal in
# the text format too and was likewise dropped.
_SAMPLE = re.compile(
    r'^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)'
    r'(?:\{(?P<labels>[^}]*)\})?'
    r'[ \t]+(?P<value>[^\s]+)'
    r'(?:[ \t]+[0-9eE.+-]+)?'          # optional sample timestamp
    r'[ \t]*(?:#.*)?$')                # optional exemplar


def parse_metrics(text: str,
                  report: dict | None = None) -> dict[str, float]:
    """Flatten a Prometheus exposition into ``{name{labels}: value}``.

    Non-finite values (``NaN``, ``+Inf``, ``-Inf``) are DROPPED rather than
    stored.  They are legal in the exposition format and a 0/0 gauge emits
    them for real, but a NaN that reaches the timeline turns every coordinate
    of its chart series into ``nan`` — the SVG stays well-formed and the line
    simply vanishes, which reads as "nothing happened".  ``json.dumps`` also
    writes bare ``NaN``/``Infinity``, which is not valid JSON for any reader
    other than Python's own.

    ``report``, when given, collects what was thrown away so the caller can
    put it in the timeline instead of losing it: ``unparsed`` (count of
    non-comment lines the regex rejected) and ``nonfinite`` (their keys).
    """
    out: dict[str, float] = {}
    unparsed = 0
    nonfinite: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = _SAMPLE.match(line)
        if not m:
            unparsed += 1
            continue
        key = m["name"] if not m["labels"] else f'{m["name"]}{{{m["labels"]}}}'
        try:
            val = float(m["value"])
        except ValueError:
            unparsed += 1
            continue
        if not math.isfinite(val):
            nonfinite.append(key)
            continue
        out[key] = val
    if report is not None:
        if unparsed:
            report["unparsed"] = unparsed
        if nonfinite:
            report["nonfinite"] = nonfinite
    return out


def series(flat: dict[str, float], name: str) -> list[float]:
    """Every labelled child of EXACTLY ``name`` (not of a longer name).

    ``k.startswith(name)`` was wrong in a way that produces a plausible
    number: ``vllm:generation_tokens`` is a prefix of
    ``vllm:generation_tokens_created`` (the unix timestamp prometheus_client
    emits beside a Counter in single-process mode) and of ``_sum`` /
    ``_count`` / ``_bucket`` should the metric ever be exposed as a
    histogram.  Summing those into the token counter adds a ~1.7e9 constant
    — harmless until a new label set appears mid-run, at which point the
    derived rate spikes by 3e8 tok/s.
    """
    return [v for k, v in flat.items()
            if k == name or k.startswith(name + "{")]


def fq_view(flat: dict[str, float]) -> dict:
    """The fungible-quant slice of a scrape, in a shape a chart can use."""
    swaps, occ = {}, {}
    for key, val in flat.items():
        if key.startswith("fq_swaps_total{"):
            layer = re.search(r'layer="([^"]+)"', key)
            if layer:
                swaps[layer[1]] = val
        elif key.startswith("fq_tier_occupancy{"):
            layer = re.search(r'layer="([^"]+)"', key)
            tier = re.search(r'tier="([^"]+)"', key)
            if layer and tier:
                occ.setdefault(layer[1], {})[tier[1]] = val
    return {
        "swaps_by_layer": swaps,
        "swaps_total": sum(swaps.values()),
        "tier_occupancy": occ,
        "jaccard": flat.get("fq_jaccard"),
        "probe_kld": flat.get("fq_probe_kld"),
        "policy_age_steps": flat.get("fq_policy_age_steps"),
        "rollbacks": flat.get("fq_rollbacks_total"),
    }


def scrape_loop(base: str, out: Path, interval: float,
                stop: threading.Event) -> None:
    # The RATE is derived on the monotonic clock, the row timestamp on the
    # wall clock.  Mixing them was a real hazard: an NTP step during a
    # half-hour run can make ``t0 - prev_t`` a few milliseconds and turn a
    # 600 tok/s serve into a 600 000 tok/s one in the chart.
    prev_tok, prev_m = None, None
    url = base.rstrip("/") + "/metrics"
    with out.open("a") as fh:
        while not stop.is_set():
            t0, m0 = time.time(), time.monotonic()
            row: dict = {"t": t0, "kind": "sample"}
            try:
                with urllib.request.urlopen(url, timeout=10) as resp:
                    report: dict = {}
                    flat = parse_metrics(resp.read().decode(), report)
                row["fq"] = fq_view(flat)
                # Absence of the fq series is a result too: without this a
                # baseline serve and a live serve whose metric names moved
                # produce the same flat-zero chart.
                row["fq_present"] = any(
                    k == "fq_swaps_total" or k.startswith("fq_") for k in flat)
                if report:
                    row["scrape_warnings"] = report
                # vLLM's own decode counter; name has moved across versions,
                # so accept either and record which one we read.
                for name in ("vllm:generation_tokens_total",
                             "vllm:generation_tokens",
                             "vllm:tokens_total"):
                    hit = series(flat, name)
                    if hit:
                        tok = sum(hit)
                        row["gen_tokens_total"] = tok
                        row["gen_tokens_metric"] = name
                        if prev_tok is not None and m0 > prev_m:
                            if tok < prev_tok:
                                # The exporter restarted (or a label set went
                                # away).  (tok - prev)/dt here is a spike of
                                # minus-a-million tok/s that fit() cannot see
                                # and the chart draws off the bottom edge.
                                row["counter_reset"] = True
                            else:
                                row["decode_tok_s"] = (
                                    (tok - prev_tok) / (m0 - prev_m))
                        prev_tok, prev_m = tok, m0
                        break
                run = series(flat, "vllm:num_requests_running")
                if run:
                    row["requests_running"] = sum(run)
            except (urllib.error.URLError, OSError, TimeoutError) as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            stop.wait(max(0.0, interval - (time.monotonic() - m0)))


# ------------------------------------------------------------------ workload
def one_request(base: str, model: str, prompt: str, max_tokens: int,
                timeout: float) -> dict:
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "stream": False,
    }).encode()
    req = urllib.request.Request(
        base.rstrip("/") + "/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode())
        usage = payload.get("usage", {})
        return {"ok": True, "latency": time.time() - t0,
                "completion_tokens": usage.get("completion_tokens"),
                "prompt_tokens": usage.get("prompt_tokens")}
    except Exception as exc:  # noqa: BLE001 - a failed request is data
        return {"ok": False, "latency": time.time() - t0,
                "error": f"{type(exc).__name__}: {exc}"}


def workload(base: str, model: str, out: Path, phases: list[tuple[str, float]],
             concurrency: int, max_tokens: int,
             stop: threading.Event) -> dict:
    """Run each (family, seconds) phase at fixed concurrency.

    Returns a totals dict so a caller can refuse to call a run that generated
    nothing a success.
    """
    counter = {"i": 0}
    lock = threading.Lock()
    totals = {"requests": 0, "ok": 0, "completion_tokens": 0}

    def worker(family: str, deadline: float, results: list[dict]) -> None:
        while time.monotonic() < deadline and not stop.is_set():
            with lock:
                idx = counter["i"]
                counter["i"] += 1
            r = one_request(base, model, build_prompt(family, idx),
                            max_tokens, timeout=600)
            r["family"] = family
            with lock:
                results.append(r)

    with out.open("a") as fh:
        for family, seconds in phases:
            # A FRESH list per phase.  Selecting out of one shared list with
            # `r["family"] == family` double-counted every phase that repeats
            # a family: measured on a stub server, a third phase that issued
            # 96 requests reported 180 and twice the real tok/s.
            results: list[dict] = []
            start, mstart = time.time(), time.monotonic()
            fh.write(json.dumps({"t": start, "kind": "phase",
                                 "event": "phase_start", "family": family,
                                 "planned_seconds": seconds,
                                 "concurrency": concurrency}) + "\n")
            fh.flush()
            print(f"[workload] phase {family} for {seconds}s "
                  f"@ concurrency {concurrency}", flush=True)
            deadline = mstart + seconds
            threads = [threading.Thread(target=worker,
                                        args=(family, deadline, results),
                                        daemon=True)
                       for _ in range(concurrency)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            with lock:
                done = list(results)
            ok = [r for r in done if r["ok"]]
            toks = sum(r.get("completion_tokens") or 0 for r in ok)
            # One clock read, used for both fields: the original took two, so
            # the published tok/s was never exactly completion_tokens/wall.
            wall = max(1e-9, time.monotonic() - mstart)
            totals["requests"] += len(done)
            totals["ok"] += len(ok)
            totals["completion_tokens"] += toks
            fh.write(json.dumps({
                "t": time.time(), "kind": "phase", "event": "phase_end",
                "family": family, "requests": len(done), "ok": len(ok),
                "failed": len(done) - len(ok),
                "completion_tokens": toks,
                "wall_seconds": wall,
                "tok_s": toks / wall,
            }) + "\n")
            fh.flush()
            if stop.is_set():
                break
    return totals


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["scrape", "workload", "both"])
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="GLM-5.2")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--interval", type=float, default=5.0,
                    help="scrape period, seconds")
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--phase", action="append", default=[],
                    metavar="FAMILY:SECONDS",
                    help=f"repeatable; families: {', '.join(FAMILIES)}")
    ap.add_argument("--duration", type=float, default=None,
                    help="scrape-only: stop after this many seconds")
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    phases: list[tuple[str, float]] = []
    for spec in args.phase:
        fam, _, secs = spec.partition(":")
        if fam not in FAMILIES:
            ap.error(f"unknown family {fam!r}; have {', '.join(FAMILIES)}")
        phases.append((fam, float(secs)))
    if args.mode in ("workload", "both") and not phases:
        ap.error("--phase is required for workload")

    stop = threading.Event()
    scraper = None
    totals: dict = {}
    if args.mode in ("scrape", "both"):
        scraper = threading.Thread(target=scrape_loop,
                                   args=(args.base, args.out, args.interval, stop),
                                   daemon=True)
        scraper.start()
    try:
        if args.mode in ("workload", "both"):
            totals = workload(args.base, args.model, args.out, phases,
                              args.concurrency, args.max_tokens, stop)
        elif args.duration:
            stop.wait(args.duration)
        else:
            while not stop.is_set():
                stop.wait(3600)
    except KeyboardInterrupt:
        print("interrupted — flushing", flush=True)
    finally:
        stop.set()
        if scraper:
            scraper.join(timeout=args.interval + 5)
    print(f"wrote {args.out}", flush=True)
    # A workload where every request failed still leaves a full, plausible
    # timeline behind: samples, phase rows, a chart with a flat zero line.
    # Say so in the exit status rather than letting the campaign score it as
    # "the model was simply idle".
    if args.mode in ("workload", "both") and not totals.get("ok"):
        print(f"FATAL: no request succeeded ({totals.get('requests', 0)} "
              f"attempted) — the timeline is not evidence of anything",
              flush=True)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
