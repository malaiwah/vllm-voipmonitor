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
import json
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
    tpl = FAMILIES[family][i % len(FAMILIES[family])]
    rng = random.Random(hash((family, i)) & 0xFFFFFFFF)
    return tpl.format(a=rng.randint(3, 97), b=rng.randint(101, 499),
                      c=rng.randint(50, 150))


# ------------------------------------------------------------------ scraping
_SAMPLE = re.compile(r'^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)'
                     r'(?:\{(?P<labels>[^}]*)\})?\s+(?P<value>[^\s]+)\s*$')


def parse_metrics(text: str) -> dict[str, float]:
    """Flatten a Prometheus exposition into {name{labels}: value}."""
    out: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        m = _SAMPLE.match(line)
        if not m:
            continue
        key = m["name"] if not m["labels"] else f'{m["name"]}{{{m["labels"]}}}'
        try:
            out[key] = float(m["value"])
        except ValueError:
            continue
    return out


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


def scrape_loop(base: str, out: Path, interval: float, stop: threading.Event,
                phases: list | None = None) -> None:
    prev_tok, prev_t = None, None
    url = base.rstrip("/") + "/metrics"
    with out.open("a") as fh:
        while not stop.is_set():
            t0 = time.time()
            row: dict = {"t": t0, "kind": "sample"}
            try:
                with urllib.request.urlopen(url, timeout=10) as resp:
                    flat = parse_metrics(resp.read().decode())
                row["fq"] = fq_view(flat)
                # vLLM's own decode counter; name has moved across versions,
                # so accept either and record which one we read.
                for name in ("vllm:generation_tokens_total",
                             "vllm:generation_tokens",
                             "vllm:tokens_total"):
                    hit = [v for k, v in flat.items() if k.startswith(name)]
                    if hit:
                        tok = sum(hit)
                        row["gen_tokens_total"] = tok
                        row["gen_tokens_metric"] = name
                        if prev_tok is not None and t0 > prev_t:
                            row["decode_tok_s"] = (tok - prev_tok) / (t0 - prev_t)
                        prev_tok, prev_t = tok, t0
                        break
                run = [v for k, v in flat.items()
                       if k.startswith("vllm:num_requests_running")]
                if run:
                    row["requests_running"] = sum(run)
            except (urllib.error.URLError, OSError, TimeoutError) as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            stop.wait(max(0.0, interval - (time.time() - t0)))


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
             concurrency: int, max_tokens: int, stop: threading.Event) -> None:
    """Run each (family, seconds) phase at fixed concurrency."""
    counter = {"i": 0}
    lock = threading.Lock()
    results: list[dict] = []

    def worker(family: str, deadline: float) -> None:
        while time.time() < deadline and not stop.is_set():
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
            start = time.time()
            fh.write(json.dumps({"t": start, "kind": "phase",
                                 "event": "phase_start", "family": family,
                                 "planned_seconds": seconds,
                                 "concurrency": concurrency}) + "\n")
            fh.flush()
            print(f"[workload] phase {family} for {seconds}s "
                  f"@ concurrency {concurrency}", flush=True)
            deadline = start + seconds
            threads = [threading.Thread(target=worker, args=(family, deadline),
                                        daemon=True)
                       for _ in range(concurrency)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            with lock:
                done = [r for r in results if r["family"] == family]
                ok = [r for r in done if r["ok"]]
                toks = sum(r.get("completion_tokens") or 0 for r in ok)
            fh.write(json.dumps({
                "t": time.time(), "kind": "phase", "event": "phase_end",
                "family": family, "requests": len(done), "ok": len(ok),
                "failed": len(done) - len(ok),
                "completion_tokens": toks,
                "wall_seconds": time.time() - start,
                "tok_s": toks / max(1e-9, time.time() - start),
            }) + "\n")
            fh.flush()
            if stop.is_set():
                break


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
    if args.mode in ("scrape", "both"):
        scraper = threading.Thread(target=scrape_loop,
                                   args=(args.base, args.out, args.interval, stop),
                                   daemon=True)
        scraper.start()
    try:
        if args.mode in ("workload", "both"):
            workload(args.base, args.model, args.out, phases,
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
