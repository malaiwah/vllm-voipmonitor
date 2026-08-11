#!/usr/bin/env python3
"""Sustained concurrent load against an OpenAI-compatible endpoint.

Purpose: hold N requests in flight for D seconds and emit one machine-readable
JSONL row per interval, so a swap timeline (``swap_evidence.py scrape``, which
writes ``fq_*`` gauges on the same epoch clock) can be overlaid on decode
throughput and we can see what live expert swaps cost *while traffic is on*.

Neither of the two off-the-shelf tools gives that timeline:

* ``llm_decode_bench.py`` reports one aggregate number per (concurrency,
  context) cell; the inside of a cell is opaque.
* ``vllm bench serve`` reports one aggregate per run, and importing it needs
  torch + the GG container. This script needs only ``httpx``.

Rows are ``{"kind": "config"|"sample"|"summary", "t": <epoch float>, ...}``.
Every row carries epoch seconds so the join against the scrape JSONL is a
straight numeric merge, no clock translation.

Token accounting prefers the exact server number: the request asks for
``stream_options.continuous_usage_stats`` so every chunk carries a cumulative
``completion_tokens``, and the per-interval count is the sum of the deltas.
If the server does not honour that, the script falls back to counting streamed
content chunks and says so in ``token_source`` — a chunk count is an estimate
(a chunk is not always one token) and must not be reported as exact.
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import random
import signal
import statistics
import sys
import time
from pathlib import Path

import httpx

# Prompt stems chosen to keep the router busy rather than to be interesting.
# Each request gets a distinct index so a prefix cache cannot serve the whole
# run from one entry, which would turn a decode benchmark into a cache test.
STEMS = [
    "Explain, step by step, how to compute the {n}th term of a sequence whose "
    "rule you must first infer from: {a}, {b}, {c}, {d}. Show every step.",
    "Write a Python function that streams {n} sorted iterators lazily and "
    "explain its time and space complexity in detail.",
    "A patient's sodium is {n} mmol/L with normal volume status. Work through "
    "the differential and the correction rate you would choose, and justify it.",
    "Écris une analyse détaillée, en {n} paragraphes, de l'impact de la "
    "quantification des modeles de langue sur l'acces communautaire a l'IA.",
    "Derive the probability that exactly two of three marbles drawn without "
    "replacement from a bag of {a} red and {b} blue are red. Show the algebra.",
    "Review this design: a lock-free SPSC ring buffer of capacity {n} in C11. "
    "Justify each memory ordering choice and name the failure mode it prevents.",
]

FILLER = (
    "context line {i}: token budget padding that carries no instruction and "
    "must be read but not acted upon; ignore its content entirely.\n"
)


def build_prompt(idx: int, prompt_tokens: int) -> str:
    rng = random.Random(idx)
    stem = STEMS[idx % len(STEMS)].format(
        n=7 + (idx % 40),
        a=rng.randint(2, 99), b=rng.randint(2, 99),
        c=rng.randint(2, 99), d=rng.randint(2, 99),
    )
    if prompt_tokens <= 0:
        return stem
    # ~14 tokens per filler line; overshooting is harmless, undershooting is not
    # what we asked for, so round up.
    lines = max(1, prompt_tokens // 14)
    pad = "".join(FILLER.format(i=idx * 100000 + i) for i in range(lines))
    return f"{pad}\n{stem}"


def pct(values: list[float], q: float) -> float | None:
    """Nearest-rank percentile; None for an empty sample."""
    if not values:
        return None
    s = sorted(values)
    k = min(len(s) - 1, max(0, int(round(q / 100.0 * len(s) + 0.5)) - 1))
    return s[k]


class Meter:
    """Everything the interval reporter needs, mutated by the workers."""

    def __init__(self) -> None:
        self.window_tokens = 0
        self.window_completed = 0
        self.window_failed = 0
        self.window_ttft: list[float] = []
        self.window_itl: list[float] = []
        self.window_latency: list[float] = []
        self.inflight = 0
        self.launched = 0
        self.completed = 0
        self.failed = 0
        self.tokens_total = 0
        self.token_source = "unknown"
        self.errors: dict[str, int] = {}

    def drain(self) -> dict:
        out = {
            "tokens": self.window_tokens,
            "completed": self.window_completed,
            "failed": self.window_failed,
            "ttft": self.window_ttft,
            "itl": self.window_itl,
            "latency": self.window_latency,
        }
        self.window_tokens = 0
        self.window_completed = 0
        self.window_failed = 0
        self.window_ttft = []
        self.window_itl = []
        self.window_latency = []
        return out


async def one_request(client: httpx.AsyncClient, url: str, body: dict,
                      meter: Meter, timeout: float) -> None:
    meter.inflight += 1
    meter.launched += 1
    t0 = time.monotonic()
    first_tok = None
    last_tok = None
    n_chunks = 0
    seen_usage = 0
    try:
        async with client.stream("POST", url, json=body, timeout=timeout) as resp:
            if resp.status_code != 200:
                await resp.aread()
                raise httpx.HTTPStatusError(
                    f"HTTP {resp.status_code}", request=resp.request, response=resp)
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if not payload or payload == "[DONE]":
                    continue
                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                usage = chunk.get("usage") or {}
                ct = usage.get("completion_tokens")
                if isinstance(ct, int) and ct >= seen_usage:
                    meter.window_tokens += ct - seen_usage
                    meter.tokens_total += ct - seen_usage
                    seen_usage = ct
                    meter.token_source = "usage"
                for choice in chunk.get("choices") or []:
                    delta = choice.get("delta") or {}
                    # A thinking model streams reasoning_content before content;
                    # both are decoded tokens and both count.
                    if delta.get("content") or delta.get("reasoning_content"):
                        now = time.monotonic()
                        if first_tok is None:
                            first_tok = now
                            # Booked in the window where it happened, not on
                            # request completion: a long request would
                            # otherwise report its TTFT minutes late, in the
                            # wrong window, which is exactly the window we are
                            # trying to line up against a swap event.
                            meter.window_ttft.append((now - t0) * 1000.0)
                        last_tok = now
                        n_chunks += 1
                        if meter.token_source != "usage":
                            meter.window_tokens += 1
                            meter.tokens_total += 1
                            meter.token_source = "chunks"
        meter.completed += 1
        meter.window_completed += 1
        if first_tok is not None and last_tok is not None and n_chunks > 1:
            meter.window_itl.append(
                (last_tok - first_tok) * 1000.0 / (n_chunks - 1))
        meter.window_latency.append((time.monotonic() - t0) * 1000.0)
    except Exception as exc:  # noqa: BLE001 - a failed request is data
        meter.failed += 1
        meter.window_failed += 1
        key = type(exc).__name__
        meter.errors[key] = meter.errors.get(key, 0) + 1
    finally:
        meter.inflight -= 1


async def worker(idx0: int, stride: int, client: httpx.AsyncClient, url: str,
                 args, meter: Meter, stop: asyncio.Event) -> None:
    i = idx0
    while not stop.is_set():
        body = {
            "model": args.model,
            "messages": [{"role": "user",
                          "content": build_prompt(i, args.prompt_tokens)}],
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "stream": True,
            "stream_options": {"include_usage": True,
                               "continuous_usage_stats": True},
        }
        await one_request(client, url, body, meter, args.request_timeout)
        i += stride


def scrape_metrics(base: str, timeout: float) -> dict:
    """Optional server-side cross-check: running requests + generation counter."""
    import re
    import urllib.request
    out: dict = {}
    try:
        with urllib.request.urlopen(base.rstrip("/") + "/metrics",
                                    timeout=timeout) as resp:
            text = resp.read().decode()
    except Exception as exc:  # noqa: BLE001
        return {"metrics_error": f"{type(exc).__name__}: {exc}"}
    for name, key in (("vllm:num_requests_running", "server_running"),
                      ("vllm:num_requests_waiting", "server_waiting"),
                      ("vllm:generation_tokens_total", "server_gen_tokens_total")):
        hits = [float(m.group(1)) for m in
                re.finditer(rf"^{re.escape(name)}(?:\{{[^}}]*\}})? ([0-9.eE+-]+)$",
                            text, re.M)]
        if hits:
            out[key] = sum(hits)
    return out


async def run(args) -> int:
    url = args.base_url.rstrip("/") + "/v1/chat/completions"
    meter = Meter()
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(sig, stop.set)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fh = out.open("a")

    def emit(row: dict) -> None:
        fh.write(json.dumps(row) + "\n")
        fh.flush()

    t_start = time.time()
    emit({
        "kind": "config", "t": t_start,
        "iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t_start)),
        "base_url": args.base_url, "model": args.model,
        "concurrency": args.concurrency, "duration_s": args.duration,
        "interval_s": args.interval, "warmup_s": args.warmup,
        "max_tokens": args.max_tokens, "prompt_tokens": args.prompt_tokens,
        "temperature": args.temperature,
    })

    limits = httpx.Limits(max_connections=args.concurrency + 8,
                          max_keepalive_connections=args.concurrency + 8)
    headers = {"Authorization": f"Bearer {args.api_key}"} if args.api_key else {}
    prev_gen = None
    samples: list[dict] = []
    async with httpx.AsyncClient(limits=limits, headers=headers) as client:
        tasks = [asyncio.create_task(
            worker(i, args.concurrency, client, url, args, meter, stop))
            for i in range(args.concurrency)]
        deadline = time.monotonic() + args.duration
        last = time.monotonic()
        try:
            while not stop.is_set() and time.monotonic() < deadline:
                await asyncio.wait([asyncio.create_task(stop.wait())],
                                   timeout=args.interval)
                now_m = time.monotonic()
                now_w = time.time()
                window = now_m - last
                last = now_m
                w = meter.drain()
                elapsed = now_w - t_start
                row = {
                    "kind": "sample", "t": now_w,
                    "iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now_w)),
                    "elapsed_s": round(elapsed, 3),
                    "window_s": round(window, 3),
                    "warmup": elapsed < args.warmup,
                    "window_output_tokens": w["tokens"],
                    "window_tok_s": round(w["tokens"] / window, 2) if window else None,
                    "window_completed": w["completed"],
                    "window_failed": w["failed"],
                    "inflight": meter.inflight,
                    "launched_total": meter.launched,
                    "completed_total": meter.completed,
                    "failed_total": meter.failed,
                    "output_tokens_total": meter.tokens_total,
                    "token_source": meter.token_source,
                    "ttft_ms": {"n": len(w["ttft"]),
                                "p50": pct(w["ttft"], 50),
                                "p90": pct(w["ttft"], 90),
                                "p99": pct(w["ttft"], 99)},
                    "itl_ms": {"p50": pct(w["itl"], 50), "p90": pct(w["itl"], 90)},
                    "latency_ms": {"p50": pct(w["latency"], 50),
                                   "p90": pct(w["latency"], 90)},
                    "errors": dict(meter.errors),
                }
                if args.metrics:
                    srv = scrape_metrics(args.base_url, 5.0)
                    gen = srv.get("server_gen_tokens_total")
                    if gen is not None and prev_gen is not None and window > 0:
                        srv["server_window_tok_s"] = round((gen - prev_gen) / window, 2)
                    if gen is not None:
                        prev_gen = gen
                    row.update(srv)
                emit(row)
                samples.append(row)
                if args.echo:
                    print(f"[{row['elapsed_s']:8.1f}s] {row['window_tok_s']!s:>9} tok/s  "
                          f"inflight={row['inflight']:3d} done={row['completed_total']:5d} "
                          f"fail={row['failed_total']:3d} "
                          f"ttft_p50={row['ttft_ms']['p50']}", flush=True)
        finally:
            stop.set()
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    measured = [s for s in samples if not s["warmup"]]
    tok = sum(s["window_output_tokens"] for s in measured)
    secs = sum(s["window_s"] for s in measured)
    ttfts = [s["ttft_ms"]["p50"] for s in measured if s["ttft_ms"]["p50"] is not None]
    summary = {
        "kind": "summary", "t": time.time(),
        "wall_seconds": round(time.time() - t_start, 2),
        "measured_seconds": round(secs, 2),
        "measured_output_tokens": tok,
        "mean_decode_tok_s": round(tok / secs, 2) if secs else None,
        "min_window_tok_s": min((s["window_tok_s"] for s in measured
                                 if s["window_tok_s"] is not None), default=None),
        "max_window_tok_s": max((s["window_tok_s"] for s in measured
                                 if s["window_tok_s"] is not None), default=None),
        "median_window_ttft_p50_ms": round(statistics.median(ttfts), 1) if ttfts else None,
        "requests_launched": meter.launched,
        "requests_completed": meter.completed,
        "requests_failed": meter.failed,
        "token_source": meter.token_source,
        "errors": dict(meter.errors),
        "samples": len(samples),
        "note": ("token counts are exact server usage deltas"
                 if meter.token_source == "usage"
                 else "ESTIMATE: server did not stream continuous usage; "
                      "tokens counted as streamed chunks"),
    }
    emit(summary)
    fh.close()
    print(json.dumps(summary, indent=2))
    return 0 if meter.completed else 1


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="GLM-5.2")
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--duration", type=float, default=600.0,
                    help="seconds of load, including --warmup")
    ap.add_argument("--interval", type=float, default=5.0,
                    help="seconds per reported window")
    ap.add_argument("--warmup", type=float, default=30.0,
                    help="leading seconds excluded from the summary (rows are "
                         "still written, flagged warmup=true)")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--prompt-tokens", type=int, default=0,
                    help="approximate synthetic input padding per request")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--request-timeout", type=float, default=900.0)
    ap.add_argument("--metrics", action="store_true",
                    help="also scrape /metrics each interval for a server-side "
                         "cross-check of tok/s and in-flight count")
    ap.add_argument("--echo", action="store_true", help="print each window row")
    ap.add_argument("--out", required=True, help="output JSONL path")
    args = ap.parse_args()
    if args.concurrency < 1:
        ap.error("--concurrency must be >= 1")
    return asyncio.run(run(args))


if __name__ == "__main__":
    sys.exit(main())
