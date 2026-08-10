#!/usr/bin/env python3
"""traffic_bench — deterministic varied-prompt traffic + decode throughput.

Stdlib only. Fires N greedy completion requests from a fixed prompt set
through C worker threads and reports aggregate decode throughput
(completion tokens / wall seconds between first send and last done) plus
per-request records. The prompt sequence is a pure function of (--n,
--seed), so A/B legs see byte-identical traffic.

Usage:
  traffic_bench.py --n 160 --concurrency 4 --max-tokens 96 \
      --out run.json [--endpoint http://127.0.0.1:8801] [--model fruit-mixed]
"""
from __future__ import annotations

import argparse
import json
import queue
import threading
import time
import urllib.request

TOPICS = [
    "the physics of glacier flow", "a recipe for sourdough rye bread",
    "the history of the telegraph", "how B-trees balance themselves",
    "the economics of container shipping", "a walk through a cedar forest",
    "the life cycle of a cicada", "tuning a two-stroke engine",
    "the geometry of soap bubbles", "medieval water mills",
    "how sonar mapping works", "the chemistry of rust",
    "training a sheepdog", "the structure of a fugue",
    "desert irrigation systems", "the design of suspension bridges",
    "reading nautical charts", "the metallurgy of Damascus steel",
    "pollination networks in meadows", "the mechanics of a piano action",
    "cold-water diving physiology", "the logistics of beekeeping",
    "printing with movable type", "the acoustics of concert halls",
]
STYLES = [
    "Explain {t} to a curious teenager.",
    "Write a technical summary of {t}.",
    "Describe {t} as a field journal entry.",
    "List the five most important facts about {t}, with reasons.",
    "Tell a short story that hinges on {t}.",
]


def prompts(n: int, seed: int) -> list[str]:
    out = []
    for i in range(n):
        t = TOPICS[(seed + i) % len(TOPICS)]
        s = STYLES[(seed + i // len(TOPICS)) % len(STYLES)]
        out.append(f"[req {i}] " + s.format(t=t))
    return out


def post(url: str, body: dict, timeout: float):
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, json.loads(r.read())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default="http://127.0.0.1:8801")
    ap.add_argument("--model", default="fruit-mixed")
    ap.add_argument("--n", type=int, default=160)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=96)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    q: queue.Queue = queue.Queue()
    for i, p in enumerate(prompts(args.n, args.seed)):
        q.put((i, p))
    records: list[dict] = []
    lock = threading.Lock()
    url = args.endpoint.rstrip("/") + "/v1/completions"

    def worker():
        while True:
            try:
                i, prompt = q.get_nowait()
            except queue.Empty:
                return
            t0 = time.perf_counter()
            try:
                status, res = post(url, {
                    "model": args.model, "prompt": prompt,
                    "max_tokens": args.max_tokens, "temperature": 0.0,
                }, args.timeout)
                usage = res.get("usage", {})
                rec = {"i": i, "status": status,
                       "latency_s": round(time.perf_counter() - t0, 3),
                       "prompt_tokens": usage.get("prompt_tokens"),
                       "completion_tokens": usage.get("completion_tokens")}
            except Exception as e:  # noqa: BLE001 — record and continue
                rec = {"i": i, "status": f"error:{type(e).__name__}",
                       "latency_s": round(time.perf_counter() - t0, 3)}
            with lock:
                records.append(rec)

    t_start = time.perf_counter()
    threads = [threading.Thread(target=worker, daemon=True)
               for _ in range(args.concurrency)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - t_start

    ok = [r for r in records if r["status"] == 200]
    completion = sum(r["completion_tokens"] or 0 for r in ok)
    prompt_toks = sum(r["prompt_tokens"] or 0 for r in ok)
    lats = sorted(r["latency_s"] for r in ok)
    summary = {
        "tag": args.tag, "n": args.n, "concurrency": args.concurrency,
        "max_tokens": args.max_tokens, "seed": args.seed,
        "ok": len(ok), "errors": len(records) - len(ok),
        "wall_s": round(wall, 3),
        "completion_tokens": completion, "prompt_tokens": prompt_toks,
        "decode_tok_per_s": round(completion / wall, 2),
        "latency_p50_s": lats[len(lats) // 2] if lats else None,
    }
    print(json.dumps(summary))
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"summary": summary, "records":
                       sorted(records, key=lambda r: r["i"])}, f, indent=1)
    return 0 if summary["errors"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
