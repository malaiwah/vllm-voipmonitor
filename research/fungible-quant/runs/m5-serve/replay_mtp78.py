#!/usr/bin/env python3
"""Replay the MTP78 calibration corpus against a live serve.

This is the traffic the convergence experiment is actually about. The
reference quant `willfalco/GLM-5.2-EXL3-TR3-3.42bpw` was built from
`reap_recall_calib.jsonl`; if our loop is going to rediscover that human's
expert choices, it has to see the same distribution the human's calibration
saw. The earlier synthetic math+code mix scored 0.3603 — a lower bound
measured on the wrong corpus.

Two things this does that a generic load driver does not:

* **Prompts come from the corpus verbatim**, in a deterministic order, with
  the corpus sha256 recorded in the output so a run can be tied to the exact
  bytes it replayed.
* **Axis filtering.** The corpus has four equal axes (general, legal,
  code_agentic, reasoning_termination). The reference is the *Coder* variant,
  so scoring the code axis separately from the whole corpus is the sharper
  test — and if the code axis converges better than the full mix, that is
  evidence the effect is real rather than an artifact of volume.

Generation is deliberately short (`--max-tokens` default 8). We are measuring
ROUTING, not output quality: every prompt token is routed during prefill, so
long generations buy little signal per unit of GPU time.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

HARNESS = Path(__file__).parent / "harness" / "load_mtp78_corpus.py"


def load_corpus(axis: str | None, limit: int | None) -> tuple[list[str], dict]:
    """Shell out to the harness loader so there is ONE corpus reader."""
    cmd = [sys.executable, str(HARNESS), "--json"]
    if axis:
        cmd += ["--axis", axis]
    if limit:
        cmd += ["--limit", str(limit)]
    meta = json.loads(subprocess.run(cmd, capture_output=True, text=True,
                                     check=True).stdout)
    cmd_show = [sys.executable, str(HARNESS), "--show"]
    if axis:
        cmd_show += ["--axis", axis]
    if limit:
        cmd_show += ["--limit", str(limit)]
    out = subprocess.run(cmd_show, capture_output=True, text=True, check=True)
    prompts = [l for l in out.stdout.splitlines() if l.strip()]
    if not prompts:
        raise SystemExit("corpus loader yielded no prompts — refusing to "
                         "replay an empty corpus")
    return prompts, meta


def one(base: str, model: str, prompt: str, max_tokens: int) -> dict:
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }).encode()
    req = urllib.request.Request(
        base.rstrip("/") + "/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            d = json.loads(r.read().decode())
        u = d.get("usage", {})
        return {"ok": True, "dt": time.time() - t0,
                "prompt_tokens": u.get("prompt_tokens"),
                "completion_tokens": u.get("completion_tokens")}
    except Exception as exc:  # noqa: BLE001 — a failure is data
        return {"ok": False, "dt": time.time() - t0,
                "error": f"{type(exc).__name__}: {exc}"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="GLM-5.2")
    ap.add_argument("--axis", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--max-tokens", type=int, default=8)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    prompts, meta = load_corpus(a.axis, a.limit)
    print(f"corpus {meta.get('corpus')} sha={str(meta.get('sha256'))[:12]} "
          f"sha_ok={meta.get('sha256_ok')} axis={a.axis or 'ALL'} "
          f"prompts={len(prompts)}", flush=True)
    if meta.get("sha256_ok") is False:
        raise SystemExit("corpus sha256 MISMATCH — refusing to replay; the "
                         "whole point is replaying the exact reference bytes")

    results: list[dict] = []
    lock = threading.Lock()
    idx = {"i": 0}
    t_start = time.time()

    def worker() -> None:
        while True:
            with lock:
                i = idx["i"]
                idx["i"] += 1
            if i >= len(prompts):
                return
            r = one(a.base, a.model, prompts[i], a.max_tokens)
            with lock:
                results.append(r)
                n = len(results)
            if n % 250 == 0:
                el = time.time() - t_start
                print(f"  {n}/{len(prompts)} in {el:.0f}s "
                      f"({n/max(el,1e-9):.1f} req/s)", flush=True)

    threads = [threading.Thread(target=worker, daemon=True)
               for _ in range(a.concurrency)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    ok = [r for r in results if r["ok"]]
    wall = time.time() - t_start
    summary = {
        "schema": "fq-corpus-replay/1",
        "corpus": meta.get("corpus"),
        "corpus_sha256": meta.get("sha256"),
        "axis": a.axis or "ALL",
        "prompts": len(prompts),
        "ok": len(ok),
        "failed": len(results) - len(ok),
        "wall_seconds": round(wall, 1),
        "prompt_tokens": sum(r.get("prompt_tokens") or 0 for r in ok),
        "completion_tokens": sum(r.get("completion_tokens") or 0 for r in ok),
        "concurrency": a.concurrency,
        "max_tokens": a.max_tokens,
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    if not ok:
        print("FATAL: every request failed", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
