#!/usr/bin/env python3
"""fq_probe — held-out probe set + teacher-forced logprob capture.

Builds the fixed-seed probe set (06-decisions: ~32 held-out prompts from
the 4-axis corpus, excluding rows used by the capture plan) and records
per-token prompt logprobs from a running OpenAI-compatible serve. The
stored logprobs are the reference leg for KLD comparisons (M2 probe, T9
ladder, 0d): a later policy/checkpoint is scored by re-running the same
probe and comparing distributions/logprobs on identical tokens.

Usage: fq_probe.py --corpus <jsonl> --plan <capture_plan.json> \
                   --endpoint http://127.0.0.1:8000 --model glm52-k3 \
                   --out <dir> [--n 32] [--seed 20260810] [--max-tokens 768]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import urllib.request
from pathlib import Path


def build_probe_set(corpus: Path, plan: Path | None, n: int, seed: int,
                    max_chars: int = 6000) -> list[dict]:
    rows = []
    with open(corpus) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if line:
                rows.append((i, line))
    used: set[int] = set()
    if plan and plan.exists():
        doc = json.loads(plan.read_text())
        for p in doc.get("passes", []):
            for s in p.get("samples", []):
                if isinstance(s, dict) and "row" in s:
                    used.add(int(s["row"]))
                elif isinstance(s, int):
                    used.add(s)
    pool = [r for r in rows if r[0] not in used]
    rng = random.Random(seed)
    picks = rng.sample(pool, n)
    out = []
    for row_idx, raw in sorted(picks):
        d = json.loads(raw)
        text = d.get("text") or d.get("content") or d.get("prompt") or ""
        out.append({"row": row_idx, "text": text[:max_chars],
                    "sha256": hashlib.sha256(text[:max_chars].encode()).hexdigest()})
    return out


def capture_logprobs(endpoint: str, model: str, probe: list[dict],
                     max_tokens: int) -> list[dict]:
    results = []
    for i, p in enumerate(probe):
        body = {"model": model, "prompt": p["text"], "max_tokens": 0,
                "echo": True, "logprobs": 0, "temperature": 0.0}
        req = urllib.request.Request(
            f"{endpoint}/v1/completions", data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as r:
            out = json.load(r)
        lp = out["choices"][0]["logprobs"]
        results.append({
            "row": p["row"], "prompt_sha256": p["sha256"],
            "tokens": lp.get("tokens"),
            "token_logprobs": lp.get("token_logprobs"),
        })
        print(f"probe {i+1}/{len(probe)}: {len(lp.get('tokens') or [])} tokens",
              flush=True)
    return results


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--plan", type=Path, default=None)
    ap.add_argument("--endpoint", default="http://127.0.0.1:8000")
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--seed", type=int, default=20260810)
    ap.add_argument("--max-tokens", type=int, default=768)
    ap.add_argument("--tag", default=None, help="label for this reference run")
    args = ap.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    probe = build_probe_set(args.corpus, args.plan, args.n, args.seed)
    (args.out / "probe_set.json").write_text(json.dumps(
        {"seed": args.seed, "n": args.n,
         "corpus_sha256": hashlib.sha256(args.corpus.read_bytes()).hexdigest(),
         "prompts": probe}, indent=1))
    tag = args.tag or args.model
    refs = capture_logprobs(args.endpoint, args.model, probe, args.max_tokens)
    ref_doc = {"model": args.model, "tag": tag,
               "mean_logprob": sum(
                   sum(x for x in r["token_logprobs"] if x is not None)
                   / max(1, len([x for x in r["token_logprobs"] if x is not None]))
                   for r in refs) / max(1, len(refs)),
               "results": refs}
    (args.out / f"logprobs-{tag}.json").write_text(json.dumps(ref_doc, indent=1))
    print(f"probe reference '{tag}': {len(refs)} prompts, "
          f"mean token logprob {ref_doc['mean_logprob']:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
