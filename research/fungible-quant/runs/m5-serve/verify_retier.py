#!/usr/bin/env python3
"""Prove that a forced re-tier MOVES WEIGHTS on a live serve.

This exists because the swap path has twice reported success it did not
deliver. The loop decided 64 swaps and applied none while
``fq_swaps_total`` climbed; the admin API answered "the serve is uniform-K"
on a checkpoint with 75 mixed layers. Both were legible only by checking an
independent surface.

So this tool never trusts the response. It reads ``GET /fq/layer/{L}`` before
and after, and asserts on the difference:

  1. the requested expert really changed tier,
  2. some other expert was really displaced,
  3. per-layer K4 cardinality is unchanged (D1 — a swap is a swap, not a
     promotion),
  4. the serve still generates coherent text afterwards.

(4) matters most. A swap that installs the wrong bytes leaves cardinality
perfect, the API happy, and the model quietly broken — so the check that the
model still works is part of the swap test, not a separate concern.

Usage:
    verify_retier.py --layer 3 --expert 1            # promote e1 to K4
    verify_retier.py --layer 3 --expert 1 --dry-run
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request

PROMPT = "In one sentence, what is mixture-of-experts routing?"


def _req(url: str, payload: dict | None = None, timeout: float = 120.0):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json"} if data else {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read()), r.status
    except urllib.error.HTTPError as e:
        # The admin API returns structured errors with real HTTP codes; the
        # body is the diagnosis, so surface it rather than the status line.
        try:
            return json.loads(e.read()), e.code
        except Exception:  # noqa: BLE001
            return {"error": {"message": str(e)}}, e.code


def layer_tiers(base: str, layer: int) -> tuple[dict[int, int], int]:
    """(expert -> k, declared n_k4) for one layer."""
    doc, _ = _req(f"{base}/fq/layer/{layer}")
    if "experts" not in doc:
        raise SystemExit(f"GET /fq/layer/{layer} unusable: {doc}")
    return ({int(e["expert"]): int(e["k"]) for e in doc["experts"]},
            int(doc.get("n_k4", -1)))


def generate(base: str, model: str) -> str:
    doc, _ = _req(f"{base}/v1/completions", {
        "model": model, "prompt": PROMPT, "max_tokens": 40,
        "temperature": 0})
    try:
        return doc["choices"][0]["text"]
    except (KeyError, IndexError):
        return ""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default="http://127.0.0.1:8100")
    ap.add_argument("--model", default="GLM-5.2")
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--expert", type=int, required=True)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-generate", action="store_true")
    a = ap.parse_args(argv)

    before, n_k4_before = layer_tiers(a.base, a.layer)
    k4_before = {e for e, k in before.items() if k == 4}
    print(f"BEFORE  layer {a.layer}: n_k4={n_k4_before} "
          f"|K4|={len(k4_before)} e{a.expert}=K{before.get(a.expert)}")
    if before.get(a.expert) == a.k:
        print(f"  expert {a.expert} is already K{a.k} — nothing to prove; "
              f"pick one at a different tier")
        return 2

    text_before = "" if a.skip_generate else generate(a.base, a.model)

    resp, status = _req(a.base + "/fq/retier", {
        "items": [{"layer": a.layer, "expert": a.expert, "k": a.k}],
        "mode": "strict_pair",
        "dry_run": bool(a.dry_run),
        "actor": "verify_retier",
        "reason": "prove the swap engine installs weights",
    })
    print(f"POST /fq/retier -> HTTP {status}")
    print(json.dumps(resp, indent=2)[:1200])
    if status >= 400:
        return 1
    if a.dry_run:
        print("\ndry run: no state change expected, not asserting")
        return 0

    after, n_k4_after = layer_tiers(a.base, a.layer)
    k4_after = {e for e, k in after.items() if k == 4}
    entered, left = sorted(k4_after - k4_before), sorted(k4_before - k4_after)
    print(f"\nAFTER   layer {a.layer}: n_k4={n_k4_after} "
          f"|K4|={len(k4_after)} e{a.expert}=K{after.get(a.expert)}")
    print(f"  entered K4: {entered}")
    print(f"  left K4:    {left}")

    ok = True
    if after.get(a.expert) != a.k:
        ok = False
        print(f"FAIL: expert {a.expert} is K{after.get(a.expert)}, "
              f"asked for K{a.k}. The API returned success and the tier map "
              f"did not move.")
    if not left:
        ok = False
        print("FAIL: nothing was displaced. Under fixed cardinality a "
              "promotion must evict someone; an unpaired promotion means the "
              "budget grew silently.")
    if len(k4_after) != len(k4_before) or n_k4_after != n_k4_before:
        ok = False
        print(f"FAIL: D1 violated — |K4| {len(k4_before)}->{len(k4_after)}, "
              f"declared {n_k4_before}->{n_k4_after}.")

    if not a.skip_generate:
        text_after = generate(a.base, a.model)
        print(f"\ngeneration before: {text_before[:110]!r}")
        print(f"generation after : {text_after[:110]!r}")
        if not text_after.strip():
            ok = False
            print("FAIL: the serve stopped generating after the swap.")
        elif len(set(text_after.split())) < 5:
            ok = False
            print("FAIL: output collapsed to near-repetition after the swap "
                  "— the installed bytes are probably wrong.")

    print("\nPASS: the swap installed and the model still works." if ok
          else "\nFAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
