#!/usr/bin/env python3
"""Does a LANGUAGE switch move the expert set where a domain switch did not?

Motivation. Under sustained load, the desired K4 set churns ~39% between
adjacent intervals, and a math -> code phase change produced no visible dip at
all: the within-domain noise swamped the between-domain signal. But math and
code are both ENGLISH. A multilingual MoE is expected to route differently for
a different script, most strongly in the layers nearest the embedding and the
output head, where token identity still dominates the representation.

So this is the decisive contrast. If Chinese moves the expert set, the earlier
null result means "English-domain contrasts are too weak to see at interval
scale", not "this router has no exploitable structure". If Chinese moves it no
more than noise does, the flatness conclusion stands on much firmer ground.

Method, deliberately paired:

  reset counters -> drive corpus A -> capture -> reset -> drive corpus B ->
  capture -> per-layer Jaccard of the top-K sets

Both arms hit the same serve, same concurrency, same token budget, back to
back. The comparison is A-vs-B on one model state, not two runs remembered
apart.

A per-layer curve is reported rather than one number, because the hypothesis
is specifically about DEPTH: language effects should show at the edges and
wash out in the middle. A single average would hide exactly the structure
being looked for.
"""
from __future__ import annotations

import argparse
import base64
import json
import random
import struct
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path


def _post(url: str, payload: dict | None, timeout: float = 300.0):
    data = None if payload is None else json.dumps(payload).encode()
    hdrs = {"Content-Type": "application/json"} if data else {}
    # The heatmap body is ~2.4x larger without this, and the endpoint says so
    # in its own warnings array.
    hdrs["Accept-Encoding"] = "gzip"
    req = urllib.request.Request(url, data=data, headers=hdrs)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = r.read()
            if r.headers.get("Content-Encoding") == "gzip":
                import gzip as _gz
                body = _gz.decompress(body)
            return json.loads(body)
    except urllib.error.HTTPError as e:
        try:
            return json.loads(e.read())
        except Exception:  # noqa: BLE001
            return {"error": str(e)}
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}


def _bf16(raw: bytes) -> list[float]:
    return [struct.unpack("<f", bytes((0, 0, raw[i], raw[i + 1])))[0]
            for i in range(0, len(raw) - 1, 2)]


def decode(doc: dict, field: str) -> list[list[float]]:
    blob = doc.get(field)
    if not blob:
        return []
    raw = base64.b64decode(blob)
    flat = (list(raw) if (doc.get("encoding") or {}).get(field) == "u8"
            else _bf16(raw))
    n_l, n_e = int(doc["num_layers"]), int(doc["num_experts"])
    return [flat[r * n_e:(r + 1) * n_e] for r in range(n_l)]


def drive(base: str, model: str, prompts: list[str], seconds: float,
          concurrency: int, max_tokens: int) -> int:
    """Hammer the serve with `prompts` for `seconds`. Returns request count."""
    stop = time.time() + seconds
    done = [0]
    lock = threading.Lock()

    def worker(seed: int):
        rng = random.Random(seed)
        while time.time() < stop:
            p = rng.choice(prompts)
            r = _post(f"{base}/v1/completions", {
                "model": model, "prompt": p,
                "max_tokens": max_tokens, "temperature": 0.7,
                "seed": rng.randint(0, 1 << 30)}, timeout=180)
            if "choices" in r:
                with lock:
                    done[0] += 1

    ts = [threading.Thread(target=worker, args=(i,), daemon=True)
          for i in range(concurrency)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(timeout=seconds + 240)
    return done[0]


def top_sets(counts: list[list[float]], k: int) -> list[set[int]]:
    out = []
    for row in counts:
        order = sorted(range(len(row)), key=lambda i: (-row[i], i))
        out.append(set(order[:k]))
    return out


def jaccard(a: set, b: set) -> float:
    return len(a & b) / max(len(a | b), 1)


def arm(base: str, model: str, name: str, prompts: list[str], secs: float,
        conc: int, mt: int, outdir: Path) -> dict:
    print(f"\n=== {name}: resetting counters ===")
    print("  ", json.dumps(_post(f"{base}/fq/heatmap/reset", {}))[:120])
    print(f"  driving {secs:.0f}s @ concurrency {conc} "
          f"({len(prompts)} prompts)...")
    n = drive(base, model, prompts, secs, conc, mt)
    doc = _post(f"{base}/fq/heatmap", None)
    p = outdir / f"lang-{name}.json"
    p.write_text(json.dumps(doc))
    counts = decode(doc, "count")
    mass = sum(sum(r) for r in counts)
    print(f"  {n} requests, step={doc.get('step')}, mass={mass:,.0f} -> {p}")
    return {"name": name, "doc": doc, "counts": counts, "requests": n,
            "mass": mass}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default="http://127.0.0.1:8100")
    ap.add_argument("--model", default="GLM-5.2")
    ap.add_argument("--zh", type=Path,
                    default=Path("/home/mbelleau/zh-corpus/zh_prompts.json"))
    ap.add_argument("--seconds", type=float, default=300)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=96)
    ap.add_argument("--topk", type=int, default=26)
    ap.add_argument("--out", type=Path, default=Path("results/bt/lang"))
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)

    zh = json.loads(a.zh.read_text())
    try:
        import swap_evidence as SE
        en = [SE.render_prompt("math", i) for i in range(400)]
    except Exception:  # noqa: BLE001 — fall back to inline English prompts
        en = [f"Explain step by step why {i} is or is not a prime number, "
              f"then summarise the reasoning." for i in range(400)]
    print(f"EN prompts: {len(en)}   ZH prompts: {len(zh)}")

    a_en = arm(a.base, a.model, "en", en, a.seconds, a.concurrency,
               a.max_tokens, a.out)
    a_zh = arm(a.base, a.model, "zh", zh, a.seconds, a.concurrency,
               a.max_tokens, a.out)

    if not a_en["counts"] or not a_zh["counts"]:
        print("\nno activation matrix captured — cannot compare")
        return 1

    layers = a_en["doc"]["layers"]
    s_en = top_sets(a_en["counts"], a.topk)
    s_zh = top_sets(a_zh["counts"], a.topk)
    js = [jaccard(x, y) for x, y in zip(s_en, s_zh)]

    # Chance floor for two independent top-k draws from n experts.
    n_e = int(a_en["doc"]["num_experts"])
    chance = a.topk / (2 * n_e - a.topk)

    print(f"\n{'layer':>6}  {'jaccard':>8}  EN∩ZH of top-{a.topk}")
    for lid, j in zip(layers, js):
        bar = "#" * int(j * 40)
        print(f"{lid:>6}  {j:>8.3f}  {bar}")

    third = max(1, len(js) // 3)
    lo, mid, hi = js[:third], js[third:2 * third], js[2 * third:]
    m = lambda v: sum(v) / max(len(v), 1)  # noqa: E731
    print(f"\nchance floor (independent draws): {chance:.3f}")
    print(f"mean jaccard EN vs ZH: {m(js):.3f}")
    print(f"  lower third  (layers {layers[0]}-{layers[third-1]}): "
          f"{m(lo):.3f}")
    print(f"  middle third (layers {layers[third]}-{layers[2*third-1]}): "
          f"{m(mid):.3f}")
    print(f"  upper third  (layers {layers[2*third]}-{layers[-1]}): "
          f"{m(hi):.3f}")

    # Test a MONOTONE gradient as well as a U-shape. The first version only
    # asked "edges vs middle" and reported NO DEPTH STRUCTURE on data whose
    # divergence falls steadily from 0.373 at layer 3 to 0.256 at layer 77 —
    # a real and strong effect, invisible to a test shaped like the wrong
    # hypothesis.
    import math as _math

    def _rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        out = [0] * len(v)
        for pos, i in enumerate(order):
            out[i] = pos
        return out

    _n = len(js)
    _rd, _rj = _rank(list(range(_n))), _rank(js)
    _md, _mj = sum(_rd) / _n, sum(_rj) / _n
    _num = sum((_rd[i] - _md) * (_rj[i] - _mj) for i in range(_n))
    _den = _math.sqrt(sum((_rd[i] - _md) ** 2 for i in range(_n))
                      * sum((_rj[i] - _mj) ** 2 for i in range(_n)))
    rho = _num / _den if _den else 0.0
    print(f"Spearman(depth, jaccard) = {rho:+.4f}")
    if rho <= -0.25:
        print(f"DEPTH GRADIENT: divergence RISES with depth (rho {rho:+.3f}) — "
              f"deeper layers are more corpus-specific.")
    elif rho >= 0.25:
        print(f"DEPTH GRADIENT: divergence FALLS with depth (rho {rho:+.3f}) — "
              f"shallow layers carry more of the difference.")

    edges, centre = m(lo + hi), m(mid)
    if edges < centre - 0.02:
        print(f"\nEDGES DIVERGE MORE than the middle "
              f"({edges:.3f} vs {centre:.3f}) — consistent with language "
              f"sensitivity concentrated near the embedding and the head.")
    elif centre < edges - 0.02:
        print(f"\nMIDDLE diverges more than the edges "
              f"({centre:.3f} vs {edges:.3f}) — NOT the expected signature; "
              f"whatever separates these corpora is not depth-localised.")
    else:
        print(f"\nNO DEPTH STRUCTURE: edges {edges:.3f} vs middle "
              f"{centre:.3f}. The language contrast is uniform across depth.")

    json.dump({"layers": layers, "jaccard": js, "chance": chance,
               "topk": a.topk,
               "requests": {"en": a_en["requests"], "zh": a_zh["requests"]},
               "mass": {"en": a_en["mass"], "zh": a_zh["mass"]}},
              open(a.out / "lang-contrast.json", "w"), indent=1)
    print(f"\nwrote {a.out / 'lang-contrast.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
