#!/usr/bin/env python3
"""Phase 0a — expert-routing stability analysis on the real GLM-5.2 layer-78 capture.

Data: hf.co/datasets/malaiwah/GLM-5.2-MTP78-calibration-capture
  3 safetensors shards, each holding  x: BF16 [rows, 6144]  (NOT downloaded)
                                and  ids: U8  [rows, 8]    (range-read only, 58.3 MB total)
  7,288,310 tokens total, top-8 routing over 256 experts, layer prefix model.layers.78.mlp.

Technique: 8-byte LE header length + JSON safetensors header via HTTP range reads
(same as poc_slice.py), then a single coalesced range read per shard covering only
the `ids` tensor.  Shards are concatenated in `row_offset` order.

Analysis (kill-criterion K3, PLAN.md section 7: tau > 0.9 => allocation stable):
  (a) adjacent-window stability: Jaccard of the top-108 expert set + Kendall tau-b
      of the full 256-expert count ranking, between consecutive windows (~100 windows);
  (b) each window vs the global (full-corpus) ranking, same two metrics;
  (c) cumulative convergence: top-108/tau of the first X% of the stream vs the full
      corpus, X = 5..100 step 5;
  (d) concentration: Gini, entropy, top-k mass shares of the expert distribution.

Pure numpy (no scipy/pandas on this box).  CPU-only, no GPU, no vLLM.
"""

import json
import os
import struct
import subprocess
import sys

import numpy as np

REPO = "malaiwah/GLM-5.2-MTP78-calibration-capture"
BASE = f"https://huggingface.co/datasets/{REPO}/resolve/main"
SHARDS = [f"capture-{i:05d}-of-00003.safetensors" for i in (1, 2, 3)]
CACHE = os.environ.get("FQ_0A_CACHE", os.path.expanduser("~/.cache/fq_phase0a"))
N_WINDOWS = 100
TOP_N = 108  # K4-tier population per layer at the 3.42 bpw operating point (PLAN.md sec. "N = 108")
N_EXPERTS = 256


# ---------------------------------------------------------------- data acquisition
def rget(url, a, b, out):
    subprocess.run(
        ["curl", "-sSL", "--max-time", "590", "-r", f"{a}-{b}", url, "-o", out],
        check=True,
    )


def fetch_ids():
    """Range-read only the `ids` tensors; returns uint8 [N, 8] in row_offset order."""
    os.makedirs(CACHE, exist_ok=True)
    npy = os.path.join(CACHE, "ids_full.npy")
    if os.path.exists(npy):
        return np.load(npy)
    parts = []
    for shard in SHARDS:
        url = f"{BASE}/{shard}"
        l8 = os.path.join(CACHE, shard + ".len8")
        rget(url, 0, 7, l8)
        hlen = struct.unpack("<Q", open(l8, "rb").read())[0]
        hj = os.path.join(CACHE, shard + ".hdr.json")
        rget(url, 8, 7 + hlen, hj)
        hdr = json.load(open(hj))
        base = 8 + hlen
        meta = hdr["__metadata__"]
        t = hdr["ids"]
        assert t["dtype"] == "U8" and t["shape"][1] == 8, t
        a, b = t["data_offsets"]
        blob = os.path.join(CACHE, shard + ".ids.bin")
        if not (os.path.exists(blob) and os.path.getsize(blob) == b - a):
            rget(url, base + a, base + b - 1, blob)
        arr = np.fromfile(blob, dtype=np.uint8).reshape(t["shape"])
        parts.append((int(meta["row_offset"]), arr))
        print(f"  {shard}: rows={t['shape'][0]} row_offset={meta['row_offset']} "
              f"ids bytes={b - a}", file=sys.stderr)
    parts.sort(key=lambda p: p[0])
    ids = np.concatenate([p[1] for p in parts])
    np.save(npy, ids)
    return ids


# ---------------------------------------------------------------- metrics
def kendall_tau_b(x, y):
    """Kendall tau-b between two same-length score vectors (ties handled). O(n^2), fine for n=256."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    dx = np.sign(x[:, None] - x[None, :])
    dy = np.sign(y[:, None] - y[None, :])
    iu = np.triu_indices(len(x), k=1)
    s = float(np.sum(dx[iu] * dy[iu]))
    n0 = len(iu[0])
    tx = n0 - float(np.sum(dx[iu] != 0))
    ty = n0 - float(np.sum(dy[iu] != 0))
    denom = np.sqrt((n0 - tx) * (n0 - ty))
    return s / denom if denom else float("nan")


def top_set(counts, n=TOP_N):
    return set(np.argsort(counts)[::-1][:n].tolist())


def jaccard(a, b):
    return len(a & b) / len(a | b)


def gini(counts):
    x = np.sort(np.asarray(counts, dtype=np.float64))
    n = len(x)
    return float((2 * np.arange(1, n + 1) - n - 1) @ x / (n * x.sum()))


def counts_of(ids_slice):
    return np.bincount(ids_slice.ravel(), minlength=N_EXPERTS).astype(np.int64)


# ---------------------------------------------------------------- analysis
def main():
    print("fetching ids ...", file=sys.stderr)
    ids = fetch_ids()
    n_tok = ids.shape[0]
    print(f"tokens={n_tok}  topk={ids.shape[1]}  experts_seen={len(np.unique(ids))}")

    windows = np.array_split(ids, N_WINDOWS)
    wc = np.stack([counts_of(w) for w in windows])          # [100, 256]
    gc = wc.sum(axis=0)                                     # global counts
    g_top = top_set(gc)

    # (a) adjacent windows
    adj_j = np.array([jaccard(top_set(wc[i]), top_set(wc[i + 1]))
                      for i in range(N_WINDOWS - 1)])
    adj_t = np.array([kendall_tau_b(wc[i], wc[i + 1]) for i in range(N_WINDOWS - 1)])

    # (b) window vs global
    glob_j = np.array([jaccard(top_set(wc[i]), g_top) for i in range(N_WINDOWS)])
    glob_t = np.array([kendall_tau_b(wc[i], gc) for i in range(N_WINDOWS)])

    def stat(v):
        return (f"mean={v.mean():.4f} median={np.median(v):.4f} min={v.min():.4f} "
                f"p5={np.percentile(v, 5):.4f} max={v.max():.4f}")

    print(f"\n(a) adjacent windows ({N_WINDOWS} windows of ~{n_tok // N_WINDOWS} tokens)")
    print(f"    top-{TOP_N} Jaccard : {stat(adj_j)}")
    print(f"    Kendall tau-b     : {stat(adj_t)}")
    print(f"    worst adjacent pairs (tau): "
          f"{[(int(i), round(float(adj_t[i]), 3)) for i in np.argsort(adj_t)[:5]]}")

    print(f"\n(b) window vs global ranking")
    print(f"    top-{TOP_N} Jaccard : {stat(glob_j)}")
    print(f"    Kendall tau-b     : {stat(glob_t)}")
    frac = (glob_t > 0.9).mean()
    print(f"    windows with tau > 0.9 vs global: {frac * 100:.0f}%")

    # membership churn of the top-N set
    in_all = set(range(N_EXPERTS))
    union = set()
    for i in range(N_WINDOWS):
        s = top_set(wc[i])
        in_all &= s
        union |= s
    print(f"    experts in top-{TOP_N} of EVERY window: {len(in_all)}; "
          f"in top-{TOP_N} of ANY window: {len(union)}")

    # (c) cumulative convergence
    print(f"\n(c) cumulative convergence: first X% vs full corpus")
    print("    X%   tokens     top-108-Jaccard   tau-b(full 256)")
    cum = np.cumsum(wc, axis=0)
    for X in range(5, 101, 5):
        k = max(1, round(N_WINDOWS * X / 100)) - 1
        cc = cum[k]
        print(f"    {X:3d}  {int(cc.sum() // 8):8d}      {jaccard(top_set(cc), g_top):.4f}"
              f"          {kendall_tau_b(cc, gc):.4f}")

    # (d) concentration / skew
    f = gc / gc.sum()
    fs = np.sort(f)[::-1]
    ent = -np.sum(f[f > 0] * np.log2(f[f > 0]))
    print(f"\n(d) global expert distribution ({N_EXPERTS} experts, uniform = {1 / N_EXPERTS:.5f})")
    print(f"    Gini={gini(gc):.4f}  entropy={ent:.3f} bits (max {np.log2(N_EXPERTS):.3f})  "
          f"max/min freq = {fs[0]:.5f}/{fs[-1]:.5f}  ratio={fs[0] / fs[-1]:.2f}")
    for k in (8, 32, 64, TOP_N, 128):
        print(f"    top-{k:3d} experts hold {fs[:k].sum() * 100:6.2f}% of routed mass")
    print(f"    per-window Gini: mean={np.mean([gini(w) for w in wc]):.4f}")

    # K3 verdict
    print(f"\nK3 kill-criterion (PLAN.md: tau > 0.9 across windows => allocation stable):")
    print(f"    adjacent-window tau mean {adj_t.mean():.4f}, min {adj_t.min():.4f}; "
          f"window-vs-global tau mean {glob_t.mean():.4f}, min {glob_t.min():.4f}")


if __name__ == "__main__":
    main()
