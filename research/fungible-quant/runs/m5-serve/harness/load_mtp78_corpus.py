#!/usr/bin/env python3
"""load_mtp78_corpus — yield the exact MTP78 / GLM-5.2 activation-corpus prompts.

WHAT THIS IS
------------
The "MTP78 activation corpus" referenced from malaiwah's HF repos is a two-layer
artifact.  This loader serves the *driving* layer, because that is what a replay
needs:

  Layer 1 — the PROMPT corpus (what this module yields)
      `reap_recall_calib.jsonl`
      sha256 cf247acc7c5da9f0600c7d6ab3b7c2fcfc54ec30b794e3b6047559285fa44df4
      12,228 rows / 4 axes (3,057 each: general, legal, code-agentic,
      reasoning-termination), 34,002,059 B.
      It is *shipped inside* brandonmusic/GLM-5.2-EXL3-TR3-3.0bpw at
      `calibration_encoder/calibration/reap_recall_calib.jsonl` -- it is not a
      standalone HF dataset.

  Layer 2 — the recorded ACTIVATIONS produced by driving layer 1
      https://huggingface.co/datasets/malaiwah/GLM-5.2-MTP78-calibration-capture
      7,288,310 rows of layer-78 MoE input hidden states (bf16 [rows, 6144])
      paired with the router's ground-truth top-8 expert ids (uint8 [rows, 8]),
      in 3 safetensors shards.  See `iter_capture_shards()` below -- NOT
      downloaded by default (it is large).

Why layer 1 is the right one to replay: the recorded capture covers layer 78
only, while a convergence run needs routing for every MoE layer 3..78.  Re-driving
the same prompts through a live serve regenerates routing at every layer, and
because the prompts are byte-identical the comparison against the reference
quant stays honest.

PROVENANCE CHAIN (why this is the same corpus the reference quant used)
-----------------------------------------------------------------------
`willfalco/GLM-5.2-EXL3-TR3-3.42bpw` (the 3.42 bpw "Coder" reference) declares
in `calibration_manifest.json`:
    corpus_sha256 = cf247acc7c5da9f0600c7d6ab3b7c2fcfc54ec30b794e3b6047559285fa44df4
which is exactly the sha256 verified by this module.  The MTP78 capture dataset
card cites the same file and hash, and the 3.42 model card credits
`malaiwah/GLM-5.2-MTP78-calibration-capture` for its layer-78 routed experts.
Same bytes, all the way through.

REPLAY SEMANTICS
----------------
`iter_prompts()` reproduces `tools/drive_corpus.py` from
malaiwah/GLM-5.2-EXL3-TR3-MTP78 (and `capture_b300.py`) exactly:
raw `record["text"]`, checkpoint tokenizer, truncate to 4096 tokens,
prompt sent as token ids, `max_tokens=1`, `temperature=0.0`, `ignore_eos=True`.
The `trim_128` option reproduces the small-final-chunk workaround
(if `len(ids) % 128` is in 1..8, drop the remainder) that the capture run used
to dodge a fork-kernel IMA.

USAGE
-----
    python3 load_mtp78_corpus.py                 # verify + stats
    python3 load_mtp78_corpus.py --axis axis3_code_agentic --limit 5 --show

    from load_mtp78_corpus import iter_prompts, corpus_path
    for rec in iter_prompts(axis="axis3_code_agentic"):
        rec.line_no, rec.axis, rec.text
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from typing import Iterable, Iterator, Optional

CORPUS_SHA256 = "cf247acc7c5da9f0600c7d6ab3b7c2fcfc54ec30b794e3b6047559285fa44df4"
CORPUS_ROWS = 12228
CORPUS_BYTES = 34002059
MAX_SAMPLE_TOKENS = 4096
AXES = ("axis1_general", "axis2_legal", "axis3_code_agentic", "axis4_reasoning_termination")

CAPTURE_DATASET = "malaiwah/GLM-5.2-MTP78-calibration-capture"
SOURCE_CHECKPOINT = "brandonmusic/GLM-5.2-EXL3-TR3-3.0bpw"
SOURCE_REVISION = "9297b9f1d53af5c67cffa01e30cc071a1ff7144b"
_REL = "calibration_encoder/calibration/reap_recall_calib.jsonl"

# Known-good local snapshot first: this file is already on disk, so a correct
# run needs zero network and zero extra disk.
_LOCAL_CANDIDATES = (
    f"/home/mbelleau/.cache/huggingface/hub/models--brandonmusic--GLM-5.2-EXL3-TR3-3.0bpw/snapshots/{SOURCE_REVISION}/{_REL}",
    f"/home/mbelleau/protensors-work/vllm-voipmonitor/research/fungible-quant/runs/dl-glm52-k3/{_REL}",
)


@dataclass(frozen=True)
class Prompt:
    """One corpus row. `text` is the raw string the capture run tokenized."""
    line_no: int
    axis: str
    text: str
    source: str
    calib_tokens: Optional[int] = None


def corpus_path(path: Optional[str] = None, allow_download: bool = False) -> str:
    """Resolve the corpus file. Env override: FQ_MTP78_CORPUS."""
    if path:
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        return path
    env = os.environ.get("FQ_MTP78_CORPUS")
    if env and os.path.exists(env):
        return env
    for c in _LOCAL_CANDIDATES:
        if os.path.exists(c):
            return c
    if allow_download:
        from huggingface_hub import hf_hub_download  # lazy: keeps import cost off the happy path
        return hf_hub_download(
            SOURCE_CHECKPOINT, _REL, revision=SOURCE_REVISION,
            token=os.environ.get("HF_TOKEN"),
        )
    raise FileNotFoundError(
        "reap_recall_calib.jsonl not found locally. It ships inside "
        f"{SOURCE_CHECKPOINT}@{SOURCE_REVISION} at {_REL}. "
        "Set FQ_MTP78_CORPUS=<path>, or pass allow_download=True "
        "(~32.4 MiB single-file download, not the whole 295 GB repo)."
    )


def verify(path: Optional[str] = None) -> dict:
    """Hash the corpus and check it against the pinned sha256."""
    p = corpus_path(path)
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    digest = h.hexdigest()
    size = os.path.getsize(p)
    return {
        "path": p,
        "sha256": digest,
        "sha256_expected": CORPUS_SHA256,
        "sha256_ok": digest == CORPUS_SHA256,
        "bytes": size,
        "bytes_ok": size == CORPUS_BYTES,
    }


def iter_prompts(
    path: Optional[str] = None,
    axis: Optional[str] = None,
    limit: int = 0,
    allow_download: bool = False,
) -> Iterator[Prompt]:
    """Yield corpus rows in file order (the order the capture run drove them).

    `axis` filters to one of AXES; `limit` caps the number of rows yielded.
    `line_no` is the 0-based physical line index, matching drive_corpus.py.
    """
    if axis is not None and axis not in AXES:
        raise ValueError(f"unknown axis {axis!r}; expected one of {AXES}")
    p = corpus_path(path, allow_download=allow_download)
    n = 0
    with open(p, encoding="utf-8") as f:
        for line_no, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            a = rec.get("axis", "?")
            if axis is not None and a != axis:
                continue
            meta = rec.get("meta") or {}
            yield Prompt(
                line_no=line_no,
                axis=a,
                text=rec["text"],
                source=rec.get("source", "?"),
                calib_tokens=meta.get("calib_tokens"),
            )
            n += 1
            if limit and n >= limit:
                return


def to_token_ids(text: str, tokenizer, trim_128: bool = True) -> list:
    """Tokenize exactly as the capture run did (drive_corpus.py semantics)."""
    ids = tokenizer.encode(text)[:MAX_SAMPLE_TOKENS]
    if trim_128:
        r = len(ids) % 128
        if 0 < r <= 8:  # lethal small-final-chunk class (fork-kernel IMA)
            ids = ids[: len(ids) - r]
    return ids


def completion_bodies(prompts: Iterable[Prompt], tokenizer, model: str = "GLM-5.2",
                      trim_128: bool = True) -> Iterator[dict]:
    """Yield /v1/completions request bodies identical to the capture drive."""
    for pr in prompts:
        ids = to_token_ids(pr.text, tokenizer, trim_128=trim_128)
        if not ids:
            continue
        yield {"model": model, "prompt": ids, "max_tokens": 1,
               "temperature": 0.0, "ignore_eos": True}


def iter_capture_shards(local_dir: Optional[str] = None):
    """Layer 2: the recorded layer-78 activations. ~large; downloads on demand.

    Yields (shard_name, x[rows,6144] bf16, ids[rows,8] uint8). Needs safetensors
    + torch. Only useful for layer-78 work -- a full-model convergence replay
    should drive `iter_prompts()` instead.
    """
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    for i in (1, 2, 3):
        name = f"capture-{i:05d}-of-00003.safetensors"
        p = hf_hub_download(CAPTURE_DATASET, name, repo_type="dataset",
                            local_dir=local_dir, token=os.environ.get("HF_TOKEN"))
        with safe_open(p, framework="pt") as f:
            yield name, f.get_tensor("x"), f.get_tensor("ids")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--path", help="explicit corpus path")
    ap.add_argument("--axis", choices=AXES, help="filter to one axis")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--show", action="store_true", help="print each prompt (truncated)")
    ap.add_argument("--download", action="store_true", help="allow single-file HF download")
    ap.add_argument("--json", action="store_true", help="emit machine-readable summary")
    a = ap.parse_args(argv)

    v = verify(a.path) if not a.path or os.path.exists(corpus_path(a.path, a.download)) else {}
    counts, chars, tokens = {}, 0, 0
    n = 0
    for pr in iter_prompts(a.path, axis=a.axis, limit=a.limit, allow_download=a.download):
        n += 1
        counts[pr.axis] = counts.get(pr.axis, 0) + 1
        chars += len(pr.text)
        tokens += pr.calib_tokens or 0
        if a.show:
            head = pr.text[:160].replace("\n", "\\n")
            print(f"[{pr.line_no:6d}] {pr.axis:28s} {pr.source:44s} {head}")

    summary = {
        "corpus": "reap_recall_calib.jsonl",
        "path": v.get("path"),
        "sha256": v.get("sha256"),
        "sha256_ok": v.get("sha256_ok"),
        "bytes": v.get("bytes"),
        "items_yielded": n,
        "expected_total_rows": CORPUS_ROWS,
        "axis_counts": counts,
        "total_chars": chars,
        "sum_calib_tokens": tokens,
    }
    if a.json:
        print(json.dumps(summary, indent=1))
    else:
        print(f"corpus   : {summary['path']}")
        print(f"sha256   : {summary['sha256']}  ok={summary['sha256_ok']} "
              f"(expected {CORPUS_SHA256[:16]}...)")
        print(f"bytes    : {summary['bytes']:,}")
        print(f"ITEM COUNT: {n:,}" + ("" if a.limit or a.axis else f"  (expected {CORPUS_ROWS:,})"))
        for k in sorted(counts):
            print(f"   {k:30s} {counts[k]:,}")
        print(f"total chars: {chars:,}   sum(meta.calib_tokens): {tokens:,}")
    ok = bool(v.get("sha256_ok")) and (a.axis or a.limit or n == CORPUS_ROWS)
    if not ok:
        print("VERIFY FAILED", file=sys.stderr)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
