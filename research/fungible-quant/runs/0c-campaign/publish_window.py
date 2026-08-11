#!/usr/bin/env python3
"""Publish a GLM-5.2 encode window as attested Progressive Tensors segments.

Runs unattended from the campaign supervisor, so it has to be correct
without supervision. Two failure modes found by review — both caused by
this script routing fresh encodes through the repack tool and then
uploading a *locally* derived manifest — are fixed here:

1. Attestations said `repack-of` with a local work dir as "source".
   These are OUR encodes: the lines are re-emitted as `encode-of`, pinning
   the base model + revision, capture fingerprint, encoder sha, quant args
   and an explicit determinism scope, before anything is uploaded.

2. The root `fq-manifest.json` was rebuilt from this window's local dir
   (K2/K5 only) and clobbered the published one (which also covers K3).
   The manifest is now rebuilt from the REMOTE inventory after upload, so
   it always describes the whole published family, whatever produced it.
"""
from __future__ import annotations

import base64
import hashlib
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, "/home/mbelleau/protensors-work/vllm-voipmonitor/"
                   "research/fungible-quant/tools")
import fq_repack  # noqa: E402
from huggingface_hub import HfApi  # noqa: E402

def _load_env() -> None:
    """Self-sufficient credentials: the supervisor runs us from a bare
    environment, and a missing HF token makes upload_large_folder hang on
    an interactive auth prompt (observed: a 57-minute futex wait that
    silently stopped the campaign publishing anything)."""
    import os
    env = Path.home() / ".fq_env"
    if env.exists() and not os.environ.get("HF_TOKEN"):
        for line in env.read_text().splitlines():
            line = line.strip().removeprefix("export ").strip()
            if "=" in line and not line.startswith("#"):
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip().strip('"\''))
        if os.environ.get("HF_TOKEN") and not os.environ.get("HUGGING_FACE_HUB_TOKEN"):
            os.environ["HUGGING_FACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]


_load_env()

REPO = "malaiwah/GLM-5.2-EXL3-FQ-segments"
CAPTURE = Path("/home/mbelleau/glm52-capture/capture_run_manifest.json")
OUT = Path("/home/mbelleau/glm52-segments")
KS = (2, 4, 5)
BASE_MODEL = "zai-org/GLM-5.2"
BASE_REV = "b4734de4facf877f85769a911abafc5283eab3d9"
ENCODER_SHA = "e9a85a47e165c8d8644354cef611efbb81dfd9ba88544ca59f0c80ee6bc75032"
ENCODER_BUNDLE = ("brandonmusic/GLM-5.2-EXL3-TR3-3.0bpw"
                  "@9297b9f1d53af5c67cffa01e30cc071a1ff7144b:calibration_encoder")
KEY = Path.home() / ".fq_keys/fq_signing.key"


def capture_fingerprint() -> str | None:
    try:
        d = json.loads(CAPTURE.read_text())
        return d.get("capture_fingerprint") or d.get("fingerprint")
    except Exception:
        return None


def stage_and_repack(k: int, fp: str, already: set[str] | None = None) -> int:
    work = Path(f"/home/mbelleau/glm52-work-k{k}")
    shards = sorted(work.glob("tr3-layer-*.safetensors"))
    # Skip layers already published. Local segments are a cache, not the
    # deliverable, and they get pruned to reclaim disk — without this check
    # the next publish sees "segment file missing" and regenerates the whole
    # back catalogue (observed: ~236 GB of rework queued behind a 56-layer
    # K2 pass, on a box with 250 GB free and an assembly running).
    if already:
        keep = []
        for f in shards:
            layer = int(f.stem.replace("tr3-layer-", ""))
            if f"layer-{layer:03d}.k{k}.safetensors" not in already:
                keep.append(f)
        skipped = len(shards) - len(keep)
        if skipped:
            print(f"K{k}: {skipped} layers already on HF — not regenerating",
                  flush=True)
        shards = keep
    if not shards:
        return 0
    with tempfile.TemporaryDirectory(prefix=f"fqstage-k{k}-") as tmp:
        stage = Path(tmp)
        for f in shards:
            layer = int(f.stem.replace("tr3-layer-", ""))
            (stage / f"model-layer-{layer:03d}.safetensors").symlink_to(f)
        print(f"K{k}: staging {len(shards)} layers -> repack", flush=True)
        fq_repack.main([
            "--snapshot", str(stage),
            "--source-repo", f"local:glm52-k{k}-encode-of",
            "--revision", BASE_REV,
            "--allow-provenance-change",
            "--base-model", BASE_MODEL,
            "--k", str(k), "--out", str(OUT),
        ])
    return len(shards)


def reattest_encode_of(k: int, fp: str) -> int:
    """Rewrite this K's attestations with the truthful encode-of predicate."""
    signer = fq_repack.Signer(KEY)
    n = 0
    for att in sorted((OUT / "attestations").glob(f"*.k{k}.jsonl")):
        first = json.loads(att.read_text().splitlines()[0])
        old = json.loads(base64.b64decode(first["payload"]))
        if old.get("predicate") == "encode-of":
            continue
        layer = int(att.name.split("layer-")[1][:3])
        seg = OUT / f"layer-{layer:03d}.k{k}.safetensors"
        payload = {
            "schema": "fq-attestation/1",
            "predicate": "encode-of",
            "fragment": {"file": seg.name,
                         "sha256": hashlib.sha256(seg.read_bytes()).hexdigest(),
                         "size": seg.stat().st_size},
            "materials": {
                "base_model": BASE_MODEL, "base_revision": BASE_REV,
                "capture_fingerprint": fp,
                "encoder": "encode_tr3_v31.py",
                "encoder_sha256": ENCODER_SHA,
                "encoder_bundle": ENCODER_BUNDLE,
            },
            "quant_args": {"K": k, "codebook": "mcg", "seed_base": 20260711,
                           "sigma_reg": 0.025, "tp": 4, "out_scales": "auto"},
            "determinism_scope": {
                "torch": "2.12.0+cu132", "gpu_arch": "sm120",
                "capture_engine": "capture_stream.py (bit-exact vs reference)",
                "note": ("bit-reproducible only within this stack; use the "
                         "equivalence-of rung across stacks"),
            },
            "expert_sha256": old.get("expert_sha256"),
            "created_utc": old.get("created_utc"),
        }
        att.write_text(signer.sign_line(payload) + "\n")
        n += 1
    return n


def rebuild_remote_manifest(api: HfApi) -> dict:
    """Authoritative manifest built from what is ACTUALLY published."""
    files = api.list_repo_files(REPO)
    root = [f for f in files if f.endswith(".safetensors") and "/" not in f]
    per_k = {}
    for f in root:
        k = int(f.split(".k")[1][0])
        per_k.setdefault(k, []).append(int(f.split("layer-")[1][:3]))
    manifest = {
        "schema": "fq-manifest/1",
        "base_model": BASE_MODEL,
        "revision": BASE_REV,
        "k_variants": sorted(per_k),
        "layout": "rank_sliced_tp4",
        "num_experts": 256,
        "sources": ["brandonmusic/GLM-5.2-EXL3-TR3-3.0bpw"],
        "signer_pubkey": fq_repack.Signer(KEY).pub_hex,
        "per_k": {
            str(k): {
                "index": f"index-k{k}.json",
                "layers": [min(v), max(v)],
                "segment_count": len(v),
                "predicate": ("repack-of" if k == 3 else "encode-of"),
                "provenance": (
                    "repack-of brandonmusic/GLM-5.2-EXL3-TR3-3.0bpw@9297b9f1"
                    if k == 3 else
                    f"encode-of {BASE_MODEL}@{BASE_REV[:8]} (encoder {ENCODER_SHA[:8]})"),
            } for k, v in sorted(per_k.items())},
        "note": ("Rebuilt from the remote inventory on every publish, so it "
                 "always describes the whole published family."),
    }
    return manifest


def main() -> int:
    fp = capture_fingerprint()
    OUT.mkdir(parents=True, exist_ok=True)
    # Ask the remote what it already has BEFORE doing any repack work.
    # Costs one API call; saves regenerating hundreds of GB.
    already: set[str] = set()
    try:
        already = {f for f in HfApi().list_repo_files(REPO)
                   if f.endswith(".safetensors") and "/" not in f}
        print(f"remote already holds {len(already)} segments", flush=True)
    except Exception as exc:  # noqa: BLE001 - offline is not fatal
        print(f"could not list remote ({exc}); will repack everything",
              flush=True)
    total = 0
    for k in KS:
        n = stage_and_repack(k, fp, already)
        if n:
            fixed = reattest_encode_of(k, fp)
            print(f"K{k}: {n} layers, {fixed} attestations re-emitted as encode-of",
                  flush=True)
        total += n
    if not total:
        print("nothing to publish", flush=True)
        return 1

    import os
    if not os.environ.get("HF_TOKEN"):
        print("NO HF TOKEN — refusing to upload (would hang on auth prompt)", flush=True)
        return 2
    api = HfApi()
    api.upload_large_folder(repo_id=REPO, folder_path=str(OUT), repo_type="model",
                            ignore_patterns=["state.json", "*.part", ".huggingface*",
                                             "fq-manifest.json"])
    manifest = rebuild_remote_manifest(api)
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump(manifest, fh, indent=1)
        tmp = fh.name
    api.upload_file(path_or_fileobj=tmp, path_in_repo="fq-manifest.json",
                    repo_id=REPO,
                    commit_message="manifest: rebuilt from remote inventory")
    print(f"PUBLISHED — manifest k_variants={manifest['k_variants']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
