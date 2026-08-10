#!/usr/bin/env python3
"""Repack a GLM-5.2 encode window's outputs into segments and publish to HF.

Encoder emits <work>/tr3-layer-LLL.safetensors; fq_repack expects
model-layer-LLL.safetensors, so each K is staged through a fresh temp dir
of symlinks (fresh => no stale entries can pollute the glob).
"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, "/home/mbelleau/protensors-work/vllm-voipmonitor/"
                   "research/fungible-quant/tools")
import fq_repack  # noqa: E402
from huggingface_hub import HfApi  # noqa: E402

CAPTURE = Path("/home/mbelleau/glm52-capture/capture_run_manifest.json")
FINGERPRINT = json.loads(CAPTURE.read_text()).get("fingerprint", "window1")
OUT = Path("/home/mbelleau/glm52-segments")
KS = (2, 5)


def stage_and_repack(k: int) -> int:
    work = Path(f"/home/mbelleau/glm52-work-k{k}")
    shards = sorted(work.glob("tr3-layer-*.safetensors"))
    if not shards:
        print(f"K{k}: no encoder outputs in {work}", flush=True)
        return 0
    with tempfile.TemporaryDirectory(prefix=f"fqstage-k{k}-") as tmp:
        stage = Path(tmp)
        for f in shards:
            layer = int(f.stem.replace("tr3-layer-", ""))
            (stage / f"model-layer-{layer:03d}.safetensors").symlink_to(f)
        print(f"K{k}: staged {len(shards)} layers -> repack", flush=True)
        fq_repack.main([
            "--snapshot", str(stage),
            "--source-repo", f"local:glm52-k{k}-encode-of-window1",
            "--revision", FINGERPRINT,
            "--base-model", "zai-org/GLM-5.2",
            "--k", str(k),
            "--out", str(OUT),
        ])
    return len(shards)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    total = sum(stage_and_repack(k) for k in KS)
    if not total:
        print("nothing to publish", flush=True)
        return 1
    HfApi().upload_large_folder(
        repo_id="malaiwah/GLM-5.2-EXL3-FQ-segments",
        folder_path=str(OUT), repo_type="model",
        ignore_patterns=["state.json", "*.part", ".huggingface*"])
    print("WINDOW-1 K2/K5 PUBLISHED", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
