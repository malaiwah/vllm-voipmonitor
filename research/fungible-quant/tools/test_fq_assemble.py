"""Tests for fq_assemble: byte-identity round trip through repack + assemble."""
import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import fq_repack  # noqa: E402
import fq_assemble  # noqa: E402
from test_fq_repack import LAYERS, E, write_shard  # noqa: E402


@pytest.fixture()
def repacked(tmp_path):
    snap = tmp_path / "snap"
    snap.mkdir()
    for i, layer in enumerate(LAYERS):
        write_shard(snap / f"model-layer-{layer:03d}.safetensors", layer, scramble=bool(i))
    (snap / "config.json").write_text("{}")
    lines = [
        f"{hashlib.sha256((snap / f'model-layer-{l:03d}.safetensors').read_bytes()).hexdigest()}"
        f"  model-layer-{l:03d}.safetensors"
        for l in LAYERS
    ]
    (snap / "MANIFEST.sha256").write_text("\n".join(lines) + "\n")
    out = tmp_path / "segments"
    fq_repack.main([
        "--snapshot", str(snap),
        "--source-repo", "test/src", "--revision", "r0",
        "--base-model", "test/base", "--out", str(out),
        "--sign-key", str(tmp_path / "k.key"),
    ])
    return snap, out, tmp_path


def test_all_k3_assembly_is_byte_identical(repacked):
    snap, segments, tmp = repacked
    policy = {
        "schema": "fq-policy/2",
        "bits_per_expert": {str(l): [3] * E for l in LAYERS},
    }
    ppath = tmp / "policy.json"
    ppath.write_text(json.dumps(policy))
    out = tmp / "assembled"
    fq_assemble.main([
        "--segments", str(segments), "--source", str(snap),
        "--policy", str(ppath), "--out", str(out),
    ])
    for layer in LAYERS:
        a = (snap / f"model-layer-{layer:03d}.safetensors").read_bytes()
        b = (out / f"model-layer-{layer:03d}.safetensors").read_bytes()
        assert hashlib.sha256(a).hexdigest() == hashlib.sha256(b).hexdigest(), layer
    assert (out / "config.json").exists(), "non-shard files copied through"


def test_missing_segment_fails_loudly(repacked):
    snap, segments, tmp = repacked
    policy = {"schema": "fq-policy/2",
              "bits_per_expert": {str(LAYERS[0]): [4] * E}}
    ppath = tmp / "p4.json"
    ppath.write_text(json.dumps(policy))
    with pytest.raises(FileNotFoundError, match="k4"):
        fq_assemble.main([
            "--segments", str(segments), "--source", str(snap),
            "--policy", str(ppath), "--out", str(tmp / "x"),
        ])
