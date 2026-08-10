"""Tests for oci_unpack: layer ordering, whiteouts, opaque dirs, path safety."""
import gzip
import hashlib
import io
import json
import tarfile
from pathlib import Path

import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent))
import oci_unpack  # noqa: E402


def make_layer(entries) -> bytes:
    """entries: list of (name, content|None for dir, mode) or ('SYMLINK', name, target)."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        for e in entries:
            if e[0] == "SYMLINK":
                info = tarfile.TarInfo(e[1])
                info.type = tarfile.SYMTYPE
                info.linkname = e[2]
                tf.addfile(info)
                continue
            name, content, mode = e
            info = tarfile.TarInfo(name)
            info.mode = mode
            if content is None:
                info.type = tarfile.DIRTYPE
                tf.addfile(info)
            else:
                data = content.encode()
                info.size = len(data)
                tf.addfile(info, io.BytesIO(data))
    return gzip.compress(buf.getvalue())


@pytest.fixture()
def layout(tmp_path):
    root = tmp_path / "oci"
    blobs = root / "blobs" / "sha256"
    blobs.mkdir(parents=True)

    def put(data: bytes) -> str:
        d = hashlib.sha256(data).hexdigest()
        (blobs / d).write_bytes(data)
        return f"sha256:{d}"

    l1 = make_layer([
        ("a", None, 0o755),
        ("a/file1", "one", 0o644),
        ("a/file2", "two", 0o644),
        ("c", None, 0o755),
        ("c/old", "stale", 0o644),
        ("../escape", "evil", 0o644),
        ("SYMLINK", "a/link", "file2"),
    ])
    l2 = make_layer([
        ("a/.wh.file1", "", 0o644),
        ("b", None, 0o755),
        ("b/new", "fresh", 0o644),
        ("c/.wh..wh..opq", "", 0o644),
        ("c/replacement", "clean", 0o644),
        ("a/file2", "two-v2", 0o600),
    ])
    config = put(json.dumps({"config": {"Env": ["X=1"], "Entrypoint": ["/bin/sh"]}}).encode())
    d1, d2 = put(l1), put(l2)
    manifest = put(json.dumps({
        "schemaVersion": 2,
        "config": {"digest": config},
        "layers": [{"digest": d1}, {"digest": d2}],
    }).encode())
    (root / "index.json").write_text(json.dumps({
        "schemaVersion": 2,
        "manifests": [{"digest": manifest,
                       "annotations": {"org.opencontainers.image.ref.name": "t1"}}],
    }))
    (root / "oci-layout").write_text('{"imageLayoutVersion":"1.0.0"}')
    return root


def test_unpack_semantics(layout, tmp_path):
    dest = tmp_path / "rootfs"
    summary = oci_unpack.unpack(layout, dest, tag="t1")

    assert not (dest / "a/file1").exists(), "whiteout must delete file1"
    assert (dest / "a/file2").read_text() == "two-v2", "upper layer wins"
    assert (dest / "b/new").read_text() == "fresh"
    assert not (dest / "c/old").exists(), "opaque dir clears lower content"
    assert (dest / "c/replacement").read_text() == "clean"
    assert (dest / "a/link").is_symlink()
    assert not (tmp_path / "escape").exists(), "path traversal must be blocked"
    assert summary["Env"] == ["X=1"]
    assert summary["Entrypoint"] == ["/bin/sh"]
    cfg = json.loads(Path(str(dest) + ".oci-config.json").read_text())
    assert cfg["layers"] == 2


def test_mode_preserved(layout, tmp_path):
    dest = tmp_path / "rootfs"
    oci_unpack.unpack(layout, dest, tag="t1")
    assert (dest / "a/file2").stat().st_mode & 0o777 == 0o600
