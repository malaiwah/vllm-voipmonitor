#!/usr/bin/env python3
"""oci_unpack — unpack an OCI-layout image to a rootfs dir without a runtime.

Applies layers in manifest order with overlayfs whiteout semantics
(`.wh.<name>` deletes, `.wh..wh..opq` makes a directory opaque), sanitizes
paths, skips device nodes (non-root), and writes `<dest>.oci-config.json`
with the image config (Env/Entrypoint/Cmd/WorkingDir) so callers can
replicate the container's runtime environment.

Usage: oci_unpack.py <oci-layout-dir> <dest-rootfs> [--tag TAG]
"""
from __future__ import annotations

import argparse
import gzip
import json
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

WH = ".wh."
OPAQUE = ".wh..wh..opq"


def read_json(path: Path):
    with open(path) as f:
        return json.load(f)


def blob_path(layout: Path, digest: str) -> Path:
    algo, hexd = digest.split(":")
    return layout / "blobs" / algo / hexd


def pick_manifest(layout: Path, tag: str | None):
    index = read_json(layout / "index.json")
    manifests = index["manifests"]
    entry = manifests[0]
    if tag:
        for m in manifests:
            if m.get("annotations", {}).get("org.opencontainers.image.ref.name") == tag:
                entry = m
                break
    man = read_json(blob_path(layout, entry["digest"]))
    if "manifests" in man:  # nested index (multi-arch)
        sub = next(
            m for m in man["manifests"]
            if m.get("platform", {}).get("architecture") == "amd64"
        )
        man = read_json(blob_path(layout, sub["digest"]))
    return man


def open_layer(path: Path) -> tarfile.TarFile:
    with open(path, "rb") as f:
        magic = f.read(4)
    if magic[:2] == b"\x1f\x8b":
        return tarfile.open(fileobj=gzip.open(path, "rb"), mode="r|")
    if magic == b"\x28\xb5\x2f\xfd":
        zstd = shutil.which("zstd") or shutil.which("unzstd")
        if not zstd:
            raise RuntimeError(f"{path}: zstd layer but no zstd binary on PATH")
        proc = subprocess.Popen([zstd, "-dc", str(path)], stdout=subprocess.PIPE)
        return tarfile.open(fileobj=proc.stdout, mode="r|")
    return tarfile.open(path, mode="r")


def safe_target(dest: Path, name: str) -> Path | None:
    parts = [p for p in Path(name).parts if p not in ("", ".", "/")]
    if any(p == ".." for p in parts):
        return None
    return dest.joinpath(*parts) if parts else None


def rm(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink(missing_ok=True)
    elif path.is_dir():
        shutil.rmtree(path, ignore_errors=True)


def apply_layer(tf: tarfile.TarFile, dest: Path) -> tuple[int, int]:
    extracted = skipped = 0
    for m in tf:
        base = Path(m.name).name
        if base == OPAQUE:
            target_dir = safe_target(dest, str(Path(m.name).parent))
            if target_dir and target_dir.is_dir():
                for child in target_dir.iterdir():
                    rm(child)
            continue
        if base.startswith(WH):
            victim = safe_target(dest, str(Path(m.name).parent / base[len(WH):]))
            if victim:
                rm(victim)
            continue
        target = safe_target(dest, m.name)
        if target is None:
            skipped += 1
            continue
        if m.isdev():
            skipped += 1
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        if m.isdir():
            target.mkdir(exist_ok=True)
            continue
        rm(target)  # replace whatever a lower layer put there
        if m.issym():
            target.symlink_to(m.linkname)
        elif m.islnk():
            src = safe_target(dest, m.linkname)
            if src and src.exists():
                try:
                    target.hardlink_to(src)
                except OSError:
                    shutil.copy2(src, target)
            else:
                skipped += 1
                continue
        else:
            f = tf.extractfile(m)
            with open(target, "wb") as out:
                if f is not None:
                    shutil.copyfileobj(f, out, length=1 << 20)
            target.chmod(m.mode & 0o7777)
        extracted += 1
    return extracted, skipped


def unpack(layout: Path, dest: Path, tag: str | None = None) -> dict:
    man = pick_manifest(layout, tag)
    config = read_json(blob_path(layout, man["config"]["digest"]))
    dest.mkdir(parents=True, exist_ok=True)
    total_e = total_s = 0
    for i, layer in enumerate(man["layers"]):
        with open_layer(blob_path(layout, layer["digest"])) as tf:
            e, s = apply_layer(tf, dest)
        total_e += e
        total_s += s
        print(f"layer {i+1}/{len(man['layers'])}: +{e} files ({s} skipped)", flush=True)
    cfg = config.get("config", {})
    summary = {
        "Env": cfg.get("Env", []),
        "Entrypoint": cfg.get("Entrypoint"),
        "Cmd": cfg.get("Cmd"),
        "WorkingDir": cfg.get("WorkingDir"),
        "Labels": cfg.get("Labels", {}),
        "layers": len(man["layers"]),
        "files": total_e,
        "skipped": total_s,
    }
    Path(str(dest) + ".oci-config.json").write_text(json.dumps(summary, indent=1))
    return summary


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("layout", type=Path)
    p.add_argument("dest", type=Path)
    p.add_argument("--tag", default=None)
    args = p.parse_args(argv)
    s = unpack(args.layout, args.dest, args.tag)
    print(f"done: {s['files']} files from {s['layers']} layers -> {args.dest}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
