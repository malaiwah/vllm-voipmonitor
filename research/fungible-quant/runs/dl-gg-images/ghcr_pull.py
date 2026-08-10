#!/usr/bin/env python3
"""Pull a ghcr.io image to an OCI layout dir without docker/podman.

Resumable: blobs already present with correct sha256 are skipped.
Usage: ghcr_pull.py <owner/pkg> <tag> <dest_dir>
Auth: GH_TOKEN env var (used as password for the ghcr token exchange).
"""
import hashlib
import json
import os
import sys
import urllib.request

REG = "https://ghcr.io"
ACCEPT = ", ".join([
    "application/vnd.oci.image.index.v1+json",
    "application/vnd.docker.distribution.manifest.list.v2+json",
    "application/vnd.oci.image.manifest.v1+json",
    "application/vnd.docker.distribution.manifest.v2+json",
])


def http_json(url, headers):
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req) as r:
        return json.load(r), dict(r.headers)


def get_token(name):
    import base64
    url = f"{REG}/token?service=ghcr.io&scope=repository:{name}:pull"
    gh = os.environ.get("GH_TOKEN", "")
    headers = {}
    if gh:
        basic = base64.b64encode(f"x:{gh}".encode()).decode()
        headers["Authorization"] = f"Basic {basic}"
    data, _ = http_json(url, headers)
    return data["token"]


def fetch_blob(name, digest, dest, tok, size=None):
    algo, hexd = digest.split(":")
    path = os.path.join(dest, "blobs", algo, hexd)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        if h.hexdigest() == hexd:
            print(f"  skip {digest[:19]} (cached)", flush=True)
            return path
        os.unlink(path)
    req = urllib.request.Request(
        f"{REG}/v2/{name}/blobs/{digest}",
        headers={"Authorization": f"Bearer {tok}"})
    h = hashlib.sha256()
    tmp = path + ".part"
    done = 0
    with urllib.request.urlopen(req) as r, open(tmp, "wb") as f:
        while True:
            chunk = r.read(1 << 22)
            if not chunk:
                break
            h.update(chunk)
            f.write(chunk)
            done += len(chunk)
    if h.hexdigest() != hexd:
        os.unlink(tmp)
        raise RuntimeError(f"sha mismatch for {digest}")
    os.rename(tmp, path)
    print(f"  got  {digest[:19]} {done/1e6:.0f} MB", flush=True)
    return path


def main(pkg, tag, dest):
    os.makedirs(dest, exist_ok=True)
    tok = get_token(pkg)
    hdr = {"Authorization": f"Bearer {tok}", "Accept": ACCEPT}
    man, mh = http_json(f"{REG}/v2/{pkg}/manifests/{tag}", hdr)
    mt = man.get("mediaType", mh.get("Content-Type", ""))
    if "index" in mt or "list" in mt:
        entry = next(m for m in man["manifests"]
                     if m.get("platform", {}).get("architecture") == "amd64"
                     and m.get("platform", {}).get("os") == "linux")
        digest = entry["digest"]
        man, _ = http_json(f"{REG}/v2/{pkg}/manifests/{digest}", hdr)
        mt = man.get("mediaType", "")
    raw = json.dumps(man, separators=(",", ":")).encode()
    # Re-fetch the exact manifest bytes for a faithful digest
    req = urllib.request.Request(
        f"{REG}/v2/{pkg}/manifests/{tag}", headers=hdr)
    with urllib.request.urlopen(req) as r:
        raw = r.read()
    mdigest = "sha256:" + hashlib.sha256(raw).hexdigest()
    algo, hexd = mdigest.split(":")
    os.makedirs(os.path.join(dest, "blobs", algo), exist_ok=True)
    with open(os.path.join(dest, "blobs", algo, hexd), "wb") as f:
        f.write(raw)
    print(f"manifest {mdigest[:19]} config+{len(man['layers'])} layers", flush=True)
    fetch_blob(pkg, man["config"]["digest"], dest, tok)
    total = 0
    for layer in man["layers"]:
        fetch_blob(pkg, layer["digest"], dest, tok, layer.get("size"))
        total += layer.get("size", 0)
    with open(os.path.join(dest, "oci-layout"), "w") as f:
        json.dump({"imageLayoutVersion": "1.0.0"}, f)
    with open(os.path.join(dest, "index.json"), "w") as f:
        json.dump({"schemaVersion": 2, "manifests": [{
            "mediaType": mt or "application/vnd.oci.image.manifest.v1+json",
            "digest": mdigest, "size": len(raw),
            "annotations": {"org.opencontainers.image.ref.name": tag},
        }]}, f)
    print(f"DONE {pkg}:{tag} -> {dest} ({total/1e9:.1f} GB layers)", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
