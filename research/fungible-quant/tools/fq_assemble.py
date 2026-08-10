#!/usr/bin/env python3
"""fq_assemble — Progressive Tensors segments + policy → bootable checkpoint.

The M0 gate tool (04-milestones.md): given per-layer per-K segment files
(fq_repack output or downloaded), a bits_per_expert policy JSON, and the
source checkpoint (for dense/attention/shared tensors and header order),
emit per-layer shards in the original GG rank-sliced layout.

Byte-identity property: assembling the all-K3 policy from K3 segments
reproduces the source shards byte-for-byte (same header, same order) —
tested in test_fq_assemble.py. Mixed policies swap in K4 expert bytes from
K4 segments as they become available.
"""
from __future__ import annotations

import argparse
import json
import mmap
import os
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from fq_repack import EXPERT_RE, read_header  # noqa: E402


def load_segment_index(seg_dir: Path, k: int) -> dict:
    p = seg_dir / f"index-k{k}.json"
    return json.loads(p.read_text()) if p.exists() else {}


class SegmentReader:
    """Random access to expert tensors inside one segment file."""

    def __init__(self, path: Path):
        self.hdr, self.body = read_header(path)
        self.hdr.pop("__metadata__", None)
        self.f = open(path, "rb")
        self.mm = mmap.mmap(self.f.fileno(), 0, access=mmap.ACCESS_READ)

    def tensor_bytes(self, name: str) -> bytes:
        a, b = self.hdr[name]["data_offsets"]
        return self.mm[self.body + a: self.body + b]

    def close(self):
        self.mm.close()
        self.f.close()


def assemble_layer(
    src_shard: Path,
    out_shard: Path,
    layer: int,
    bits_for_expert,           # callable expert_id -> K
    readers: dict[int, SegmentReader],
) -> dict:
    """Write one output shard using the source header as the template."""
    hdr, body_off = read_header(src_shard)
    meta = hdr.pop("__metadata__", None)
    counts = {}
    with open(src_shard, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        ordered = sorted(hdr.items(), key=lambda kv: kv[1]["data_offsets"][0])
        header = dict(hdr) if meta is None else {"__metadata__": meta, **hdr}
        hj = json.dumps(header, separators=(",", ":")).encode()
        hj += b" " * ((8 - len(hj) % 8) % 8)
        tmp = out_shard.with_suffix(".part")
        with open(tmp, "wb") as out:
            out.write(struct.pack("<Q", len(hj)))
            out.write(hj)
            for name, t in ordered:
                m = EXPERT_RE.match(name)
                if m:
                    expert = int(m.group(2))
                    k = bits_for_expert(expert)
                    data = readers[k].tensor_bytes(name)
                    counts[k] = counts.get(k, 0) + 1
                else:
                    a, b = t["data_offsets"]
                    data = mm[body_off + a: body_off + b]
                want = t["data_offsets"][1] - t["data_offsets"][0]
                if len(data) != want:
                    raise AssertionError(
                        f"{name}: segment bytes {len(data)} != source slot {want} "
                        f"(K-swap changes tensor size; template header only valid "
                        f"when all Ks match the source — use --reindex)")
                out.write(data)
        mm.close()
    os.replace(tmp, out_shard)
    return counts


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--segments", required=True, type=Path, help="segment dir (repack output)")
    p.add_argument("--source", required=True, type=Path, help="source checkpoint snapshot")
    p.add_argument("--policy", required=True, type=Path, help="fq-policy JSON (bits_per_expert)")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--layers", default=None)
    args = p.parse_args(argv)

    policy = json.loads(args.policy.read_text())
    bpe = policy["bits_per_expert"]
    ks = sorted({k for v in bpe.values() for k in v})
    args.out.mkdir(parents=True, exist_ok=True)

    wanted = None
    if args.layers:
        wanted = set()
        for part in args.layers.split(","):
            if "-" in part:
                a, b = part.split("-")
                wanted.update(range(int(a), int(b) + 1))
            else:
                wanted.add(int(part))

    total = {}
    for src_shard in sorted(args.source.glob("model-layer-*.safetensors")):
        layer = int(src_shard.stem.split("-")[-1])
        if wanted is not None and layer not in wanted:
            continue
        lb = bpe.get(str(layer))
        if lb is None:
            # dense layer (or out-of-policy): copy through
            out = args.out / src_shard.name
            if not out.exists():
                out.write_bytes(src_shard.read_bytes())
            continue
        readers = {}
        for k in sorted(set(lb)):
            seg = args.segments / f"layer-{layer:03d}.k{k}.safetensors"
            if not seg.exists():
                raise FileNotFoundError(f"missing segment {seg} for layer {layer}")
            readers[k] = SegmentReader(seg)
        counts = assemble_layer(
            src_shard, args.out / src_shard.name, layer,
            lambda e: lb[e], readers)
        for r in readers.values():
            r.close()
        total[layer] = counts
        print(f"layer {layer:3d}: {counts}", flush=True)

    for extra in args.source.iterdir():
        if extra.is_file() and not extra.name.startswith("model-layer-"):
            dst = args.out / extra.name
            if not dst.exists():
                dst.write_bytes(extra.read_bytes())
    print(f"assembled {len(total)} MoE layers -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
