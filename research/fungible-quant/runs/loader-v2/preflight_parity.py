"""Preflight: progressive stream (042 policy) must equal the assembled
fruit-mixed-042 checkpoint tensor-for-tensor, byte-for-byte."""
import importlib.util, json, struct, sys, time
from pathlib import Path

PKG = Path("/home/mbelleau/src/gg-vllm/vllm/model_executor/layers/quantization/exl3_fungible")
def load(name, fn):
    s = importlib.util.spec_from_file_location(name, PKG / fn)
    m = importlib.util.module_from_spec(s); sys.modules[name] = m
    s.loader.exec_module(m); return m
fr = load("fq_fragments_standalone", "fragments.py")
pg = load("fq_progressive_standalone", "progressive.py")

spec = pg.ProgressiveSpec.from_env(
    "/home/mbelleau/fq-0c/fruit-k3",
    environ={},
    overrides={
        "manifest_dir": "/home/mbelleau/fq-0c/fruit-segments",
        "policy": "/home/mbelleau/fq-0c/policy-fruit-mixed-042.json",
        "dense_source": "/home/mbelleau/fq-0c/fruit-k3",
    },
)
resolver = spec.make_resolver(environ={}, cache_dir="/tmp/claude-1000/fqcache-preflight")

# reference: assembled checkpoint
ASM = Path("/home/mbelleau/fq-0c/fruit-mixed-042")
def read_header(p):
    with open(p, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n

ref = {}  # name -> (path, body, entry)
for shard in sorted(ASM.glob("*.safetensors")):
    hdr, body = read_header(shard)
    hdr.pop("__metadata__", None)
    for name, t in hdr.items():
        ref[name] = (shard, body, t)

import mmap
mmaps = {}
def ref_bytes(name):
    shard, body, t = ref[name]
    if shard not in mmaps:
        f = open(shard, "rb")
        mmaps[shard] = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    a, b = t["data_offsets"]
    return mmaps[shard][body + a: body + b]

t0 = time.time()
count = 0
mismatch = 0
seen = set()
for name, tensor in pg.progressive_weights_iterator(spec, resolver, tp_rank=None, log=lambda m: print(m, flush=True)):
    seen.add(name)
    got = tensor.contiguous().view(-1).view(__import__("torch").uint8) if tensor.numel() else None
    import torch
    raw = bytes(tensor.contiguous().flatten().view(torch.uint8).numpy().tobytes()) if tensor.numel() else b""
    want = ref_bytes(name)
    if raw != bytes(want):
        mismatch += 1
        if mismatch < 5:
            print("MISMATCH", name, len(raw), len(want))
    count += 1
missing = set(ref) - seen
print(f"tensors={count} mismatches={mismatch} missing_from_stream={len(missing)} wall={time.time()-t0:.1f}s")
print("resolver stats:", resolver.stats)
if missing:
    print("sample missing:", sorted(missing)[:5])
sys.exit(1 if (mismatch or missing) else 0)
