#!/usr/bin/env bash
# Assemble the full 3.0bpw GLM-5.2 checkpoint from K3 Progressive Tensors
# segments, in layer batches that respect the disk floor.
#
# Why batches: fq_assemble stages its ENTIRE output before the final swap, so
# one full-model run would need 316.4 GB of staging at once.  This box has
# ~325 GB free with a 120 GB floor, so the whole model cannot be staged (nor
# stored) as a second independent physical copy.  Each batch is assembled for
# real, its shards are hashed by the assembler itself (fq-assembly.json
# products), compared against the source MANIFEST, and then materialized into
# the final checkpoint as a whole-file reflink clone of the byte-identical
# source blob -- provably the same bytes, at zero incremental disk.
set -euo pipefail

REPO=/home/mbelleau/protensors-work/vllm-voipmonitor
PY=/home/mbelleau/venvs/fq/bin/python
SEG=/home/mbelleau/fq-segments/GLM-5.2-EXL3-FQ
SRC=/home/mbelleau/.cache/huggingface/hub/models--brandonmusic--GLM-5.2-EXL3-TR3-3.0bpw/snapshots/9297b9f1d53af5c67cffa01e30cc071a1ff7144b
POLICY=$REPO/research/fungible-quant/runs/m5-serve/policy-k3-uniform.json
SIGNER=a58b7bb79ba5845716aa6fee7d54e714ef243c2875f23a617e1ef3247c565525
OUT=/home/mbelleau/glm52-k3-assembled
STAGE=/home/mbelleau/glm52-k3-stage
EVID=$REPO/research/fungible-quant/runs/m5-serve/evidence
FLOOR=$((150 * 1000 * 1000 * 1000))   # abort well above the 120 GB hard floor

mkdir -p "$OUT" "$EVID"

free_bytes() { df -B1 --output=avail /home | tail -1 | tr -d ' '; }

check_floor() {
  local f; f=$(free_bytes)
  if [ "$f" -lt "$FLOOR" ]; then
    echo "ABORT: free=$f below floor=$FLOOR" >&2; exit 9
  fi
  echo "[disk] free=$(( f / 1000000000 )) GB"
}

echo "=== start $(date -u +%FT%TZ) ==="; check_floor

for RANGE in 0-2 3-12 13-22 23-32 33-42 43-52 53-62 63-72 73-78; do
  echo "--- batch $RANGE $(date -u +%FT%TZ) ---"
  check_floor
  rm -rf "$STAGE"
  $PY "$REPO/research/fungible-quant/tools/fq_assemble.py" \
    --segments "$SEG" --source "$SRC" --policy "$POLICY" \
    --out "$STAGE" --layers "$RANGE" --trust-signer "$SIGNER"

  # Compare the assembler's OWN hashes of the bytes it just wrote against the
  # source MANIFEST, then materialize each verified shard by reflinking the
  # byte-identical source blob (zero incremental disk).
  $PY - "$RANGE" <<'PYEOF'
import json, os, subprocess, sys
from pathlib import Path
rng = sys.argv[1]
STAGE = Path("/home/mbelleau/glm52-k3-stage")
OUT   = Path("/home/mbelleau/glm52-k3-assembled")
SRC   = Path("/home/mbelleau/.cache/huggingface/hub/models--brandonmusic--GLM-5.2-EXL3-TR3-3.0bpw/snapshots/9297b9f1d53af5c67cffa01e30cc071a1ff7144b")
EVID  = Path("/home/mbelleau/protensors-work/vllm-voipmonitor/research/fungible-quant/runs/m5-serve/evidence")

manifest = {}
for line in (SRC / "MANIFEST.sha256").read_text().splitlines():
    if not line.strip():
        continue
    sha, name = line.split(None, 1)
    manifest[name.strip().lstrip("*")] = sha

rec = json.loads((STAGE / "fq-assembly.json").read_text())
# products: [{"file": ..., "sha256": ...}] -- sha256 of the bytes the
# assembler actually wrote, recomputed by the tool from the output tree.
products = {p["file"]: p["sha256"] for p in rec["products"]}
rows, bad = [], []
for name, sha in sorted(products.items()):
    if not (name.startswith("model-") and name.endswith(".safetensors")):
        continue
    src_sha = manifest.get(name)
    ok = (src_sha == sha)
    rows.append({"shard": name, "assembled_sha256": sha,
                 "source_sha256": src_sha, "bit_exact": ok})
    if not ok:
        bad.append(name)

if bad:
    print(f"DIVERGENT SHARDS in batch {rng}: {bad}", file=sys.stderr)
    # keep the divergent assembled bytes for forensics instead of discarding
    for n in bad:
        os.replace(STAGE / n, OUT / n)
    sys.exit(8)

for r in rows:
    n = r["shard"]
    dst = OUT / n
    if dst.exists():
        dst.unlink()
    # whole-file reflink: offset 0 -> 4K-congruent, so XFS really shares extents
    subprocess.run(["cp", "--reflink=always", str((SRC / n).resolve()), str(dst)],
                   check=True)

(EVID / f"batch-{rng}.json").write_text(json.dumps(
    {"range": rng, "shards": rows,
     "segments_verified": rec["verification"]["segments_verified"],
     "trusted_signers": rec["verification"]["trusted_signers"],
     "mode": rec["verification"]["mode"]}, indent=1) + "\n")
print(f"batch {rng}: {len(rows)} shards bit-exact, materialized by reflink")
PYEOF

  # keep the last batch's config/metadata + assembly record for the final merge
  rm -rf "$EVID/last-meta"; mkdir -p "$EVID/last-meta"
  for f in "$STAGE"/*; do
    case "$(basename "$f")" in
      model-layer-*.safetensors) ;;
      *) cp -a "$f" "$EVID/last-meta/" ;;
    esac
  done
  rm -rf "$STAGE"
  check_floor
done

echo "=== batches done $(date -u +%FT%TZ) ==="
