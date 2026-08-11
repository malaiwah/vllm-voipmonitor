#!/usr/bin/env bash
# Assemble the mixed K3/K5 GLM-5.2 checkpoint from Progressive Tensors
# segments, in layer batches that respect the disk floor.
#
# Same batching rationale as assemble-full.sh, but the materialization rule is
# the general one:
#   * a shard whose assembled sha256 EQUALS the source shard (a pure-K3 layer)
#     is stored as a whole-file XFS reflink clone of the byte-identical source
#     blob -- provably the same bytes, zero incremental disk;
#   * a shard that genuinely DIFFERS (a K5-bearing layer) is renamed into the
#     final checkpoint, so the delivered file is literally the assembler's
#     output.  rename() on the same filesystem is free, so the only real disk
#     cost is the mixed shards themselves.
set -euo pipefail

REPO=/home/mbelleau/protensors-work/vllm-voipmonitor
PY=/home/mbelleau/venvs/fq/bin/python
SEG=/home/mbelleau/fq-segments-mixed-k3k5
SRC=/home/mbelleau/.cache/huggingface/hub/models--brandonmusic--GLM-5.2-EXL3-TR3-3.0bpw/snapshots/9297b9f1d53af5c67cffa01e30cc071a1ff7144b
POLICY=$REPO/research/fungible-quant/runs/m5-serve/policy-mixed-k3k5.json
SIGNER=a58b7bb79ba5845716aa6fee7d54e714ef243c2875f23a617e1ef3247c565525
OUT=/home/mbelleau/glm52-mixed-k3k5
STAGE=/home/mbelleau/glm52-mixed-stage
EVID=$REPO/research/fungible-quant/runs/m5-serve/evidence-mixed
FLOOR=$((150 * 1000 * 1000 * 1000))

mkdir -p "$OUT" "$EVID"
free_bytes() { df -B1 --output=avail /home | tail -1 | tr -d ' '; }
check_floor() {
  local f; f=$(free_bytes)
  if [ "$f" -lt "$FLOOR" ]; then echo "ABORT: free=$f below floor=$FLOOR" >&2; exit 9; fi
  echo "[disk] free=$(( f / 1000000000 )) GB"
}

echo "=== start $(date -u +%FT%TZ) ==="; check_floor

# Batch order and size are disk-driven, not cosmetic.  Pure-K3 layers cost
# zero permanent disk (reflink) so they go FIRST; the 12 K5-bearing layers are
# the only ones that consume real bytes (~4.68 GB each) so they go LAST, when
# the transient staging is smallest.  Batches are kept to <=6 layers so that
# check_floor's 150 GB gate minus one batch's staging still clears the 120 GB
# hard floor even if the concurrent campaign takes space mid-run.
for RANGE in 0-2 3-8 9-14 15-20 21-26 27-32 33-34 \
             47-52 53-58 59-64 65-70 71-76 77-78 \
             35-37 38-40 41-43 44-46; do
  echo "--- batch $RANGE $(date -u +%FT%TZ) ---"
  check_floor
  rm -rf "$STAGE"
  $PY "$REPO/research/fungible-quant/tools/fq_assemble.py" \
    --segments "$SEG" --source "$SRC" --policy "$POLICY" \
    --out "$STAGE" --layers "$RANGE" --trust-signer "$SIGNER" --reflink

  $PY - "$RANGE" <<'PYEOF'
import json, os, subprocess, sys
from pathlib import Path
rng = sys.argv[1]
STAGE = Path("/home/mbelleau/glm52-mixed-stage")
OUT   = Path("/home/mbelleau/glm52-mixed-k3k5")
SRC   = Path("/home/mbelleau/.cache/huggingface/hub/models--brandonmusic--GLM-5.2-EXL3-TR3-3.0bpw/snapshots/9297b9f1d53af5c67cffa01e30cc071a1ff7144b")
EVID  = Path("/home/mbelleau/protensors-work/vllm-voipmonitor/research/fungible-quant/runs/m5-serve/evidence-mixed")

manifest = {}
for line in (SRC / "MANIFEST.sha256").read_text().splitlines():
    if line.strip():
        sha, name = line.split(None, 1)
        manifest[name.strip().lstrip("*")] = sha

rec = json.loads((STAGE / "fq-assembly.json").read_text())
products = {p["file"]: p["sha256"] for p in rec["products"]}
rows = []
for name, sha in sorted(products.items()):
    if not (name.startswith("model-") and name.endswith(".safetensors")):
        continue
    src_sha = manifest.get(name)
    same = (src_sha == sha)
    dst = OUT / name
    if dst.exists():
        dst.unlink()
    if same:
        subprocess.run(["cp", "--reflink=always", str((SRC / name).resolve()), str(dst)],
                       check=True)
        how = "reflink-from-source"
    else:
        os.replace(STAGE / name, dst)      # rename: free, keeps assembler bytes
        how = "moved-assembler-output"
    rows.append({"shard": name, "assembled_sha256": sha, "source_sha256": src_sha,
                 "identical_to_source": same, "materialized": how,
                 "bytes": dst.stat().st_size})

(EVID / f"batch-{rng}.json").write_text(json.dumps(
    {"range": rng, "shards": rows,
     "segments_verified": rec["verification"]["segments_verified"],
     "trusted_signers": rec["verification"]["trusted_signers"],
     "mode": rec["verification"]["mode"]}, indent=1) + "\n")
n_new = sum(1 for r in rows if not r["identical_to_source"])
print(f"batch {rng}: {len(rows)} shards ({n_new} genuinely mixed, "
      f"{len(rows)-n_new} identical->reflink)")
PYEOF

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
