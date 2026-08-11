#!/bin/bash
# capture_heatmap.sh — sample the live activation matrix while traffic runs.
#
# The heatmap is only meaningful with real routing behind it: sampled against
# an idle model every cell is zero and the picture is a lie of omission. So
# this is meant to run ALONGSIDE load, not before it.
set -eu
BASE=${1:-http://127.0.0.1:8100}
OUT=${2:-results/bt/heatmap}
N=${3:-6}
GAP=${4:-45}
mkdir -p "$OUT"
for i in $(seq 1 "$N"); do
  ts=$(date -u +%H%M%S)
  f="$OUT/heatmap-$(printf %02d "$i")-$ts.json"
  if curl -fsS -m 20 "$BASE/fq/heatmap" -o "$f"; then
    # Report the only thing that distinguishes a real sample from an empty
    # one: how much routing mass it actually captured.
    python3 - "$f" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
m=d.get("matrix") or d.get("counts") or []
tot=sum(sum(r) for r in m) if m and isinstance(m[0],list) else 0
nz=sum(1 for r in m for v in r if v) if m and isinstance(m[0],list) else 0
print(f"  {sys.argv[1].split('/')[-1]}: {len(m)} layers, {nz} active cells, {tot:,.0f} total activations")
PY
  else
    echo "  sample $i: endpoint unavailable"
  fi
  [ "$i" -lt "$N" ] && sleep "$GAP"
done
echo "captured $(ls "$OUT"/heatmap-*.json 2>/dev/null | wc -l) samples in $OUT"
