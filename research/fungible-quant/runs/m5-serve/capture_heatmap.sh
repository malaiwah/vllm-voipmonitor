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
# fq-heatmap/1: layers[] each carrying a per-expert vector. Report the
# activation mass, because that is the only thing separating a real sample
# from a well-formed empty one.
rows=d.get("layers") or []
def vec(r):
    if isinstance(r,dict):
        for k in ("counts","experts","values","activations","mass"):
            if isinstance(r.get(k),list): return r[k]
        return []
    return r if isinstance(r,list) else []
tot=sum(sum(v for v in vec(r) if isinstance(v,(int,float))) for r in rows)
nz=sum(1 for r in rows for v in vec(r) if isinstance(v,(int,float)) and v)
print(f"  {sys.argv[1].split('/')[-1]}: {len(rows)} layers, step={d.get('step')}, "
      f"{nz} active cells, {tot:,.0f} activation mass")
PY
  else
    echo "  sample $i: endpoint unavailable"
  fi
  [ "$i" -lt "$N" ] && sleep "$GAP"
done
echo "captured $(ls "$OUT"/heatmap-*.json 2>/dev/null | wc -l) samples in $OUT"
