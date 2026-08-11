#!/bin/bash
# deploy-fq.sh — push the working-tree exl3_fungible into the GG rootfs.
#
# WHY THIS EXISTS
# The serve does NOT run /home/mbelleau/src/gg-vllm. It runs the vLLM inside
# the extracted r33 rootfs, whose site-packages carries a COPY of our package
# (the M2 dryrun deploy). Editing and committing the source tree therefore has
# no effect on the next boot — silently. That cost a full boot cycle: a serve
# came up "successfully" without the composition table, the stats dump, or the
# histc sentinel fix, all of which had been committed hours earlier.
#
# Run this after ANY change to exl3_fungible or the two hook files, before
# booting a serve. It prints a diff summary so a no-op deploy is visible.
set -euo pipefail

SRC=${FQ_SRC:-/home/mbelleau/src/gg-vllm}
ROOT=${FQ_ROOTFS:-/home/mbelleau/rootfs/gg-v20-r33}
DEST=$ROOT/opt/venv/lib/python3.12/site-packages/vllm

PKG_REL=model_executor/layers/quantization/exl3_fungible
HOOKS=(
  "model_executor/model_loader/__init__.py"
  "v1/worker/gpu_worker.py"
)

[ -d "$SRC/vllm/$PKG_REL" ] || { echo "no source package at $SRC/vllm/$PKG_REL" >&2; exit 2; }
[ -d "$DEST/$PKG_REL" ] || { echo "no deployed package at $DEST/$PKG_REL" >&2; exit 2; }

echo "=== deploying exl3_fungible: $SRC -> $DEST"

changed=0
for f in "$SRC/vllm/$PKG_REL"/*.py; do
  b=$(basename "$f")
  d="$DEST/$PKG_REL/$b"
  if [ ! -f "$d" ]; then
    echo "  NEW  $b"
    changed=$((changed + 1))
  elif ! cmp -s "$f" "$d"; then
    echo "  DIFF $b ($(diff <(cat "$d") <(cat "$f") | grep -c '^[<>]') changed lines)"
    changed=$((changed + 1))
  fi
done
for h in "${HOOKS[@]}"; do
  if [ -f "$SRC/vllm/$h" ] && [ -f "$DEST/$h" ] && ! cmp -s "$SRC/vllm/$h" "$DEST/$h"; then
    echo "  DIFF $h"
    changed=$((changed + 1))
  fi
done

if [ "$changed" -eq 0 ]; then
  echo "  already in sync — nothing to deploy"
  exit 0
fi

# Stale .pyc for a module whose .py we replace is a real hazard here: the
# rootfs python may prefer a cached bytecode with a matching mtime.
rm -rf "$DEST/$PKG_REL/__pycache__"
cp -f "$SRC/vllm/$PKG_REL"/*.py "$DEST/$PKG_REL/"
for h in "${HOOKS[@]}"; do
  [ -f "$SRC/vllm/$h" ] && cp -f "$SRC/vllm/$h" "$DEST/$h"
done

echo "  deployed $changed file(s)"
echo "=== verifying import in the rootfs runtime"
CUDA_VISIBLE_DEVICES="" "$(dirname "$0")/gg-run.sh" python -c "
import importlib, sys
base = 'vllm.model_executor.layers.quantization.exl3_fungible'
mods = ['stats', 'policy', 'store', 'decision_log', 'fragments', 'progressive',
        'progressive_loader', 'swap', 'lazy_encode', 'loop', 'integration',
        'occupancy_table']
missing = []
for m in mods:
    try:
        importlib.import_module(f'{base}.{m}')
    except Exception as e:
        missing.append(f'{m}: {type(e).__name__}: {e}')
if missing:
    print('IMPORT FAILURES:'); [print('  ' + x) for x in missing]; sys.exit(1)
print(f'  all {len(mods)} modules import OK in the rootfs')
" 2>&1 | grep -viE "^\s*$|warning" | tail -5
