#!/bin/bash
# K2 to FULL model coverage (operator priority, 2026-08-10): finish the
# progressive fast-load tier across every MoE layer before spending GPU
# time on other tiers. K5 and the K4 complement backfill afterwards.
#
# Ring per window: capture (resume from preserved boundary) -> encode K2
# on all 4 GPUs -> publish -> delete that window's capture. Resumable:
# already-encoded layers are skipped by the driver's done-JSON check, and
# a killed run restarts at its window boundary.
set -u
RUNS=/home/mbelleau/protensors-work/vllm-voipmonitor/research/fungible-quant/runs
TOOLS=/home/mbelleau/protensors-work/vllm-voipmonitor/research/fungible-quant/tools/fruit-encoder
SNAP_O=$(ls -d /home/mbelleau/.cache/huggingface/hub/models--zai-org--GLM-5.2/snapshots/*/)
PY=/home/mbelleau/venvs/fq/bin/python

# layers 19..78 in windows of 8 (3-18 already encoded at K2)
for START in 19 27 35 43 51 59 67 75; do
  END=$(( START + 7 )); [ $END -gt 78 ] && END=78
  echo "[$(date -u +%FT%TZ)] === window $START-$END: capture ==="
  LAYERS=$START-$END STOP=$END "$RUNS/0c-campaign/run-capture-glm52.sh" \
    >> "$RUNS/0c-campaign/capture-glm52.log" 2>&1 || {
      echo "[$(date -u +%FT%TZ)] capture FAILED for $START-$END — stopping ring"; exit 1; }

  echo "[$(date -u +%FT%TZ)] === window $START-$END: encode K2 ==="
  "$RUNS/gg-env/gg-run.sh" python "$TOOLS/fruit_encode_driver.py" \
    --encode --bits 2 --workers 4 --gpus 4,5,6,7 \
    --src "$SNAP_O" --work /home/mbelleau/glm52-work-k2 \
    --capture-dir /home/mbelleau/glm52-capture --layers $START-$END \
    >> "$RUNS/0c-campaign/glm52-encode-k2.log" 2>&1
  echo "[$(date -u +%FT%TZ)] K2 $START-$END exit $?"

  echo "[$(date -u +%FT%TZ)] === window $START-$END: publish ==="
  $PY "$RUNS/0c-campaign/publish_window.py" \
    >> "$RUNS/0c-campaign/publish-k2.log" 2>&1 && {
      for L in $(seq -w $START $END); do rm -rf "/home/mbelleau/glm52-capture/layer_$L"; done
      echo "[$(date -u +%FT%TZ)] window $START-$END published + capture pruned"; }
  df -h /home | awk 'NR==2{print "  disk free: "$4}'
done
echo "[$(date -u +%FT%TZ)] K2 COMPLETE across layers 3-78"
