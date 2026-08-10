#!/bin/bash
# Window-2 ring: capture layers 11-18 (resuming from the preserved
# boundary_011), then encode K2 (the progressive fast-load tier — the
# operator's priority) with K5 following if the window is still free.
set -u
RUNS=/home/mbelleau/protensors-work/vllm-voipmonitor/research/fungible-quant/runs
TOOLS=/home/mbelleau/protensors-work/vllm-voipmonitor/research/fungible-quant/tools/fruit-encoder
SNAP_O=$(ls -d /home/mbelleau/.cache/huggingface/hub/models--zai-org--GLM-5.2/snapshots/*/)

echo "[$(date -u +%FT%TZ)] window-2 capture: layers 11-18 (resume from boundary_011)"
LAYERS=11-18 STOP=18 $RUNS/0c-campaign/run-capture-glm52.sh || {
  echo "[$(date -u +%FT%TZ)] window-2 capture FAILED"; exit 1; }

echo "[$(date -u +%FT%TZ)] window-2 encodes: K2 first (fast-load tier)"
$RUNS/gg-env/gg-run.sh python $TOOLS/fruit_encode_driver.py \
  --encode --bits 2 --workers 4 --gpus 4,5,6,7 \
  --src "$SNAP_O" --work /home/mbelleau/glm52-work-k2 \
  --capture-dir /home/mbelleau/glm52-capture --layers 11-18 \
  > $RUNS/0c-campaign/glm52-encode-k2-w2.log 2>&1
echo "[$(date -u +%FT%TZ)] window-2 K2 exit $?"

$RUNS/gg-env/gg-run.sh python $TOOLS/fruit_encode_driver.py \
  --encode --bits 5 --workers 4 --gpus 4,5,6,7 \
  --src "$SNAP_O" --work /home/mbelleau/glm52-work-k5 \
  --capture-dir /home/mbelleau/glm52-capture --layers 11-18 \
  > $RUNS/0c-campaign/glm52-encode-k5-w2.log 2>&1
echo "[$(date -u +%FT%TZ)] window-2 K5 exit $?"

/home/mbelleau/venvs/fq/bin/python $RUNS/0c-campaign/publish_window.py \
  > $RUNS/0c-campaign/publish-window2.log 2>&1
echo "[$(date -u +%FT%TZ)] window-2 ring complete"
