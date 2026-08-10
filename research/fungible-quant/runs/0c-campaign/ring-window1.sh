#!/bin/bash
# GLM-5.2 window-1 encode ring: K2 + K5 for layers 3-10 (complement matrix).
set -u
until [ -f /home/mbelleau/glm52-capture/capture_run_manifest.json ]; do sleep 15; done
echo "[$(date -u +%FT%TZ)] capture sealed — ring begins"
run_k() { # bits gpus workers
  /home/mbelleau/protensors-work/vllm-voipmonitor/research/fungible-quant/runs/gg-env/gg-run.sh python /home/mbelleau/protensors-work/vllm-voipmonitor/research/fungible-quant/tools/fruit-encoder/fruit_encode_driver.py \
    --encode --bits $1 --workers $3 --gpus $2 \
    --src "/home/mbelleau/.cache/huggingface/hub/models--zai-org--GLM-5.2/snapshots/b4734de4facf877f85769a911abafc5283eab3d9/" --work /home/mbelleau/glm52-work-k$1 \
    --capture-dir /home/mbelleau/glm52-capture --layers 3-10 \
    > /home/mbelleau/protensors-work/vllm-voipmonitor/research/fungible-quant/runs/0c-campaign/glm52-encode-k$1.log 2>&1
  echo "[$(date -u +%FT%TZ)] GLM-5.2 K$1 window-1 exit $?"
}
run_k 2 4,5 2 &
run_k 5 6,7 2 &
wait
echo "[$(date -u +%FT%TZ)] window-1 ring complete (K2+K5, layers 3-10)"
