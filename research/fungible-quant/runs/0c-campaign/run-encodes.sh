#!/bin/bash
# Fruit 4-K encode fan-out: one GPU per K, all from the sealed capture.
set -u
run_k() { # bits gpu
  /home/mbelleau/protensors-work/vllm-voipmonitor/research/fungible-quant/runs/gg-env/gg-run.sh python /home/mbelleau/protensors-work/vllm-voipmonitor/research/fungible-quant/tools/fruit-encoder/fruit_encode_driver.py \
    --encode --bits $1 --workers 1 --gpus $2 \
    --capture-dir /home/mbelleau/fq-0c/capture \
    > /home/mbelleau/protensors-work/vllm-voipmonitor/research/fungible-quant/runs/0c-campaign/encode-k$1.log 2>&1
  echo "[$(date -u +%FT%TZ)] K$1 encode exit $?"
}
run_k 3 4 &
run_k 4 6 &
run_k 2 7 &
# K5 takes GPU 5 once T1 has verdicted
( until grep -qE "^T1 (PASS|FAIL)" /home/mbelleau/protensors-work/vllm-voipmonitor/research/fungible-quant/runs/t1-graph-freeze/t1.log 2>/dev/null; do sleep 30; done
  run_k 5 5 ) &
wait
echo "[$(date -u +%FT%TZ)] all K encodes finished"
