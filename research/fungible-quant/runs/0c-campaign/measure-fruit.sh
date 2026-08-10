#!/bin/bash
# 0c proxy stage 2: per-target dKL measurement K3 vs K4 against BF16 ref.
set -u
until [ -f /home/mbelleau/fq-0c/fruit-k3/.convert-complete ] && [ -f /home/mbelleau/fq-0c/fruit-k4/.convert-complete ]; do sleep 30; done
attempt=0
while true; do
  attempt=$((attempt+1))
  CUDA_VISIBLE_DEVICES=6 /home/mbelleau/protensors-work/vllm-voipmonitor/research/fungible-quant/runs/gg-env/gg-run.sh python /home/mbelleau/src/exllamav3/util/measure.py 2>/dev/null \
    || CUDA_VISIBLE_DEVICES=6 /home/mbelleau/protensors-work/vllm-voipmonitor/research/fungible-quant/runs/gg-env/gg-run.sh python - <<'PY'
import sys
sys.argv = ["measure_model",
    "-i", "/home/mbelleau/fq-0c/fruit-k3", "/home/mbelleau/fq-0c/fruit-k4",
    "-r", "/home/mbelleau/.cache/huggingface/hub/models--malaiwah--GLM-5.2-SIQ-Fruit-Instruct-bf16/snapshots/678954f65e056a0f508e21eeb9251c655bb9463f/",
    "-o", "/home/mbelleau/fq-0c/measurement.json",
    "-l", "3"]
from exllamav3.conversion.measure_model import parser, main, prepare
_args = parser.parse_args()
_in, _js, ok, err = prepare(_args)
if not ok:
    raise SystemExit(f"prepare failed: {err}")
main(_in, _js)
PY
  [ -f /home/mbelleau/fq-0c/measurement.json ] && break
  echo "[$(date -u +%FT%TZ)] measure attempt $attempt failed; retry in 20s" >&2
  sleep 20
done
echo "[$(date -u +%FT%TZ)] 0c proxy measurement complete"
