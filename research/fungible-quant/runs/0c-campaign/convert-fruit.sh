#!/bin/bash
# 0c proxy stage 1: flat K$1 conversion of SIQ-Fruit on GPU $2 (resumable).
set -u
K=$1; GPU=$2
OUT=/home/mbelleau/fq-0c/fruit-k$K
WORK=/home/mbelleau/fq-0c/work-k$K
mkdir -p "$OUT" "$WORK"
attempt=0
while true; do
  attempt=$((attempt+1))
  CUDA_VISIBLE_DEVICES=$GPU /home/mbelleau/protensors-work/vllm-voipmonitor/research/fungible-quant/runs/gg-env/gg-run.sh python /home/mbelleau/src/exllamav3/convert.py \
    -i "/home/mbelleau/.cache/huggingface/hub/models--malaiwah--GLM-5.2-SIQ-Fruit-Instruct-bf16/snapshots/678954f65e056a0f508e21eeb9251c655bb9463f/" -w "$WORK" -o "$OUT" -b $K -pm $( [ -d "$WORK/out_tensor" ] && echo -resume ) && break
  echo "[$(date -u +%FT%TZ)] convert k$K attempt $attempt failed; retry in 15s" >&2
  sleep 15
done
touch "$OUT/.convert-complete"
echo "[$(date -u +%FT%TZ)] fruit K$K conversion complete"
