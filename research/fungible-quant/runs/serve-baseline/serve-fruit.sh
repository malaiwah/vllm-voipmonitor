#!/bin/bash
# serve-fruit.sh — boot a Fruit proxy checkpoint TP4 on GPUs 0-3 (fq M0 mixed gate).
# Usage: serve-fruit.sh <model_dir> <served_name> [extra vllm args...]
set -u
MODEL=$1; NAME=$2; shift 2
GG=/home/mbelleau/protensors-work/vllm-voipmonitor/research/fungible-quant/runs/gg-env/gg-run.sh
export CUDA_VISIBLE_DEVICES=0,1,2,3
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_USE_B12X_MOE=1
export VLLM_USE_B12X_SPARSE_INDEXER=1
exec "$GG" python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --served-model-name "$NAME" \
  --host 127.0.0.1 \
  --port 8801 \
  --trust-remote-code \
  --tensor-parallel-size 4 \
  --quantization exl3 \
  --attention-backend B12X_MLA_SPARSE \
  --moe-backend b12x \
  --max-model-len 4096 \
  --max-num-seqs 4 \
  --gpu-memory-utilization 0.30 \
  "$@"
