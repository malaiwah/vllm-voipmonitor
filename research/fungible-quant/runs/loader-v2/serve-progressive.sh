#!/bin/bash
# serve-progressive.sh — boot Fruit DIRECTLY from Progressive Tensors
# segments + a bitrate policy (loader v2): --load-format progressive, no
# assembled checkpoint on disk.
#
# Usage: serve-progressive.sh <policy.json> <served_name> <port> [extra args...]
# Env overrides: VLLM_FQ_MANIFEST_DIR, VLLM_FQ_DENSE_SOURCE, VLLM_FQ_CACHE.
set -eu
POLICY=$1; NAME=$2; PORT=$3; shift 3

GG=$(cd "$(dirname "$0")/../gg-env" && pwd)/gg-run.sh
export VLLM_FQ_MANIFEST_DIR=${VLLM_FQ_MANIFEST_DIR:-/home/mbelleau/fq-0c/fruit-segments}
export VLLM_FQ_POLICY=$POLICY
export VLLM_FQ_DENSE_SOURCE=${VLLM_FQ_DENSE_SOURCE:-/home/mbelleau/fq-0c/fruit-k3}
export VLLM_FQ_CACHE=${VLLM_FQ_CACHE:-/home/mbelleau/cache/fq}

# Synthesize the mixed hybrid_tr3_tail + quantization_config stub from the
# policy (tier bitmap goes under $VLLM_FQ_CACHE/boot, referenced by absolute
# path) — the GG loader contract of runs/serve-baseline/fruit-mixed-report.md.
OV=$("$GG" python -m vllm.model_executor.layers.quantization.exl3_fungible.progressive \
  --extra-overrides '{"use_index_cache":true}')

export CUDA_VISIBLE_DEVICES=0,1,2,3
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_USE_B12X_MOE=1
export VLLM_USE_B12X_SPARSE_INDEXER=1

exec "$GG" python -m vllm.entrypoints.openai.api_server \
  --model "$VLLM_FQ_DENSE_SOURCE" \
  --served-model-name "$NAME" \
  --host 127.0.0.1 \
  --port "$PORT" \
  --trust-remote-code \
  --tensor-parallel-size 4 \
  --quantization exl3 \
  --load-format progressive \
  --attention-backend B12X_MLA_SPARSE \
  --moe-backend b12x \
  --max-model-len 4096 \
  --max-num-seqs 4 \
  --gpu-memory-utilization 0.30 \
  --kv-cache-dtype fp8_ds_mla \
  --hf-overrides "$OV" \
  "$@"
