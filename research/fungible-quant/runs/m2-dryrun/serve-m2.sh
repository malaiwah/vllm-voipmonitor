#!/bin/bash
# serve-m2.sh — Fruit mixed-042 TP4 serve for the M2 dryrun evidence runs.
# Usage: serve-m2.sh <off|on-default|on-evidence> [extra vllm args...]
#   off         VLLM_FQ_ENABLE=0 (A/B baseline)
#   on-default  FQ on, spec-default interval 3000 (M1 overhead leg)
#   on-evidence FQ on, short interval 200 + dwell 250 (M2 decision evidence)
# All modes: fq_reload worker extension (memory RPC) + a fresh
# PROMETHEUS_MULTIPROC_DIR so /metrics aggregates worker-process metrics.
set -u
BASE=/home/mbelleau/protensors-work/vllm-voipmonitor/research/fungible-quant
MODE=$1; shift || true

export PROMETHEUS_MULTIPROC_DIR=/home/mbelleau/fq-0c/fq-prom-m2
rm -rf "$PROMETHEUS_MULTIPROC_DIR"; mkdir -p "$PROMETHEUS_MULTIPROC_DIR"

# Private JIT caches: the gg-run defaults are SHARED with the 0c encoder
# campaign (GPUs 4-7, 4 writer processes). Every serve boot after the
# encoder started (20:12) died with an illegal memory access in the first
# traffic waves — including VLLM_FQ_ENABLE=0 and a known-good-replica
# boot — pointing at concurrent JIT cache mutation, not the FQ code.
# Fresh private dirs remove that coupling (slower first JIT, then warm).
export CUDA_CACHE_PATH=/home/mbelleau/cache/jit-m2/cuda
export TRITON_CACHE_DIR=/home/mbelleau/cache/jit-m2/triton
mkdir -p "$CUDA_CACHE_PATH" "$TRITON_CACHE_DIR"

case "$MODE" in
  off)
    export VLLM_FQ_ENABLE=0
    ;;
  on-default)
    export VLLM_FQ_ENABLE=1 VLLM_FQ_APPLY_MODE=dryrun
    ;;
  on-evidence)
    export VLLM_FQ_ENABLE=1 VLLM_FQ_APPLY_MODE=dryrun
    export VLLM_FQ_INTERVAL_STEPS=200 VLLM_FQ_DWELL_STEPS=250
    ;;
  *) echo "unknown mode: $MODE" >&2; exit 2 ;;
esac
if [ "$MODE" != off ]; then
  export VLLM_FQ_POLICY=/home/mbelleau/fq-0c/policy-fruit-mixed-042.json
  export VLLM_FQ_EPS_ROOT=/home/mbelleau/fq-0c
  export VLLM_FQ_CACHE_ROOT=/home/mbelleau/fq-0c/fq-cache-m2
fi

# Mitigations for the box-level MLA query-BMM cuBLAS flake (status 14 in
# safe_query_bmm.cu) that appears whenever the 0c encoder campaign is
# hammering GPUs 4-7: keep the caching allocator on plain cudaMalloc
# (no expandable_segments VMM) and load all CUDA modules eagerly at init
# instead of lazily on first use.
export CUDA_MODULE_LOADING=EAGER

# Dev-mode endpoints (/collective_rpc, /pause, /resume) for the memory
# RPC and the M3 reload path.
export VLLM_SERVER_DEV_MODE=1

GG="$BASE/runs/gg-env/gg-run.sh"
export CUDA_VISIBLE_DEVICES=0,1,2,3
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_B12X_MOE=1
export VLLM_USE_B12X_SPARSE_INDEXER=1
exec "$GG" python -m vllm.entrypoints.openai.api_server \
  --model /home/mbelleau/fq-0c/fruit-mixed-042 \
  --served-model-name fruit-mixed \
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
  --kv-cache-dtype fp8_ds_mla --hf-overrides '{"use_index_cache":true}' \
  --worker-extension-cls fq_reload.FqReloadWorker \
  "$@"
