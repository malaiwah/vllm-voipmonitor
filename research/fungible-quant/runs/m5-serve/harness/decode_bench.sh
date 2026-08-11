#!/bin/bash
# decode_bench.sh — local-inference-lab decode (TG) throughput matrix.
#
# Wraps https://github.com/local-inference-lab/llm-inference-bench
# (llm_decode_bench.py), cloned at /home/mbelleau/bench/llm-inference-bench.
# This is the tool the rtx6kpro wiki numbers are produced with, so a Progressive
# Tensors serve measured this way is directly comparable to models/glm5.2_v20.md.
#
# Usage: decode_bench.sh <endpoint> <output.json> [extra bench args...]
#   decode_bench.sh http://127.0.0.1:8000 /path/to/decode.json
#
# Env:
#   MODEL        served-model-name        (default GLM-5.2)
#   CONCURRENCY  comma list               (default 1,4,8,16,32)
#   CONTEXTS     comma list, tokens       (default 0,4096)
#   DURATION     seconds per matrix cell  (default 30)
#   MAX_TOKENS   generation cap           (default 1024)
#   HWMON        1 to keep the nvidia-smi hardware panel (default 0: off, so
#                the harness never touches the GPUs it is measuring)
#
# DEFAULTS ARE SIZED TO serve-glm52.sh: FQ_MAXLEN defaults to 8192, and vLLM
# rejects any request whose prompt + max_tokens exceeds the window, so
# context 4096 + 1024 generated leaves headroom. The top concurrency matches
# FQ_MAXSEQS (32); asking for more only measures queueing. Raise CONTEXTS only
# together with FQ_MAXLEN on the serve.
#
# Measures, per (concurrency, context) cell:
#   - Sustained Decode aggregate tok/s, from OpenAI stream cumulative
#     completion_tokens (exact when the server honours continuous_usage_stats;
#     the JSON records which source was used in results[].aggregate_source)
#   - per-request tok/s, TTFT, TTST, ITL, request latency (avg/p50/p90/p99)
#   - prefill tok/s as prompt_tokens/TTFT from the scout request per context
#   - Prometheus validation counters and effective concurrency when /metrics
#     is exposed
set -euo pipefail
ENDPOINT=${1:?usage: decode_bench.sh <endpoint> <output.json> [extra args...]}
OUT=${2:?usage: decode_bench.sh <endpoint> <output.json> [extra args...]}
shift 2

MODEL=${MODEL:-GLM-5.2}
CONCURRENCY=${CONCURRENCY:-1,4,8,16,32}
CONTEXTS=${CONTEXTS:-0,4096}
DURATION=${DURATION:-30}
MAX_TOKENS=${MAX_TOKENS:-1024}
BENCH=/home/mbelleau/bench/llm-inference-bench/llm_decode_bench.py
BENCHPY=/home/mbelleau/venvs/bench/bin/python

HW=(--no-hw-monitor)
[ "${HWMON:-0}" = 1 ] && HW=()

mkdir -p "$(dirname "$OUT")"
exec "$BENCHPY" "$BENCH" \
  --host "$ENDPOINT" \
  --model "$MODEL" \
  --concurrency "$CONCURRENCY" \
  --contexts "$CONTEXTS" \
  --duration "$DURATION" \
  --max-tokens "$MAX_TOKENS" \
  --display-mode plain \
  ${HW[@]+"${HW[@]}"} \
  --output "$OUT" \
  "$@"
