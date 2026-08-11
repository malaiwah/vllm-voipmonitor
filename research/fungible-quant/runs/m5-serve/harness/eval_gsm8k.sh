#!/bin/bash
# eval_gsm8k.sh — GSM8K against a live OpenAI-compatible serve.
#
# Usage: eval_gsm8k.sh <endpoint> <output_dir> [extra lm_eval args...]
#   eval_gsm8k.sh http://127.0.0.1:8000 /path/to/out
#
# Env:
#   MODEL        served-model-name                  (default GLM-5.2)
#   BACKEND      lm-eval | bench                    (default lm-eval)
#   TASK         lm-eval task name                  (default gsm8k_cot_zeroshot;
#                use gsm8k_cot for the 8-shot CoT variant)
#   ITEMS        subsample size, 0 = all 1319       (default 250)
#   SEED         subsample seed                     (default 1234)
#   CONCURRENCY  in-flight requests                 (default 16)
#   MAX_GEN      generation cap per item, tokens    (default 3072)
#   API_KEY      bearer token if the serve wants one
#
# SUBSAMPLE: the default is 250 of 1319 items, chosen with a fixed seed, NOT
# the first 250 (lm-eval's --limit takes a prefix, and the split is not
# shuffled). 250 items gives a 95% CI of roughly +/-5 pp on an absolute
# accuracy near 90%, which is enough to catch a serve that is broken but not
# enough to resolve a 1-2 pp quantization delta -- for that, set ITEMS=0 and
# run the full set, or use BACKEND=bench with --compare-baseline for paired
# McNemar statistics. Any subsampled run MUST be reported as a subsample.
set -euo pipefail
ENDPOINT=${1:?usage: eval_gsm8k.sh <endpoint> <output_dir> [extra args...]}
OUTDIR=${2:?usage: eval_gsm8k.sh <endpoint> <output_dir> [extra args...]}
shift 2

MODEL=${MODEL:-GLM-5.2}
BACKEND=${BACKEND:-lm-eval}
TASK=${TASK:-gsm8k_cot_zeroshot}
ITEMS=${ITEMS:-250}
SEED=${SEED:-1234}
CONCURRENCY=${CONCURRENCY:-16}
MAX_GEN=${MAX_GEN:-3072}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LMEVAL=/home/mbelleau/venvs/lmeval/bin/lm_eval
LMEVALPY=/home/mbelleau/venvs/lmeval/bin/python
BENCH=/home/mbelleau/bench/llm-inference-bench/llm_decode_bench.py
BENCHPY=/home/mbelleau/venvs/bench/bin/python

mkdir -p "$OUTDIR"
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export HF_DATASETS_TRUST_REMOTE_CODE=1

if [ "$BACKEND" = bench ]; then
  # llm_decode_bench --profile-runs N is a deterministic evenly-spread slice,
  # its own (also defensible) answer to the same subsampling problem.
  RUNS=()
  [ "$ITEMS" -gt 0 ] && RUNS=(--profile-runs "$ITEMS")
  exec "$BENCHPY" "$BENCH" \
    --host "$ENDPOINT" --model "$MODEL" \
    --test-profile gsm8k \
    --max-tokens "$MAX_GEN" \
    --profile-concurrency "$CONCURRENCY" \
    ${RUNS[@]+"${RUNS[@]}"} \
    --display-mode plain --no-hw-monitor \
    --output "$OUTDIR/gsm8k_bench.json" "$@"
fi

SAMPLES=()
if [ "$ITEMS" -gt 0 ]; then
  HF_HUB_OFFLINE=$HF_HUB_OFFLINE "$LMEVALPY" "$HERE/subsample.py" \
    --task "$TASK" --n "$ITEMS" --seed "$SEED" \
    --out "$OUTDIR/gsm8k_subsample_${ITEMS}_seed${SEED}.json"
  SAMPLES=(--samples "$OUTDIR/gsm8k_subsample_${ITEMS}_seed${SEED}.json")
  echo "NOTE: this is a ${ITEMS}-item SUBSAMPLE of GSM8K test (1319), seed ${SEED}." \
    | tee "$OUTDIR/SUBSAMPLE.txt"
fi

[ -n "${API_KEY:-}" ] && export OPENAI_API_KEY="$API_KEY"
exec "$LMEVAL" \
  --model local-chat-completions \
  --model_args "base_url=${ENDPOINT%/}/v1/chat/completions,model=${MODEL},num_concurrent=${CONCURRENCY},max_retries=3,timeout=3600,tokenized_requests=False" \
  --tasks "$TASK" \
  --apply_chat_template \
  --gen_kwargs "max_gen_toks=${MAX_GEN},temperature=0" \
  --output_path "$OUTDIR" \
  --log_samples \
  --seed 1234 \
  ${SAMPLES[@]+"${SAMPLES[@]}"} \
  "$@"
