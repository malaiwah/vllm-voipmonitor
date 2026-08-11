#!/bin/bash
# eval_gpqa.sh — GPQA Diamond (198 items) against a live OpenAI-compatible serve.
#
# Usage: eval_gpqa.sh <endpoint> <output_dir> [extra lm_eval args...]
#   eval_gpqa.sh http://127.0.0.1:8000 /path/to/out
#
# Env:
#   MODEL        served-model-name                  (default GLM-5.2)
#   BACKEND      lm-eval | bench                    (default lm-eval)
#   TASK         lm-eval task name                  (default gpqa_diamond_cot_zeroshot)
#   CONCURRENCY  in-flight requests                 (default 16)
#   MAX_GEN      generation cap per item, tokens    (default 6144)
#   API_KEY      bearer token if the serve wants one
#
# MAX_GEN fits serve-glm52.sh's default FQ_MAXLEN=8192: vLLM rejects a request
# whose prompt + max_tokens exceeds the window, and a GPQA item prompt runs to
# ~800 tokens. GPQA reasoning traces are long, so 6144 will still truncate some
# items and every truncation scores as wrong. To measure the model rather than
# the token budget, boot the serve with FQ_MAXLEN=32768 and set MAX_GEN=16384;
# either way, report the truncation count alongside the accuracy.
#
# BACKEND=lm-eval  EleutherAI lm-evaluation-harness, generative CoT, chat
#                  endpoint, exact-match on "Answer: <letter>". Externally
#                  comparable to published GPQA Diamond numbers.
# BACKEND=bench    local-inference-lab llm_decode_bench.py --test-profile
#                  gpqa-diamond. Same 198 items, adds Wilson CI, per-category
#                  accuracy, completion-token percentiles and paired McNemar
#                  A/B against an earlier run (--compare-baseline).
#
# All 198 items are always run: at ~5 min/item worst case the set is small
# enough that subsampling would only cost statistical power for no real saving.
set -euo pipefail
ENDPOINT=${1:?usage: eval_gpqa.sh <endpoint> <output_dir> [extra args...]}
OUTDIR=${2:?usage: eval_gpqa.sh <endpoint> <output_dir> [extra args...]}
shift 2

MODEL=${MODEL:-GLM-5.2}
BACKEND=${BACKEND:-lm-eval}
TASK=${TASK:-gpqa_diamond_cot_zeroshot}
CONCURRENCY=${CONCURRENCY:-16}
MAX_GEN=${MAX_GEN:-6144}
LMEVAL=/home/mbelleau/venvs/lmeval/bin/lm_eval
BENCH=/home/mbelleau/bench/llm-inference-bench/llm_decode_bench.py
BENCHPY=/home/mbelleau/venvs/bench/bin/python

mkdir -p "$OUTDIR"

# Datasets are already cached; going offline makes a network hiccup during a
# paid-for serve window impossible rather than merely unlikely.
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export HF_DATASETS_TRUST_REMOTE_CODE=1

if [ "$BACKEND" = bench ]; then
  exec "$BENCHPY" "$BENCH" \
    --host "$ENDPOINT" --model "$MODEL" \
    --test-profile gpqa-diamond \
    --max-tokens "$MAX_GEN" \
    --profile-concurrency "$CONCURRENCY" \
    --display-mode plain --no-hw-monitor \
    --output "$OUTDIR/gpqa_diamond_bench.json" "$@"
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
  "$@"
