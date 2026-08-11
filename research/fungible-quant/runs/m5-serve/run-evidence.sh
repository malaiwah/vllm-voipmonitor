#!/bin/bash
# run-evidence.sh — sequence the M5 end-to-end evidence campaign.
#
#   run-evidence.sh <off|dryrun|live> <checkpoint-dir> [tag]
#
# Boots a serve, waits for it to be genuinely ready (not merely listening),
# records a baseline, saturates it while recording a metrics timeline, runs a
# quality eval, then shuts down and renders the charts. Every artifact lands
# under runs/m5-serve/results/<tag>/ so a run is reproducible and comparable.
#
# Deliberately fail-loud: if the serve does not come up, or the probe returns
# nonsense, the script stops rather than producing an empty-but-plausible
# result directory.
# -e matters here: without it the script exits 0 after ANY post-probe
# failure, because its last command is `ls | tee`. That produces exactly
# the empty-but-plausible result directory this header claims to prevent.
set -euo pipefail
BASE=/home/mbelleau/protensors-work/vllm-voipmonitor/research/fungible-quant
RUN=$BASE/runs/m5-serve
PY=/home/mbelleau/venvs/fq/bin/python

MODE=${1:?usage: run-evidence.sh <off|dryrun|live> <checkpoint-dir> [tag]}
CKPT=${2:?checkpoint dir required}
TAG=${3:-$MODE}
PORT=${FQ_PORT:-8000}
OUT=$RUN/results/$TAG
mkdir -p "$OUT"

BASE_URL=http://127.0.0.1:$PORT
say() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$OUT/run.log"; }

# ---------------------------------------------------------------- boot
say "booting serve mode=$MODE ckpt=$CKPT tag=$TAG"
# setsid: put the serve in its OWN process group so cleanup can signal the
# whole tree. Killing only the parent leaves the four TP workers alive holding
# ~76 GiB of GPU memory each, which then blocks every later run on this box.
# It must be a NEW group: without job control the script shares its own pgid
# with the child, so `kill -- -$$` would take out the script itself.
setsid env FQ_MODEL="$CKPT" FQ_PORT="$PORT" \
  "$RUN/serve-glm52.sh" "$MODE" > "$OUT/serve.log" 2>&1 &
SERVE_PID=$!
sleep 1
SERVE_PGID=$(ps -o pgid= -p "$SERVE_PID" 2>/dev/null | tr -d ' ')
say "serve pid=$SERVE_PID pgid=${SERVE_PGID:-unknown}"
cleanup() {
  local rc=$?
  say "shutting down serve pid=$SERVE_PID pgid=${SERVE_PGID:-?} (exit rc=$rc)"
  if [ -n "${SERVE_PGID:-}" ] && [ "$SERVE_PGID" != "$$" ]; then
    kill -TERM -- "-$SERVE_PGID" 2>/dev/null
    for _ in $(seq 1 30); do
      kill -0 -- "-$SERVE_PGID" 2>/dev/null || break
      sleep 2
    done
    kill -KILL -- "-$SERVE_PGID" 2>/dev/null
  else
    kill -9 "$SERVE_PID" 2>/dev/null
  fi
  # Verify, do not assume: a surviving worker silently poisons the next run.
  local left
  left=$(pgrep -f "VLLM::" | wc -l)
  [ "$left" -gt 0 ] && say "WARN: $left VLLM worker process(es) still alive"
  return $rc
}
trap cleanup EXIT

# Readiness: /health alone can pass before the model can generate, so we also
# require one real completion. A 600-iteration x 10s ceiling is ~100 minutes,
# which is generous for a 295 GiB checkpoint cold off disk.
say "waiting for readiness (this is a 295 GiB load; cold boots take a while)"
READY=0
for i in $(seq 1 600); do
  if ! kill -0 "$SERVE_PID" 2>/dev/null; then
    say "FATAL: serve process died during boot — see serve.log"
    tail -40 "$OUT/serve.log" | sed 's/^/    /' | tee -a "$OUT/run.log"
    exit 1
  fi
  if curl -sf -m 5 "$BASE_URL/health" >/dev/null 2>&1; then
    READY=1; say "health OK after ~$((i*10))s"; break
  fi
  sleep 10
done
[ "$READY" = 1 ] || { say "FATAL: never became healthy"; exit 1; }

# ---------------------------------------------------------------- probe
say "generation probe"
PROBE=$(curl -sf -m 300 "$BASE_URL/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{"model":"GLM-5.2","messages":[{"role":"user","content":"In one sentence, what is mixture-of-experts routing?"}],"max_tokens":64,"temperature":0}')
echo "$PROBE" > "$OUT/probe.json"
TEXT=$($PY -c "
import json,sys
try:
    d=json.load(open('$OUT/probe.json'))
    print(d['choices'][0]['message']['content'].strip()[:300])
except Exception as e:
    print('PROBE_PARSE_FAIL', e)
")
say "probe says: $TEXT"
case "$TEXT" in
  PROBE_PARSE_FAIL*|"") say "FATAL: probe produced no usable text"; exit 1 ;;
esac

# ---------------------------------------------------------------- baseline
say "baseline decode measurement"
$PY "$RUN/swap_evidence.py" both \
  --base "$BASE_URL" --out "$OUT/timeline-baseline.jsonl" \
  --interval 5 --concurrency "${FQ_BASE_CC:-8}" --max-tokens 128 \
  --phase "math:${FQ_BASE_SECS:-120}" 2>&1 | tee -a "$OUT/run.log"

# ---------------------------------------------------------------- saturate
# Four domains back to back. The point is router movement: a fixed prompt mix
# never shifts the hot-expert set, which is exactly why the M2 dryrun saw zero
# proposed swaps.
say "saturation + domain shift (this is the swap evidence window)"
$PY "$RUN/swap_evidence.py" both \
  --base "$BASE_URL" --out "$OUT/timeline-main.jsonl" \
  --interval 5 --concurrency "${FQ_CC:-24}" --max-tokens "${FQ_TOK:-256}" \
  --phase "math:${FQ_PHASE:-420}" \
  --phase "code:${FQ_PHASE:-420}" \
  --phase "prose_multiling:${FQ_PHASE:-420}" \
  --phase "biomed:${FQ_PHASE:-420}" 2>&1 | tee -a "$OUT/run.log"

# ---------------------------------------------------------------- snapshot
say "capturing final metrics + policy state"
curl -sf -m 15 "$BASE_URL/metrics" > "$OUT/metrics-final.txt" 2>/dev/null \
  || say "WARN: could not scrape final /metrics"
if [ -d "$RUN/artifacts" ]; then
  cp -a "$RUN/artifacts" "$OUT/fq-artifacts" 2>/dev/null \
    && say "copied loop artifacts (decision log, committed policy)"
fi
grep -aiE "FQ (interval|resolve|swap)|swap|rollback" "$OUT/serve.log" \
  > "$OUT/fq-lines.log" 2>/dev/null
say "fq log lines captured: $(wc -l < "$OUT/fq-lines.log" 2>/dev/null || echo 0)"

# ---------------------------------------------------------------- eval
# GSM8K by default, not GPQA: cost is tokens, not items. GPQA Diamond's 198
# graduate items run ~4000 tokens each; a 250-item GSM8K subsample runs ~800,
# making it roughly 4x cheaper. GPQA is opt-in via FQ_EVAL=gpqa.
EVAL=${FQ_EVAL:-gsm8k}
EVAL_SH=$RUN/harness/eval_${EVAL}.sh
if [ "${FQ_SKIP_EVAL:-0}" = 1 ]; then
  say "eval skipped by request (FQ_SKIP_EVAL=1)"
elif [ ! -x "$EVAL_SH" ]; then
  # Fail LOUD. The previous guard looked for run-gpqa.sh while the harness
  # ships eval_gpqa.sh, so the campaign logged "skipping eval" and produced
  # zero quality numbers while still exiting 0 — a silent hole in the very
  # evidence the run exists to gather.
  say "FATAL: eval script $EVAL_SH not found or not executable"
  ls -la "$RUN/harness/" | sed 's/^/    /' | tee -a "$OUT/run.log"
  exit 4
else
  say "quality eval ($EVAL) via $EVAL_SH"
  if "$EVAL_SH" "$BASE_URL" "$OUT/eval-$EVAL" 2>&1 | tee -a "$OUT/run.log"; then
    say "eval completed"
  else
    say "WARN: eval exited non-zero — results may be partial"
  fi
fi

# ---------------------------------------------------------------- charts
say "rendering charts"
$PY "$RUN/make_charts.py" "$OUT/timeline-main.jsonl" \
  --out "$OUT/swap-timeline.svg" \
  --title "Live expert re-tiering under load — $TAG" \
  2>&1 | tee -a "$OUT/run.log"

ls -la "$OUT" | sed 's/^/    /' | tee -a "$OUT/run.log" || true
# Assert the run actually produced evidence before claiming success.
rows=$(cat "$OUT"/timeline-*.jsonl 2>/dev/null | wc -l)
if [ "$rows" -lt 5 ]; then
  say "FATAL: only $rows timeline rows — refusing to report success"
  exit 5
fi
say "DONE — artifacts in $OUT ($rows timeline rows)"
