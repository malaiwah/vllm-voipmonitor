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
set -uo pipefail
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
FQ_MODEL=$CKPT FQ_PORT=$PORT "$RUN/serve-glm52.sh" "$MODE" \
  > "$OUT/serve.log" 2>&1 &
SERVE_PID=$!
cleanup() {
  say "shutting down serve pid=$SERVE_PID"
  kill "$SERVE_PID" 2>/dev/null
  # the API server spawns TP workers; give them a chance to exit cleanly
  for _ in $(seq 1 30); do kill -0 "$SERVE_PID" 2>/dev/null || break; sleep 2; done
  kill -9 "$SERVE_PID" 2>/dev/null
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
if [ "${FQ_SKIP_EVAL:-0}" != 1 ] && [ -x "$RUN/harness/run-gpqa.sh" ]; then
  say "quality eval (GPQA Diamond)"
  "$RUN/harness/run-gpqa.sh" "$BASE_URL" "$OUT/gpqa.json" \
    2>&1 | tee -a "$OUT/run.log"
else
  say "skipping eval (FQ_SKIP_EVAL=${FQ_SKIP_EVAL:-0}, harness present=$([ -x "$RUN/harness/run-gpqa.sh" ] && echo yes || echo no))"
fi

# ---------------------------------------------------------------- charts
say "rendering charts"
$PY "$RUN/make_charts.py" "$OUT/timeline-main.jsonl" \
  --out "$OUT/swap-timeline.svg" \
  --title "Live expert re-tiering under load — $TAG" \
  2>&1 | tee -a "$OUT/run.log"

say "DONE — artifacts in $OUT"
ls -la "$OUT" | sed 's/^/    /' | tee -a "$OUT/run.log"
