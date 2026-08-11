#!/bin/bash
# run-4axis.sh — replay each MTP78 corpus axis separately against a live serve,
# with a clean stats boundary between them.
#
# Produces the two things still missing from the evidence set:
#   * the four per-axis routing dumps the flagship heatmap image is drawn from;
#   * a per-axis convergence score, which is what actually settles whether the
#     CODE axis matters most or whether any corpus traffic would do. Only one
#     axis has been measured so far, so "the corpus matters" is established
#     while "this axis matters more" is not.
#
# The stats file is rotated between axes rather than reset server-side: the
# policy loop reads the same decayed counters, and zeroing them mid-flight
# would perturb a running decision window. Rotating the DUMP gives a clean
# per-axis boundary without touching engine state.
set -euo pipefail
RUN=/home/mbelleau/protensors-work/vllm-voipmonitor/research/fungible-quant/runs/m5-serve
PY=/home/mbelleau/venvs/fq/bin/python
BASE=${FQ_BASE:-http://127.0.0.1:8000}
OUT=$RUN/results/axes
STATS=$OUT/stats.jsonl
CC=${FQ_CC:-16}
mkdir -p "$OUT"

say() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$OUT/run.log"; }

curl -sf -m 10 "$BASE/health" >/dev/null || { say "FATAL: serve not healthy at $BASE"; exit 1; }

AXES="axis1_general axis2_legal axis3_code_agentic axis4_reasoning_termination"
for AX in $AXES; do
  say "=== axis $AX"
  # Rotate the dump so this axis's records stand alone. The collector keeps
  # accumulating (its window is decayed, not reset) — what we isolate is the
  # OBSERVATION, and the scorer reads the last record, which is dominated by
  # the traffic that just ran.
  : > "$STATS"
  if ! $PY "$RUN/replay_mtp78.py" --base "$BASE" --axis "$AX" \
        --concurrency "$CC" --max-tokens 8 \
        --out "$OUT/replay-$AX.json" 2>&1 | tail -3 | tee -a "$OUT/run.log"; then
    say "WARN: replay failed for $AX — continuing to the next axis"
    continue
  fi
  cp "$STATS" "$OUT/stats-$AX.jsonl"
  n=$(wc -l < "$OUT/stats-$AX.jsonl")
  say "$AX: $n stats records captured"
  if [ "$n" -lt 2 ]; then
    say "WARN: $AX produced only $n records — too few to score meaningfully"
    continue
  fi
  # Score against the human reference. --signal mass is the point of this run
  # (gate mass is now recorded); the scorer fails loudly if it is absent.
  $PY "$RUN/score_convergence.py" --reference "$RUN/reference-coder-quant.json" \
      --stats "$OUT/stats-$AX.jsonl" --signal mass \
      --out "$OUT/convergence-$AX.json" 2>&1 | tee -a "$OUT/run.log" || \
    say "WARN: mass scoring failed for $AX (falling back to count)" && \
    $PY "$RUN/score_convergence.py" --reference "$RUN/reference-coder-quant.json" \
      --stats "$OUT/stats-$AX.jsonl" --signal count \
      --out "$OUT/convergence-$AX-count.json" 2>&1 | tee -a "$OUT/run.log" || true
done

say "=== summary"
$PY - "$OUT" <<'PY' 2>&1 | tee -a "$OUT/run.log"
import json, sys, glob, os
out = sys.argv[1]
rows = []
for f in sorted(glob.glob(os.path.join(out, "convergence-axis*.json"))):
    d = json.load(open(f))
    rows.append((os.path.basename(f), d.get("signal"), d.get("layers_scored"),
                 d.get("mean_per_layer_jaccard"), d.get("lift_over_chance"),
                 d.get("fraction_of_human_agreement")))
if not rows:
    print("no per-axis convergence scores produced")
else:
    print(f"{'file':44s} {'sig':5s} {'L':>3s} {'jaccard':>8s} {'xchance':>8s} {'%human':>7s}")
    for r in rows:
        print(f"{r[0]:44s} {str(r[1]):5s} {str(r[2]):>3s} {r[3]:>8} {str(r[4]):>8} {str(r[5]):>7}")
PY
say "DONE — artifacts in $OUT"
