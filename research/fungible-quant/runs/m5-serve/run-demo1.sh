#!/bin/bash
# run-demo1.sh — the convergence demo, end to end, in one command.
#
#   run-demo1.sh <checkpoint-dir> [tag]
#
# Boots GLM-5.2 with a seeded K3/K4 policy, replays the MTP78 calibration
# corpus the human reference was built from, runs GSM8K, then scores how well
# live routing rediscovered that human's expert-precision map and renders the
# charts. Everything lands under runs/m5-serve/results/<tag>/.
#
# Modelled on run-evidence.sh, which already learned these lessons the hard
# way, plus four gates that are specific to this demo:
#
#   1. The checkpoint must PHYSICALLY match the policy. A K3/K4 swap is a
#      1-for-1 trade inside a fixed-capacity mixed slab (swap.py
#      MixedLayerState / _validate_layer). A uniform-K3 layer has no K4 slab,
#      so a policy that declares K4 slots there cannot ever be applied — the
#      run would look healthy and produce zero promotions.
#   2. Every declared K4 slot must be backed by a real fragment. It is
#      make_scenario1_policy.py that guarantees this; here we re-check it,
#      because VLLM_FQ_K_FALLBACK=3 turns an unbacked slot into a silent
#      downgrade rather than an error.
#   3. A stale committed policy in the store WINS over VLLM_FQ_POLICY
#      (loop.build_from_env resolution order), so a re-run with an edited
#      policy would quietly execute the old one.
#   4. GPUs 0-3 must actually be free. Booting on top of a live serve is how
#      you lose two runs instead of one.
#
# Fail-loud everywhere: no step is allowed to leave an empty-but-plausible
# result directory behind.
set -euo pipefail

BASE=/home/mbelleau/protensors-work/vllm-voipmonitor/research/fungible-quant
RUN=$BASE/runs/m5-serve
PY=/home/mbelleau/venvs/fq/bin/python

CKPT=${1:?usage: run-demo1.sh <checkpoint-dir> [tag]}
TAG=${2:-demo1}
PORT=${FQ_PORT:-8000}
OUT=$RUN/results/$TAG
BASE_URL=http://127.0.0.1:$PORT

# serve-glm52.sh hardcodes these two; we need the paths to clear stale state
# before boot and to collect artifacts after. Drift between the two files
# would silently point us at the wrong store, so assert it rather than
# assume it.
CACHE_ROOT=/home/mbelleau/fq-0c/fq-cache-m5
ARTIFACT_DIR=$RUN/artifacts

# Segment dirs the resolver may pull K4 fragments from, colon separated.
# Default adds the primed 3.42bpw K4 family (layers 3-10) to the published
# K2/K5 family. Override to point at our own K4 encode once it lands.
FQ_K4_DIRS=${FQ_K4_DIRS:-/home/mbelleau/fq-primed/segments-342/expanded}
# Fraction of each layer's K4 fragment pool occupied at boot. MUST be < 1:
# a saturated pool has zero legal promotion targets, so the loop can never
# trade and the demo shows a flat line that looks like a bug.
FILL=${FQ_FILL:-0.5}
# Per-rank memory the demo is allowed to spend above uniform K3.
MAX_GIB=${FQ_MAX_GIB:-8.835}

mkdir -p "$OUT"
say() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$OUT/run.log"; }
die() { say "FATAL: $*"; exit 1; }
have() { [ -s "$1" ] || die "missing or empty: $1${2:+ ($2)}"; }

say "=== demo 1: live re-tiering vs the human 3.42bpw Coder quant ==="
say "ckpt=$CKPT tag=$TAG port=$PORT out=$OUT"

# ------------------------------------------------------------- 0. preflight
say "--- step 0: preflight"
[ -d "$CKPT" ] || die "no checkpoint dir $CKPT"
for f in "$RUN/make_scenario1_policy.py" "$RUN/score_convergence.py" \
         "$RUN/make_charts.py" "$RUN/replay_mtp78.py" "$RUN/swap_evidence.py" \
         "$RUN/reference-coder-quant.json" \
         "$RUN/harness/load_mtp78_corpus.py"; do
  have "$f"
done
[ -x "$RUN/serve-glm52.sh" ]      || die "serve-glm52.sh not executable"
[ -x "$RUN/harness/eval_gsm8k.sh" ] || die "harness/eval_gsm8k.sh not executable"
[ -x "$BASE/runs/gg-env/gg-run.sh" ] || die "gg-run.sh not executable"
have /home/mbelleau/.fq_keys/fq_signing.pub \
  "serve-glm52.sh refuses to boot with trust filtering silently disabled"

grep -q "VLLM_FQ_CACHE_ROOT=$CACHE_ROOT" "$RUN/serve-glm52.sh" \
  || die "serve-glm52.sh no longer uses cache root $CACHE_ROOT — this script's
      stale-policy check would target the wrong store"
grep -q 'VLLM_FQ_ARTIFACT_DIR=\$RUN/artifacts' "$RUN/serve-glm52.sh" \
  || die "serve-glm52.sh no longer writes artifacts to $ARTIFACT_DIR"

if curl -sf -m 3 "$BASE_URL/health" >/dev/null 2>&1; then
  die "something is already serving on port $PORT — refusing to boot a second
      engine on the same GPUs. Set FQ_PORT and free GPUs 0-3 first."
fi
if [ "${FQ_ALLOW_BUSY_GPU:-0}" != 1 ]; then
  busy=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
         | awk -F', ' '$1 < 4 && $2 > 2000 {print $1":"$2"MiB"}' | tr '\n' ' ')
  [ -z "$busy" ] || die "GPUs 0-3 are not free ($busy). Stop the other serve,
      or set FQ_ALLOW_BUSY_GPU=1 if you know those bytes are yours."
fi
free_gb=$(df -B1G --output=avail /home | tail -1 | tr -d ' ')
say "disk free: ${free_gb} GB"
[ "$free_gb" -ge 20 ] || die "under 20 GB free — logs and eval samples need room"

# ------------------------------------------------------- 1. seeded policy
say "--- step 1: build the seeded policy from real K4 fragment coverage"
POLICY=$OUT/policy-demo1.json
COVERAGE=$OUT/k4-coverage.json
COV_ARGS=(--coverage-dir /home/mbelleau/glm52-segments)
IFS=':' read -r -a _k4dirs <<< "$FQ_K4_DIRS"
for d in "${_k4dirs[@]}"; do
  [ -n "$d" ] && COV_ARGS+=(--coverage-dir "$d")
done
$PY "$RUN/make_scenario1_policy.py" \
  --mode seeded --fill-fraction "$FILL" --max-extra-gib "$MAX_GIB" --tp 4 \
  "${COV_ARGS[@]}" --out "$POLICY" --coverage-out "$COVERAGE" \
  2>&1 | tee -a "$OUT/run.log"
have "$POLICY"; have "$COVERAGE"

read -r N_SLOTS N_COVERED N_TRADABLE MANIFEST <<< "$($PY - "$POLICY" <<'PYEOF'
import json, sys
d = json.load(open(sys.argv[1]))
b = d["budget"]["n_k4_per_layer"]
prov = d["provenance"]
# A layer can only trade if its budget leaves at least one unoccupied
# fragment to promote INTO: swaps are 1-for-1 inside a fixed-capacity K4
# slab, so budget == pool means zero legal targets forever.
rows = prov.get("per_layer_coverage", [])
tradable = sum(1 for r in rows if int(r.get("promotion_candidates", 0)) > 0)
print(sum(b.values()), len(prov["covered_layers"]), tradable, d["manifest"])
PYEOF
)"
say "policy: $N_SLOTS K4 slots over $N_COVERED covered layers ($N_TRADABLE
    tradable), manifest ${MANIFEST:0:16}"
[ "$N_SLOTS" -gt 0 ] || die "the policy declares zero K4 slots — with no K4
      fragments on disk there is nothing to promote. Point FQ_K4_DIRS at a
      dir holding layer-NNN.k4.safetensors + index-k4.json, or run the
      observe-mode experiment instead (see demo1-README.md)."
# Header gate 1 restated in numbers: a saturated pool has no legal promotion
# target, so the loop shows a flat line for the whole run and the operator
# reads it as a bug. Refuse before spending an hour of GPU on it.
[ "$N_TRADABLE" -gt 0 ] || die "every K4-covered layer saturates its fragment
      pool (budget == pool size), so the loop has ZERO legal promotion
      targets and can never swap. Lower FQ_FILL (currently $FILL) or widen
      FQ_K4_DIRS."
[ "$N_TRADABLE" -eq "$N_COVERED" ] || say "WARNING: only $N_TRADABLE of
    $N_COVERED covered layer(s) have promotion candidates; the rest saturate
    their fragment pool and cannot swap"
# ---- policy gate end (test_run_demo1.py extracts the block above)

# --------------------------------------- 2. checkpoint matches the policy
say "--- step 2: verify the checkpoint physically carries the declared K4 slabs"
$PY - "$CKPT" "$POLICY" "$OUT/checkpoint-match.json" <<'PYEOF' \
  2>&1 | tee -a "$OUT/run.log"
"""The gate that decides whether this run can produce evidence at all.

A promotion moves an expert between two pre-allocated slabs; capacity is
fixed when the layer loads. So the checkpoint's own tier bitmap must declare
K4 on exactly the experts the policy declares K4 on. Anything else and the
swap engine either refuses the layer (validate) or silently never promotes.
"""
import json, sys
from pathlib import Path

ckpt, policy_path, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
policy = json.loads(policy_path.read_text())
bmp_path = ckpt / "tier_bitmap.json"
if not bmp_path.is_file():
    sys.exit(f"FATAL: {bmp_path} missing — cannot tell what tiers this "
             f"checkpoint physically holds")
bmp = json.loads(bmp_path.read_text())

rows, bad = [], []
for layer, bits in sorted(policy["bits_per_expert"].items(), key=lambda kv: int(kv[0])):
    want = {e for e, b in enumerate(bits) if b == 4}
    entry = bmp.get(layer)
    if entry is None:
        bad.append(f"L{layer}: absent from the checkpoint's tier_bitmap")
        continue
    have_bits = entry["bits_per_expert"] if isinstance(entry, dict) else entry
    have = {e for e, b in enumerate(have_bits) if int(b) == 4}
    rows.append({"layer": int(layer), "policy_k4": len(want),
                 "checkpoint_k4": len(have), "match": want == have})
    if want != have:
        bad.append(f"L{layer}: policy wants {len(want)} K4 experts, "
                   f"checkpoint holds {len(have)} "
                   f"({len(want - have)} declared-but-absent, "
                   f"{len(have - want)} present-but-undeclared)")

out.write_text(json.dumps({"checkpoint": str(ckpt), "layers": rows,
                           "mismatches": bad}, indent=1))
if bad:
    print("\n".join("  " + b for b in bad[:12]))
    if len(bad) > 12:
        print(f"  ... and {len(bad) - 12} more")
    sys.exit(
        "FATAL: checkpoint does not carry the policy's K4 slabs.\n"
        "  A uniform-K3 checkpoint CANNOT promote: there is no K4 slab to\n"
        "  promote into, and no runtime flag changes that. Assemble a mixed\n"
        "  checkpoint against this exact policy first:\n"
        f"    fq_assemble.py --segments <k3+k4 family> --source <base>\\\n"
        f"      --policy {policy_path} --out <mixed ckpt> --reflink\n"
        "  See demo1-README.md 'what is still missing'.")
n = sum(r["policy_k4"] for r in rows)
print(f"checkpoint matches the policy on all {len(rows)} layers ({n} K4 experts)")
PYEOF

# ------------------------------------------------- 3. clear stale run state
say "--- step 3: clear stale loop state for manifest ${MANIFEST:0:16}"
STORE=$CACHE_ROOT/fq/policy/$MANIFEST
# FQ_DRY_RUN validates every input that can be checked without a GPU and
# stops before touching one. Placed AFTER the two gates above and BEFORE any
# mutation of the shared store, so a dry run can never disturb a live run.
if [ "${FQ_DRY_RUN:-0}" = 1 ]; then
  say "DRY RUN: inputs validated, stopping before boot"
  [ -e "$STORE/current.json" ] && say "note: a committed policy exists at
      $STORE and would be moved aside by a real run"
  say "DRY RUN OK — policy $POLICY, coverage $COVERAGE"
  exit 0
fi
if [ -e "$STORE/current.json" ]; then
  if [ "${FQ_KEEP_STORE:-0}" = 1 ]; then
    say "WARNING: reusing committed policy in $STORE (FQ_KEEP_STORE=1) — the
        boot policy on disk will be IGNORED in favour of it"
  else
    mv "$STORE" "$STORE.superseded.$(date -u +%s)"
    say "moved a previously committed policy aside; boot policy will be used"
  fi
fi
rm -rf "$ARTIFACT_DIR"; mkdir -p "$ARTIFACT_DIR"

# ------------------------------------------------------------- 4. boot
say "--- step 4: boot the serve (live apply mode)"
STATS=$OUT/stats.jsonl
setsid env \
  FQ_MODEL="$CKPT" FQ_PORT="$PORT" \
  VLLM_FQ_POLICY="$POLICY" \
  VLLM_FQ_DUMP_STATS="$STATS" \
  VLLM_FQ_LOCAL_SEGMENTS="$FQ_K4_DIRS" \
  VLLM_FQ_TABLE_EVERY_INTERVALS="${FQ_TABLE_EVERY:-1}" \
  FQ_INTERVAL="${FQ_INTERVAL:-200}" FQ_DWELL="${FQ_DWELL:-250}" \
  "$RUN/serve-glm52.sh" live > "$OUT/serve.log" 2>&1 &
SERVE_PID=$!
sleep 1
# `|| SERVE_PGID=""` is load-bearing: under `set -o pipefail` a serve that
# died inside the first second makes `ps` exit 1, the assignment fails, and
# `set -e` kills this script BEFORE the EXIT trap below is installed and
# before the readiness loop can tail serve.log. Empty is a state cleanup
# already handles (it falls back to killing SERVE_PID directly).
SERVE_PGID=$(ps -o pgid= -p "$SERVE_PID" 2>/dev/null | tr -d ' ') || SERVE_PGID=""
say "serve pid=$SERVE_PID pgid=${SERVE_PGID:-unknown}"

SCRAPE_PID=""
cleanup() {
  local rc=$?
  [ -n "$SCRAPE_PID" ] && kill -INT "$SCRAPE_PID" 2>/dev/null
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
  local left
  left=$(pgrep -f "VLLM::" | wc -l)
  [ "$left" -gt 0 ] && say "WARN: $left VLLM worker process(es) still alive"
  return $rc
}
trap cleanup EXIT

say "waiting for readiness (295 GiB checkpoint; a cold boot takes a while)"
READY=0
for i in $(seq 1 600); do
  if ! kill -0 "$SERVE_PID" 2>/dev/null; then
    say "serve process died during boot — last 40 log lines:"
    tail -40 "$OUT/serve.log" | sed 's/^/    /' | tee -a "$OUT/run.log"
    die "boot failed"
  fi
  if curl -sf -m 5 "$BASE_URL/health" >/dev/null 2>&1; then
    READY=1; say "health OK after ~$((i*10))s"; break
  fi
  sleep 10
done
[ "$READY" = 1 ] || die "never became healthy"

# ------------------------------------------------------------- 5. probe
say "--- step 5: generation probe"
curl -sf -m 300 "$BASE_URL/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{"model":"GLM-5.2","messages":[{"role":"user","content":"In one sentence, what is mixture-of-experts routing?"}],"max_tokens":64,"temperature":0}' \
  > "$OUT/probe.json" || die "probe request failed"
TEXT=$($PY -c "
import json
try:
    d = json.load(open('$OUT/probe.json'))
    print(d['choices'][0]['message']['content'].strip()[:300])
except Exception as e:
    print('PROBE_PARSE_FAIL', e)
")
say "probe says: $TEXT"
case "$TEXT" in
  PROBE_PARSE_FAIL*|"") die "probe produced no usable text — a checkpoint whose
      tiers do not match its policy answers with garbage, so stop here" ;;
esac

# ------------------------------------- 6. startup composition + metrics tap
say "--- step 6: record the startup composition table"
curl -sf -m 15 "$BASE_URL/metrics" > "$OUT/metrics-start.txt" 2>/dev/null \
  || say "WARN: could not scrape start /metrics"
cp "$POLICY" "$OUT/policy-boot.json"
# The loop prints a layer x K occupancy table (occupancy_table.py) on the
# intervals selected by VLLM_FQ_TABLE_EVERY_INTERVALS. Capture what exists at
# t0 so the final table can be diffed against a recorded start rather than a
# remembered one.
grep -aiE "occupancy|composition|FQ loop:" "$OUT/serve.log" \
  > "$OUT/composition-start.txt" 2>/dev/null || true
say "startup composition lines: $(wc -l < "$OUT/composition-start.txt")"

$PY "$RUN/swap_evidence.py" scrape --base "$BASE_URL" \
  --out "$OUT/timeline.jsonl" --interval "${FQ_SCRAPE_INTERVAL:-5}" \
  --duration "${FQ_SCRAPE_MAX:-14400}" >> "$OUT/run.log" 2>&1 &
SCRAPE_PID=$!
say "metrics scraper pid=$SCRAPE_PID -> timeline.jsonl"

# ----------------------------------------------------- 7. replay the corpus
# This is the traffic the experiment is actually about: the exact
# reap_recall_calib.jsonl bytes the reference quant was calibrated on. A
# different prompt mix measures convergence against the wrong distribution.
say "--- step 7: replay the MTP78 calibration corpus"
REPLAY_ARGS=(--base "$BASE_URL" --out "$OUT/replay.json"
             --concurrency "${FQ_CC:-24}" --max-tokens "${FQ_REPLAY_TOK:-8}")
[ -n "${FQ_AXIS:-}" ]  && REPLAY_ARGS+=(--axis "$FQ_AXIS")
[ -n "${FQ_LIMIT:-}" ] && REPLAY_ARGS+=(--limit "$FQ_LIMIT")
$PY "$RUN/replay_mtp78.py" "${REPLAY_ARGS[@]}" 2>&1 | tee -a "$OUT/run.log"
have "$OUT/replay.json"

# ------------------------------------------------------------- 8. GSM8K
say "--- step 8: GSM8K"
if [ "${FQ_SKIP_EVAL:-0}" = 1 ]; then
  say "eval skipped by request (FQ_SKIP_EVAL=1)"
else
  if "$RUN/harness/eval_gsm8k.sh" "$BASE_URL" "$OUT/eval-gsm8k" \
       2>&1 | tee -a "$OUT/run.log"; then
    say "eval completed"
  else
    say "WARN: eval exited non-zero — quality numbers may be partial"
  fi
  ls "$OUT/eval-gsm8k" >/dev/null 2>&1 \
    || say "WARN: eval produced no output directory"
fi

# ------------------------------------------------------------- 9. snapshot
say "--- step 9: dump stats and loop state"
# `|| true` on the kill, not just the wait: the scraper exits on its own once
# --duration (FQ_SCRAPE_MAX, default 4 h) elapses, and killing a process that
# is already gone returns 1 — which under `set -e` would abort the run HERE,
# after the whole experiment, before scoring and charts.
kill -INT "$SCRAPE_PID" 2>/dev/null || true
wait "$SCRAPE_PID" 2>/dev/null || true
SCRAPE_PID=""
curl -sf -m 15 "$BASE_URL/metrics" > "$OUT/metrics-final.txt" 2>/dev/null \
  || say "WARN: could not scrape final /metrics"
[ -d "$ARTIFACT_DIR" ] && cp -a "$ARTIFACT_DIR" "$OUT/fq-artifacts" 2>/dev/null \
  && say "copied loop artifacts (decision log, committed policy)"
[ -f "$STORE/current.json" ] && cp "$STORE/current.json" "$OUT/policy-final.json" \
  && say "copied the loop's final committed policy"
grep -aiE "FQ (interval|resolve|swap|loop)|swap|rollback|fallback|substitut" \
  "$OUT/serve.log" > "$OUT/fq-lines.log" 2>/dev/null || true
grep -aiE "occupancy|composition" "$OUT/serve.log" \
  > "$OUT/composition-final.txt" 2>/dev/null || true
say "fq log lines: $(wc -l < "$OUT/fq-lines.log" 2>/dev/null || echo 0)"

have "$STATS" "the loop never dumped routing stats — VLLM_FQ_DUMP_STATS did
      not reach the engine, or no decision interval was ever reached"
ROWS=$(wc -l < "$STATS")
say "stats intervals recorded: $ROWS"
[ "$ROWS" -ge 2 ] || die "only $ROWS stats interval(s) — too little traffic to
      rank experts. Raise FQ_LIMIT or lower FQ_INTERVAL."

# `grep -c` prints 0 AND exits 1 when nothing matches, so the obvious
# `$(grep -c ... || echo 0)` yields the two-line string "0\n0", `[ -eq ]`
# then errors with "integer expression expected", and the || branch fires —
# printing the fallback-bitrate WARNING on every clean run. Put the fallback
# on the assignment instead, where it only replaces a genuinely empty value.
SUBS=$(grep -aic "substitut" "$OUT/fq-lines.log" 2>/dev/null) || SUBS=0
[ "$SUBS" -eq 0 ] || say "WARNING: $SUBS fragment substitution line(s) — some
    promotions were served at a FALLBACK bitrate (VLLM_FQ_K_FALLBACK=3), so
    the occupancy table overstates what is physically loaded"

# --------------------------------------------------- 10. score convergence
say "--- step 10: score convergence against the human reference"
SIGNAL=${FQ_SIGNAL:-mass}
# The collector aliases `mass` to `count` when no topk-weights getter is
# bound, and the two are indistinguishable to a consumer except by the arrays
# happening to be equal — which a uniform router would also produce. Every
# run before gg-vllm 6e08f683d ranked on hit frequency under a "mass" label.
# Ask the dump rather than assume.
MASS_REAL=$($PY -c "
import json
recs = [json.loads(l) for l in open('$STATS') if l.strip()]
print(bool(recs[-1].get('mass_is_real')))
" 2>/dev/null || echo Unknown)
say "mass_is_real=$MASS_REAL (scoring signal: $SIGNAL)"
if [ "$SIGNAL" = mass ] && [ "$MASS_REAL" != True ]; then
  say "WARNING: scoring --signal mass while the collector reports mass is NOT
      real gate mass. The number below is a hit-frequency ranking wearing a
      mass label. Also scoring --signal count so the two are comparable."
  $PY "$RUN/score_convergence.py" \
    --reference "$RUN/reference-coder-quant.json" --stats "$STATS" \
    --signal count --out "$OUT/convergence-count.json" \
    2>&1 | tee -a "$OUT/run.log" || say "WARN: count-signal scoring failed"
fi
$PY "$RUN/score_convergence.py" \
  --reference "$RUN/reference-coder-quant.json" --stats "$STATS" \
  --signal "$SIGNAL" --out "$OUT/convergence.json" \
  2>&1 | tee -a "$OUT/run.log" \
  || die "scoring failed — see run.log"
have "$OUT/convergence.json"

# Circular layers must not be quoted as convergence: on those the only K4
# fragments available are the reference's own picks.
$PY - "$POLICY" "$OUT/convergence.json" "$OUT/SCOPE.md" <<'PYEOF' \
  2>&1 | tee -a "$OUT/run.log"
import json, sys
from pathlib import Path
policy = json.loads(Path(sys.argv[1]).read_text())
conv = json.loads(Path(sys.argv[2]).read_text())
prov = policy["provenance"]
circ = prov.get("circular_layers", [])
cov = prov.get("covered_layers", [])
lines = [
    "# Scope of this run",
    "",
    f"- Convergence scored on **{conv['layers_scored']} layers** from routing "
    "alone. Routing needs no K4 weights, so this leg is at full strength.",
    f"- Live promotion physically possible on **{len(cov)} layers**: {cov}.",
]
if circ:
    lines += [
        f"- Of those, **{len(circ)} are CIRCULAR**: {circ}. Their K4 fragment "
        "pool is a subset of the reference's own K4 set, so the loop can only "
        "promote experts the human already chose. Treat promotion there as a "
        "MECHANISM demonstration; it is not evidence of agreement.",
    ]
non_circ = [l for l in cov if l not in circ]
lines += [f"- Non-circular live-promotion layers: **{len(non_circ)}** "
          f"{non_circ}."]
Path(sys.argv[3]).write_text("\n".join(lines) + "\n")
print("\n".join(lines))
PYEOF

# --------------------------------------------------------- 11. charts
say "--- step 11: render charts"
if [ "$(wc -l < "$OUT/timeline.jsonl" 2>/dev/null || echo 0)" -ge 5 ]; then
  $PY "$RUN/make_charts.py" "$OUT/timeline.jsonl" \
    --out "$OUT/swap-timeline.svg" \
    --title "Live expert re-tiering vs a human quant — $TAG" \
    2>&1 | tee -a "$OUT/run.log"
  have "$OUT/swap-timeline.svg"
else
  die "timeline has fewer than 5 samples — the metrics scraper never got a
      usable /metrics; refusing to render a chart of nothing"
fi

# --------------------------------------------------------- 12. verdict
ls -la "$OUT" | sed 's/^/    /' | tee -a "$OUT/run.log" || true
$PY - "$OUT/convergence.json" "$OUT/replay.json" <<'PYEOF' | tee -a "$OUT/run.log"
import json, sys
c = json.load(open(sys.argv[1]))
r = json.load(open(sys.argv[2]))
print(f"corpus         : {r['corpus']} axis={r['axis']} "
      f"{r['ok']}/{r['prompts']} ok")
print(f"mean Jaccard   : {c['mean_per_layer_jaccard']:.4f} "
      f"(chance {c['chance_floor_expected']:.4f}, "
      f"human-human {c.get('human_human_ceiling')})")
print(f"lift over chance: {c.get('lift_over_chance')}x, "
      f"{c.get('fraction_of_human_agreement')} of human agreement")
PYEOF
say "DONE — artifacts in $OUT ($ROWS stats intervals)"
