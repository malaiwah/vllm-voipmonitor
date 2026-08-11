#!/bin/bash
# campaign-supervisor.sh — self-driving multi-K quantization campaign.
#
# Runs forever with NO operator and NO Claude session: picks the next unit
# of work, grabs whatever GPUs are idle, encodes, publishes to HF, prunes,
# repeats. Designed to keep a rented box busy for as long as the rental
# lasts, and to be safely restartable at any moment.
#
#   tmux new -d -s fq -n campaign  (or use the existing fq session)
#   tmux send-keys -t fq:campaign '<this script>' C-m
#
# Resilience:
#  * Idempotent: work already done is detected from encoder done-JSONs and
#    skipped, so a kill -9 / preemption / reboot loses at most one layer.
#  * Self-healing: any failing step is retried on the next pass instead of
#    aborting the campaign.
#  * GPU-adaptive: uses every GPU that is idle at the moment an encode
#    starts, so freeing a serve automatically speeds the campaign up.
#  * Capture is deterministic (validated bit-exact), so a window
#    re-captured for a later tier yields byte-identical activations —
#    tiers stay hessian-identical even when encoded days apart.
#
# Tier order is the operator's: K2 to full coverage first (the progressive
# fast-load base is useless partial), then K5, then the K4 complement that
# community priming does not already cover.
set -u
RUNS=/home/mbelleau/protensors-work/vllm-voipmonitor/research/fungible-quant/runs
CAMP=$RUNS/0c-campaign
TOOLS=/home/mbelleau/protensors-work/vllm-voipmonitor/research/fungible-quant/tools/fruit-encoder
PY=/home/mbelleau/venvs/fq/bin/python
SNAP_O=$(ls -d /home/mbelleau/.cache/huggingface/hub/models--zai-org--GLM-5.2/snapshots/*/ | head -1)
CAPTURE=/home/mbelleau/glm52-capture
STATE=$CAMP/campaign-state.json
LOG=$CAMP/campaign-supervisor.log
TIERS="2 4 5"   # K5 last: serving K5 as a mixed tier is blocked by the
                # SM120 shared-memory limit (see m5-serve/k5-shared-memory-limit.md),
                # so K4 segments are worth more right now.
# MoE layers are 3-77. Layer 78 is the MTP (multi-token prediction)
# layer: it is NOT a main-model MoE layer, capture_stream rejects it
# ("layers must be a nonempty subset of MoE layers"), and the stats
# collector never binds it either. LAST=78 made the final window 75-78,
# which failed every pass and left K2 permanently 3 layers short.
FIRST=3; LAST=77; STEP=8
MIN_FREE_GB=180          # refuse to start a window below this
export HOME=/home/mbelleau
# credentials for the publish step (missing token => silent hang)
[ -f "$HOME/.fq_env" ] && . "$HOME/.fq_env"

log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$LOG"; }

free_gpus() {   # idle = <1 GiB resident (utilization can read 100% spuriously)
  # A GPU listed in .reserved-gpus is off limits even when idle: a serve that
  # is still loading looks idle for minutes, and the campaign would grab the
  # cards out from under it. Operator writes e.g. "0,1,2,3" to that file.
  local res=""
  [ -f "$CAMP/.reserved-gpus" ] && res=$(tr -d ' \n' < "$CAMP/.reserved-gpus")
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits |
    awk -F', *' -v res=",$res," '$2 < 1024 && index(res, ","$1",") == 0 {printf "%s,", $1}' |
    sed 's/,$//'
}

layers_done() { # tier -> count of done-JSONs in [a,b]
  local t=$1 a=$2 b=$3 n=0 L
  for L in $(seq $a $b); do
    [ -f "/home/mbelleau/glm52-work-k$t/layer-$(printf %03d $L).done.json" ] && n=$((n+1))
  done
  echo $n
}

write_state() {
  $PY - "$STATE" "$@" <<'PY' 2>/dev/null || true
import json, sys, time
p, tier, win, phase = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
try:    d = json.load(open(p))
except Exception: d = {"schema": "fq-campaign/1", "history": []}
d["current"] = {"tier": tier, "window": win, "phase": phase,
                "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
d["history"] = (d.get("history", []) + [d["current"]])[-200:]
json.dump(d, open(p, "w"), indent=1)
PY
}

# Single-owner lock: refuse to start if another supervisor OR a stray
# encode from an earlier ring is still running — two encoders writing one
# work dir race on the same layer files.
LOCK=$CAMP/.supervisor.lock
if [ -f "$LOCK" ] && kill -0 "$(cat "$LOCK" 2>/dev/null)" 2>/dev/null; then
  echo "supervisor already running (pid $(cat "$LOCK")) — exiting"; exit 0
fi
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT
FIRST_TIER=$(echo $TIERS | awk '{print $1}')
while pgrep -f "fruit_encode_driver.*--bits $FIRST_TIER " >/dev/null; do
  log "a K$FIRST_TIER encode is already running — waiting to take ownership"
  sleep 120
done

log "supervisor start (tiers: $TIERS, layers $FIRST-$LAST, step $STEP)"
while true; do
  PROGRESS=0
  for T in $TIERS; do
    for START in $(seq $FIRST $STEP $LAST); do
      END=$((START + STEP - 1)); [ $END -gt $LAST ] && END=$LAST
      WANT=$((END - START + 1))
      [ "$(layers_done $T $START $END)" -eq "$WANT" ] && continue

      FREEGB=$(df --output=avail -BG /home | tail -1 | tr -dc 0-9)
      if [ "${FREEGB:-0}" -lt "$MIN_FREE_GB" ]; then
        log "disk ${FREEGB}G < ${MIN_FREE_GB}G — publishing/pruning before more work"
        $PY "$CAMP/publish_window.py" >> "$CAMP/publish-auto.log" 2>&1 && \
          find "$CAPTURE" -maxdepth 1 -name 'layer_*' -mmin +120 -exec rm -rf {} + 2>/dev/null
        # WAIT on this window, do not advance past it. `continue` steps the
        # INNER loop to the NEXT window, so a sustained low-disk period walks
        # silently through the whole work list and resumes at an arbitrary
        # point. Observed: K2 was 72/76 with only window 75-78 left; the guard
        # fired for ~10 minutes, skipped that window and every K4 window, and
        # the campaign came back up encoding K5 -- the tier the operator had
        # explicitly deprioritised.
        sleep 120
        FREEGB=$(df --output=avail -BG /home | tail -1 | tr -dc 0-9)
        [ "${FREEGB:-0}" -lt "$MIN_FREE_GB" ] && continue   # still low: re-check
        log "disk recovered to ${FREEGB}G — resuming window $START-$END"
      fi

      # -- capture (skipped when this window's layers are already present)
      NEED_CAP=0
      for L in $(seq $START $END); do
        [ -d "$CAPTURE/layer_$(printf %03d $L)" ] || NEED_CAP=1
      done
      if [ $NEED_CAP -eq 1 ]; then
        # Stale-state guard. capture_stream resumes from state.json's
        # per-shard packs_done. If a window's captures were PRUNED to reclaim
        # disk, that state still claims the packs were emitted, so the shards
        # skip every layer, exit 0, and the seal dies with
        # "seal: layer N missing x.bin" -- on every retry, forever.
        # A window with NO layer dirs on disk has nothing to resume, so the
        # state can only be stale. Mid-window resume (some dirs present) is
        # preserved, which is the case the state file exists for.
        # A layer dir holding only *.partial files is an INTERRUPTED capture,
        # not a finished one. This box is pre-emptible, so that is the normal
        # crash shape: the shard's state still says the packs were emitted, it
        # skips the layer ("layer 4 already complete; skip"), and the seal
        # dies on the missing x.bin. Delete those dirs so they are recaptured.
        PARTIAL=0
        for L in $(seq $START $END); do
          D="$CAPTURE/layer_$(printf %03d $L)"
          [ -d "$D" ] || continue
          if [ ! -f "$D/x.bin" ] || [ -n "$(ls "$D"/*.partial 2>/dev/null)" ]; then
            log "window $START-$END: layer $L capture is incomplete (partial files) — removing"
            rm -rf "$D"; PARTIAL=1
          fi
        done
        HAVE=0
        for L in $(seq $START $END); do
          [ -d "$CAPTURE/layer_$(printf %03d $L)" ] && HAVE=1
        done
        # Any removal invalidates the shard state that claimed those packs.
        if [ $PARTIAL -eq 1 ] && [ -f "$CAPTURE/state.json" ]; then
          log "window $START-$END: clearing capture state after removing partial layers"
          rm -f "$CAPTURE/state.json"; rm -rf "$CAPTURE/work"
        fi
        if [ $HAVE -eq 0 ] && [ -f "$CAPTURE/state.json" ]; then
          log "window $START-$END: no layer dirs but state.json exists — clearing stale capture state"
          rm -f "$CAPTURE/state.json"
          rm -rf "$CAPTURE/work"
        fi
        write_state "$T" "$START-$END" capture
        log "K$T window $START-$END: capture"
        LAYERS=$START-$END STOP=$END "$CAMP/run-capture-glm52.sh" \
          >> "$CAMP/capture-glm52.log" 2>&1 || { log "capture FAILED $START-$END (retry next pass)"; sleep 60; continue; }
      fi

      # -- encode on whatever is idle right now
      # Only a SAME-TIER encode can collide (same work dir, same done-JSONs).
      # A different tier running on other GPUs is fine — and is how the idle
      # GPUs during another tier's run get used at all.
      if pgrep -f "fruit_encode_driver.*--bits $T " >/dev/null; then
        log "K$T already encoding — waiting"; sleep 120; continue
      fi
      G=$(free_gpus); [ -z "$G" ] && { log "no idle GPU — waiting"; sleep 180; continue; }
      W=$(echo "$G" | tr ',' '\n' | wc -l)
      write_state "$T" "$START-$END" "encode(gpus=$G)"
      log "K$T window $START-$END: encode on GPUs $G ($W workers)"
      "$RUNS/gg-env/gg-run.sh" python "$TOOLS/fruit_encode_driver.py" \
        --encode --bits "$T" --workers "$W" --gpus "$G" \
        --src "$SNAP_O" --work "/home/mbelleau/glm52-work-k$T" \
        --capture-dir "$CAPTURE" --layers "$START-$END" \
        >> "$CAMP/glm52-encode-k$T.log" 2>&1
      log "K$T window $START-$END: encode exit $? ($(layers_done $T $START $END)/$WANT layers)"

      # -- reclaim BEFORE publishing, not only after.
      # The disk guard runs once per window, at the top. With 8 GPUs an
      # encode now finishes in half the time and produces artifacts twice as
      # fast, so a single window can consume more headroom than existed when
      # it started -- observed: 46G free, 134G under the floor, with no guard
      # able to fire because none of them run mid-window. This window's own
      # captures are dead the moment its encode exits 0, so drop them here
      # rather than waiting for the post-publish prune.
      FREEGB=$(df --output=avail -BG /home | tail -1 | tr -dc 0-9)
      if [ "${FREEGB:-0}" -lt "$MIN_FREE_GB" ]; then
        log "post-encode disk ${FREEGB}G < ${MIN_FREE_GB}G — reclaiming this window's captures early"
        DONE_ALL=1
        for L in $(seq $START $END); do
          [ -f "/home/mbelleau/glm52-work-k$T/layer-$(printf %03d $L).done.json" ] || DONE_ALL=0
        done
        if [ $DONE_ALL -eq 1 ]; then
          for L in $(seq $START $END); do rm -rf "$CAPTURE/layer_$(printf %03d $L)"; done
          log "reclaimed to $(df --output=avail -BG /home | tail -1 | tr -dc 0-9)G"
        else
          log "NOT reclaiming: window $START-$END is not fully encoded"
        fi
      fi

      # -- publish + prune
      write_state "$T" "$START-$END" publish
      if $PY "$CAMP/publish_window.py" >> "$CAMP/publish-auto.log" 2>&1; then
        log "K$T window $START-$END: published"
        # Prune this window's capture as soon as its segments are published.
        # A later tier re-captures it; capture is deterministic (validated
        # bit-exact) so the re-captured activations are byte-identical and
        # tiers stay hessian-identical. Holding captures for the whole K2
        # pass instead filled the disk (observed: 455 -> 213 GB).
        # Defer only while an encoder is reading THIS window. A blanket
        # "any encoder" check never clears once tiers overlap continuously,
        # and disk creeps to the floor (observed: 190 GB with two windows
        # pinned). Match the window's own --layers argument.
        if pgrep -af "fruit_encode_driver.*--layers $START-$END" >/dev/null; then
          log "window $START-$END: prune deferred — an encode is still reading it"
        else
          for L in $(seq $START $END); do rm -rf "$CAPTURE/layer_$(printf %03d $L)"; done
          log "window $START-$END: capture pruned after publish"
        fi
      else
        log "publish failed (retry next pass)"
      fi
      PROGRESS=1
    done
    # a tier is only left behind once every window of it is complete
  done
  if [ $PROGRESS -eq 0 ]; then
    write_state all all idle
    log "all tiers complete for layers $FIRST-$LAST — idling (rechecking hourly)"
    sleep 3600
  fi
done
