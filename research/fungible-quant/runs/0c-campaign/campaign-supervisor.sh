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
TIERS="2 5 4"
FIRST=3; LAST=78; STEP=8
MIN_FREE_GB=180          # refuse to start a window below this
export HOME=/home/mbelleau

log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$LOG"; }

free_gpus() {   # idle = <1 GiB resident (utilization can read 100% spuriously)
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits |
    awk -F', *' '$2 < 1024 {printf "%s,", $1}' | sed 's/,$//'
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
        sleep 120; continue
      fi

      # -- capture (skipped when this window's layers are already present)
      NEED_CAP=0
      for L in $(seq $START $END); do
        [ -d "$CAPTURE/layer_$(printf %03d $L)" ] || NEED_CAP=1
      done
      if [ $NEED_CAP -eq 1 ]; then
        write_state "$T" "$START-$END" capture
        log "K$T window $START-$END: capture"
        LAYERS=$START-$END STOP=$END "$CAMP/run-capture-glm52.sh" \
          >> "$CAMP/capture-glm52.log" 2>&1 || { log "capture FAILED $START-$END (retry next pass)"; sleep 60; continue; }
      fi

      # -- encode on whatever is idle right now
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

      # -- publish + prune
      write_state "$T" "$START-$END" publish
      if $PY "$CAMP/publish_window.py" >> "$CAMP/publish-auto.log" 2>&1; then
        log "K$T window $START-$END: published"
        if [ "$T" = "$(echo $TIERS | awk '{print $NF}')" ]; then
          for L in $(seq $START $END); do rm -rf "$CAPTURE/layer_$(printf %03d $L)"; done
          log "window $START-$END: capture pruned (last tier done)"
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
