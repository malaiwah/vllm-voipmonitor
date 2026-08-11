#!/bin/bash
# pause-when-k4-done.sh — let the in-flight K4 window finish, then stop the
# campaign cleanly and report what can be reclaimed.
#
# Not a kill: killing mid-encode wastes the window and leaves partial captures
# that the next start has to detect and redo. This waits for the artifact
# (75 done-JSONs), then stops the supervisor so it cannot claim a new window,
# and leaves every encoder output in place for the publisher.
set -u
CAMP=/home/mbelleau/protensors-work/vllm-voipmonitor/research/fungible-quant/runs/0c-campaign
log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$CAMP/pause.log"; }

log "waiting for K4 to reach 75/75 (currently $(ls /home/mbelleau/glm52-work-k4/*.done.json 2>/dev/null | wc -l))"
while [ "$(ls /home/mbelleau/glm52-work-k4/*.done.json 2>/dev/null | wc -l)" -lt 75 ]; do
  sleep 60
done
log "K4 complete: 75/75"

# Let a running encode finish writing; only then remove the supervisor.
while pgrep -f "fruit_encode_driver" >/dev/null; do
  log "encode still finishing — waiting"
  sleep 60
done

SPID=$(cat "$CAMP/.supervisor.lock" 2>/dev/null)
if [ -n "${SPID:-}" ] && kill -0 "$SPID" 2>/dev/null; then
  kill "$SPID" && log "supervisor stopped (pid $SPID)"
fi
rm -f "$CAMP/.supervisor.lock"
for p in $(pgrep -f "capture_stream"); do kill "$p" 2>/dev/null && log "capture $p stopped"; done

# Publish anything not yet on HF, so nothing local is load-bearing.
log "final publish pass"
/home/mbelleau/venvs/fq/bin/python "$CAMP/publish_window.py" >> "$CAMP/publish-auto.log" 2>&1 \
  && log "publish ok" || log "publish returned non-zero — check publish-auto.log"

log "PAUSED. Reclaimable once verified on HF:"
du -sh /home/mbelleau/glm52-capture /home/mbelleau/glm52-work-k2 /home/mbelleau/glm52-work-k4 \
       /home/mbelleau/glm52-work-k5 /home/mbelleau/glm52-segments 2>/dev/null | tee -a "$CAMP/pause.log"
df -h /home | awk 'NR==2{print "  disk now: "$4" free"}' | tee -a "$CAMP/pause.log"
