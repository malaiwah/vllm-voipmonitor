#!/bin/bash
# One-glance health sweep. Reports process-liveness + CR-cleaned fresh tails,
# and — more importantly — whether each job's log is STILL GROWING. A stalled
# job and a healthy one look identical if you only print the last line.
#
# Rewritten 2026-08-11: the previous version still watched the window-1 ring,
# its publisher, and a :8801 Fruit serve, all of which finished hours earlier.
# It reported alive=0 for jobs that were *supposed* to be gone while saying
# nothing about the four that were actually running, which is worse than no
# sweep at all — it reads as "pipeline idle" when the box is fully busy.
RUNS=/home/mbelleau/protensors-work/vllm-voipmonitor/research/fungible-quant/runs
STAMP=/tmp/claude-1000/fq-sweep-sizes

mkdir -p "$(dirname "$STAMP")"; touch "$STAMP"

grew() { # logfile -> "+N lines" | "STALLED" | "-"
  local f=$1 key now prev
  [ -f "$f" ] || { echo "no-log"; return; }
  key=$(echo "$f" | md5sum | cut -c1-8)
  now=$(wc -c < "$f" 2>/dev/null || echo 0)
  prev=$(grep "^$key " "$STAMP" 2>/dev/null | tail -1 | awk '{print $2}')
  grep -v "^$key " "$STAMP" > "$STAMP.tmp" 2>/dev/null; mv "$STAMP.tmp" "$STAMP"
  echo "$key $now" >> "$STAMP"
  if [ -z "$prev" ]; then echo "first-look"
  elif [ "$now" -gt "$prev" ]; then echo "+$((now - prev))B"
  else echo "STALLED"; fi
}

report() { # label pattern logfile
  local alive tail delta
  # pgrep -fc prints 0 AND exits 1 on no-match, so a `|| echo 0` fallback
  # renders "0\n0" and wrecks the column alignment.
  alive=$(pgrep -fc "$2" 2>/dev/null | head -1); alive=${alive:-0}
  delta=$(grew "$3")
  # Label honestly, because a sweep that cries wolf gets ignored:
  #   alive + growing -> +NB      (working)
  #   alive + flat    -> quiet    (e.g. supervisor logs only every 120s)
  #   dead  + grew    -> exited   (finished since the last sweep)
  #   dead  + flat    -> idle     (not running; NOT the same as stuck)
  # Nothing here can tell "idle" from "should be running but isn't" — that
  # needs the tier-coverage lines below, which show whether work remains.
  if [ "$delta" = STALLED ]; then
    [ "$alive" -gt 0 ] && delta="quiet" || delta="idle"
  elif [ "$alive" -eq 0 ] && [ "${delta#+}" != "$delta" ]; then
    delta="exited"
  fi
  tail=""; [ -f "$3" ] && tail=$(tr '\r' '\n' < "$3" | grep -vE '^$' | tail -1 | cut -c1-62)
  printf "%-13s alive=%-3s %-11s | %s\n" "$1" "$alive" "$delta" "$tail"
}

echo "=== fungible-quant sweep $(date -u +%FT%TZ)"
report supervisor "campaign-supervisor[.]sh"    "$RUNS/0c-campaign/campaign-supervisor.log"
report encode-k2  "fruit_encode_driver.*bits 2" "$RUNS/0c-campaign/glm52-encode-k2.log"
report encode-k5  "fruit_encode_driver.*bits 5" "$RUNS/0c-campaign/glm52-encode-k5.log"
report capture    "capture_stream[.]py"         "$RUNS/0c-campaign/capture-glm52.log"
report publish    "publish_window[.]py"         "$RUNS/0c-campaign/publish-auto.log"
report m5-serve   "vllm.*api_server"            "$RUNS/m5-serve/results/live/serve.log"

# Tier coverage is the actual deliverable; process liveness is only a proxy.
for K in 2 4 5; do
  n=$(ls /home/mbelleau/glm52-work-k$K/*.done.json 2>/dev/null | wc -l)
  [ "$n" -gt 0 ] && printf "  K%-2s layers encoded: %s/76\n" "$K" "$n"
done

# M5 evidence-run artifacts, when that campaign is active.
for d in "$RUNS"/m5-serve/results/*/; do
  [ -d "$d" ] || continue
  tl=$(cat "$d"/timeline-*.jsonl 2>/dev/null | wc -l)
  printf "  m5 %-12s timeline rows: %s\n" "$(basename "$d")" "$tl"
done

if curl -fsS -m 3 http://127.0.0.1:8000/health >/dev/null 2>&1; then
  echo "  m5-serve :8000 HEALTHY"
else
  echo "  m5-serve :8000 down"
fi

echo "--- GPUs:"; nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader | paste -sd' ' -
FREE=$(df --output=avail -BG /home | tail -1 | tr -dc 0-9)
printf -- "--- disk: %sG free%s\n" "$FREE" \
  "$([ "$FREE" -lt 180 ] && echo '  *** BELOW CAMPAIGN FLOOR (180G) ***')"
