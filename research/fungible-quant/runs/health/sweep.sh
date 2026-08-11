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
# Newest serve log across all result dirs — the tag varies per run
# (baseline-k3, live, scenario1...), and hardcoding one made a running serve
# report "no-log" while it was mid-boot.
# serve-attempt*.log, NOT serve.log: the attempts are where the live boots
# write, and globbing only serve.log silently pinned this to a dead run --
# reporting "quiet" for hours while a healthy boot streamed beside it.
M5LOG=$(ls -t "$RUNS"/m5-serve/results/*/serve*.log 2>/dev/null | head -1)

# TWO instances now run concurrently (GPUs 0-3 and 4-7). Reporting only the
# newest log makes the sweep blind to whichever stack was started first --
# which is exactly the one that has been running long enough to be
# interesting. Report each instance separately, by results dir.
report_instance() {
  local tag=$1 port=$2
  local log
  log=$(ls -t "$RUNS"/m5-serve/results/$tag/serve*.log 2>/dev/null | head -1)
  [ -n "$log" ] || return 0
  local ly dl up
  ly=$(grep "Worker_TP0" "$log" 2>/dev/null | grep -oE "FQ progressive layer [0-9]+" | sort -u | wc -l)
  dl=$(grep -E "FQ downloads" "$log" 2>/dev/null | tail -1 | sed 's/^.*\] //')
  if [ -z "$dl" ]; then
    local c l
    c=$(grep -c "FQ progressive L[0-9]*: cached" "$log" 2>/dev/null)
    l=$(grep -c "no fetch" "$log" 2>/dev/null)
    dl="warm: ${c} cached, ${l} local (0 fetched)"
  fi
  up=down
  curl -fsS -m 3 "http://127.0.0.1:$port/health" >/dev/null 2>&1 && up=HEALTHY
  # The two lines that say whether the LIVE APPLY path is doing anything.
  local bound applied
  # Both are logged once PER RANK, so a raw line count reports 4 for a single
  # event — the same fourfold inflation the layer counter had. Count distinct
  # timestamps instead: one install is one moment, however many ranks say so.
  bound=$(grep -c "live apply BOUND" "$log" 2>/dev/null)
  applied=$(grep "swap(s) INSTALLED" "$log" 2>/dev/null \
            | grep -oE "[0-9]{2}:[0-9]{2}:[0-9]{2}" | sort -u | wc -l)
  swapped=$(grep "swap(s) INSTALLED" "$log" 2>/dev/null | tail -1 \
            | grep -oE "[0-9]+ swap" | head -1 | grep -oE "[0-9]+")
  echo "  [$tag :$port $up] ${ly}/76 layers | $dl"
  echo "      apply bound on ${bound} rank(s), ${applied} install(s), last ${swapped:-0} swaps"
}
report m5-serve   "vllm.*api_server"            "${M5LOG:-/nonexistent}"

# Tier coverage is the actual deliverable; process liveness is only a proxy.
for K in 2 4 5; do
  n=$(ls /home/mbelleau/glm52-work-k$K/*.done.json 2>/dev/null | wc -l)
  [ "$n" -gt 0 ] && printf "  K%-2s layers encoded: %s/75\n" "$K" "$n"
done

# M5 evidence-run artifacts, when that campaign is active.
for d in "$RUNS"/m5-serve/results/*/; do
  [ -d "$d" ] || continue
  tl=$(cat "$d"/timeline-*.jsonl 2>/dev/null | wc -l)
  printf "  m5 %-12s timeline rows: %s\n" "$(basename "$d")" "$tl"
done

# Probe every port a serve has actually used. Hardcoding 8000 reported
# "down" for a server healthy on 8100.
M5PORT=$(grep -oE '\-\-port [0-9]+' "${M5LOG:-/dev/null}" 2>/dev/null | tail -1 | awk '{print $2}')
M5UP=""
for _p in ${M5PORT:-} 8100 8000; do
  if curl -fsS -m 3 "http://127.0.0.1:$_p/health" >/dev/null 2>&1; then M5UP=$_p; break; fi
done
# Progress signal for a boot still loading: bytes delivered, not liveness.
if [ -n "${M5LOG:-}" ]; then
  _dl=$(grep -E "FQ downloads" "$M5LOG" 2>/dev/null | tail -1 | sed 's/^.*\] //')
  # A WARM boot emits no "FQ downloads" line at all — nothing is downloaded.
  # Without a substitute the progress signal vanished exactly when restarts
  # became fast enough to be the normal case, leaving a loading serve
  # indistinguishable from a stuck one.
  if [ -z "$_dl" ]; then
    _cached=$(grep -c "FQ progressive L[0-9]*: cached" "$M5LOG" 2>/dev/null)
    _local=$(grep -c "no fetch" "$M5LOG" 2>/dev/null)
    [ "${_cached:-0}" -gt 0 ] || [ "${_local:-0}" -gt 0 ] && \
      _dl="warm boot: ${_cached} segments from cache, ${_local} local (0 fetched)"
  fi
  # DISTINCT layers on ONE rank. Counting matched lines sums across all four
  # TP ranks and reports "248/76", which reads as either nonsense or done.
  _ly=$(grep "Worker_TP0" "$M5LOG" 2>/dev/null | grep -oE "FQ progressive layer [0-9]+" | sort -u | wc -l)
  [ -n "$_dl" ] && echo "  m5-serve load: ${_ly}/76 layers | $_dl"
fi
report_instance demo1 8100
report_instance demo2 8200
if [ -n "$M5UP" ]; then
  echo "  m5-serve :$M5UP HEALTHY"
else
  echo "  m5-serve :${M5PORT:-8100} down"
fi

echo "--- GPUs:"; nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader | paste -sd' ' -
FREE=$(df --output=avail -BG /home | tail -1 | tr -dc 0-9)
printf -- "--- disk: %sG free%s\n" "$FREE" \
  "$([ "$FREE" -lt 180 ] && echo '  *** BELOW CAMPAIGN FLOOR (180G) ***')"
