#!/bin/bash
# reap-devices.sh <gpu-list> — free specific GPUs, by DEVICE, never by pattern.
#
# Two failures, one root cause: matching against a vLLM worker's command line.
#
#   pkill -f vllm      matched NOTHING. The workers exec through the rootfs
#                      ld-linux shim, so their argv contains neither "vllm"
#                      nor "VLLM::", and four orphans sat holding 22 GiB each
#                      while a fresh boot tried to start on top of them.
#
#   pgrep -f 8200      matched TOO MUCH. It killed a healthy serve on port
#                      8100, because --hf-overrides carries 150+ per-layer
#                      sha256 digests and one of them
#                      (75469d18...e07d1e9b8200...) contains "8200". A 20 KB
#                      argv will contain almost any short string you can think
#                      of.
#
# The device is the only identity that means anything here. nvidia-smi knows
# exactly which processes hold which GPU; ask it.
#
# Usage:  reap-devices.sh 0,1,2,3
#         reap-devices.sh 4,5,6,7 --dry-run
set -u
DEVICES=${1:?usage: reap-devices.sh <gpu-list e.g. 0,1,2,3> [--dry-run]}
DRY=${2:-}

pids_on() {
  nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$DEVICES" \
    2>/dev/null | tr -d ' ' | grep -E '^[0-9]+$' | sort -u
}
used_on() {
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
    -i "$DEVICES" 2>/dev/null | awk -F', *' '{printf "%s:%sMiB ", $1, $2}'
}

echo "GPUs $DEVICES before: $(used_on)"
P=$(pids_on)
if [ -z "$P" ]; then
  echo "no compute processes on $DEVICES — nothing to reap"
  exit 0
fi
echo "compute processes on $DEVICES: $(echo "$P" | tr '\n' ' ')"
if [ "$DRY" = "--dry-run" ]; then
  echo "(dry run: not killing)"
  exit 0
fi

for p in $P; do kill -9 "$p" 2>/dev/null && echo "  killed $p"; done

# Driver memory release lags the kill, consistently by 10-20s on this box, and
# a boot started too early sees the cards as occupied and reaps ITSELF. Wait
# for the memory to actually come back rather than for the processes to go.
for i in $(seq 1 8); do
  sleep 5
  left=$(pids_on)
  free=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
         -i "$DEVICES" 2>/dev/null | awk '$1 > 1024' | wc -l)
  [ -z "$left" ] && [ "$free" -eq 0 ] && break
  # A second pass: a worker can outlive the first signal.
  for p in $left; do kill -9 "$p" 2>/dev/null; done
done

echo "GPUs $DEVICES after:  $(used_on)"
left=$(pids_on)
if [ -n "$left" ]; then
  echo "WARNING: still holding $DEVICES: $(echo "$left" | tr '\n' ' ')" >&2
  exit 1
fi
