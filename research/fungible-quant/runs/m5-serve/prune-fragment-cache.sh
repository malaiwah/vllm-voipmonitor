#!/bin/bash
# prune-fragment-cache.sh — keep the per-expert fragment cache bounded.
#
# The cache has no eviction: a progressive boot writes EVERY expert through
# it (~14 MiB x 256 = ~3.5 GiB/layer, ~264 GiB for GLM-5.2). The loader fix
# stops re-caching slices of prefetched segments, but a boot started before
# that fix still fills the disk, and the campaign floor is 180 G.
#
# Safe to run against a live boot: each expert is resolved once and its
# payload is already on the GPU. A later miss re-fetches.
set -eu
CACHE=${VLLM_FQ_CACHE:-/home/mbelleau/cache/fq-demo1}/fragments
FLOOR_GB=${FQ_DISK_FLOOR_GB:-200}
KEEP_MIN=${FQ_FRAGMENT_KEEP_MIN:-10}

free_gb() { df -BG --output=avail /home | tail -1 | tr -dc '0-9'; }

while :; do
  avail=$(free_gb)
  if [ "$avail" -lt "$FLOOR_GB" ]; then
    before=$(du -sm "$CACHE" 2>/dev/null | cut -f1 || echo 0)
    find "$CACHE" -type f ! -newermt "-${KEEP_MIN} minutes" -delete 2>/dev/null || true
    find "$CACHE" -type d -empty -delete 2>/dev/null || true
    after=$(du -sm "$CACHE" 2>/dev/null | cut -f1 || echo 0)
    echo "$(date -u +%FT%TZ) disk ${avail}G < ${FLOOR_GB}G — pruned fragments ${before}MB -> ${after}MB"
  fi
  sleep 120
done
