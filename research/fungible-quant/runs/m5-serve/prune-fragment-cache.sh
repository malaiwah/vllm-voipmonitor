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
#
# TIERED ON PURPOSE: this evicts fragments/ ONLY, never segments/.
#   segments/  whole-layer objects — the warm-restart asset. A hot restart
#              slices these locally instead of re-fetching ~230 GiB, so
#              evicting them turns a warm boot into a cold one wearing a warm
#              label. Pinned.
#   fragments/ per-expert payloads derived from those segments. Speculative
#              and reconstructible. Evictable.
# With VLLM_FQ_KEEP_LAYERS=1 the segments tier supersedes fragments entirely,
# so pruning fragments hard costs nothing.
#
# The floor was 200 G, set while the encode campaign was competing for disk.
# The campaign is paused and 136 GB was reclaimed, so a floor that high now
# fires against a cache we deliberately want large -- it would thrash against
# its own boot. Lowered to leave the system real headroom without punishing
# the thing under test.
set -eu
CACHE=${VLLM_FQ_CACHE:-/home/mbelleau/cache/fq-demo1}/fragments
FLOOR_GB=${FQ_DISK_FLOOR_GB:-90}
KEEP_MIN=${FQ_FRAGMENT_KEEP_MIN:-10}

free_gb() { df -BG --output=avail /home | tail -1 | tr -dc '0-9'; }

while :; do
  avail=$(free_gb)
  if [ "$avail" -lt "$FLOOR_GB" ]; then
    before=$(du -sm "$CACHE" 2>/dev/null | cut -f1 || echo 0)
    # ABSOLUTE cutoff, not "-N minutes": bfs (a drop-in find replacement
    # present on this box) rejects relative -newermt outright. Combined with
    # the `|| true` below, that turns the prune into a silent no-op -- the
    # loop keeps running, the log keeps printing, and the disk keeps filling.
    # An absolute ISO timestamp is accepted by both GNU find and bfs.
    cutoff=$(date -u -d "${KEEP_MIN} minutes ago" +%Y-%m-%dT%H:%M:%S)
    find "$CACHE" -type f ! -newermt "$cutoff" -delete 2>/dev/null || true
    find "$CACHE" -type d -empty -delete 2>/dev/null || true
    after=$(du -sm "$CACHE" 2>/dev/null | cut -f1 || echo 0)
    echo "$(date -u +%FT%TZ) disk ${avail}G < ${FLOOR_GB}G — pruned fragments ${before}MB -> ${after}MB"
    # A prune that frees nothing while under the floor is a broken prune, not
    # a clean cache: say so instead of looping quietly until the disk fills.
    if [ "$before" -gt 1024 ] && [ "$after" -ge "$before" ]; then
      echo "$(date -u +%FT%TZ) WARNING: under floor but freed 0 MB — prune is not working (find/cutoff?)"
    fi
  fi
  sleep 120
done
