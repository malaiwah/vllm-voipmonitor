#!/bin/bash
# Pull the canonical v20-r33 baseline image via skopeo to an OCI layout.
set -u
IMG=docker.io/voipmonitor/vllm:gilded-gnosis-v20-vllmfa13d33-b12x06db0f4-fi1ac6942-cu132-20260809-r33
DEST=/home/mbelleau/images/gg-v20-r33
attempt=0
while true; do
  attempt=$((attempt+1))
  /home/mbelleau/tools/skopeo-env/bin/skopeo copy --retry-times 3 \
    "docker://$IMG" "oci:$DEST:r33" && break
  echo "[$(date -u +%FT%TZ)] attempt $attempt failed; retry in 30s" >&2
  sleep 30
done
echo "[$(date -u +%FT%TZ)] r33 image pulled to $DEST"
