#!/bin/bash
# Pull both GG container images to OCI layouts under /home (persistent).
# Resumable: cached blobs are sha256-verified and skipped on retry.
set -u
source ~/.fq_env
JOB_DIR="$(cd "$(dirname "$0")" && pwd)"
DEST=/home/mbelleau/images
PY=/home/mbelleau/venvs/fq/bin/python

pull() { # pkg tag dest
  local attempt=0
  while true; do
    attempt=$((attempt+1))
    "$PY" "$JOB_DIR/ghcr_pull.py" "$1" "$2" "$3" && return 0
    echo "[$(date -u +%FT%TZ)] $1:$2 attempt $attempt failed; retry in 20s" >&2
    sleep 20
  done
}

pull malaiwah/gilded-gnosis-v20 r12-field-review "$DEST/gilded-gnosis-v20-r12-field-review"
pull malaiwah/glm52-exl3-vast latest "$DEST/glm52-exl3-vast-latest"
echo "[$(date -u +%FT%TZ)] all images pulled" | tee "$JOB_DIR/done.marker"
