#!/bin/bash
# Resumable download of zai-org/GLM-5.2 (1.51 TB, 282 shards), pinned revision.
# Crash/reboot-safe: hf download resumes from the HF cache; loop retries forever.
set -u
source ~/.fq_env
export PATH="$HOME/.local/bin:$PATH"
export HF_HUB_ENABLE_HF_TRANSFER=1
VENV=/home/mbelleau/venvs/fq
REPO=zai-org/GLM-5.2
REV=b4734de4facf877f85769a911abafc5283eab3d9
JOB_DIR="$(cd "$(dirname "$0")" && pwd)"
STATE="$JOB_DIR/state.json"

write_state() {
  printf '{"job":"dl-glm52-orig","repo":"%s","revision":"%s","status":"%s","attempt":%d,"updated_utc":"%s"}\n' \
    "$REPO" "$REV" "$1" "$2" "$(date -u +%FT%TZ)" > "$STATE.tmp" && mv "$STATE.tmp" "$STATE"
}

attempt=0
while true; do
  attempt=$((attempt+1))
  write_state running "$attempt"
  "$VENV/bin/hf" download "$REPO" --revision "$REV" && break
  echo "[$(date -u +%FT%TZ)] attempt $attempt failed; retrying in 30s" >&2
  write_state retrying "$attempt"
  sleep 30
done
write_state done "$attempt"
echo "[$(date -u +%FT%TZ)] download complete"
