#!/bin/bash
# preflight.sh — cheap checks to run the moment the serve answers, before
# spending an hour of GPU time on a benchmark that was pointed at the wrong
# model name or a server that streams without usage.
#
# Usage: preflight.sh [endpoint] [model]
set -uo pipefail
ENDPOINT=${1:-http://127.0.0.1:8000}
MODEL=${2:-GLM-5.2}
rc=0

say() { printf '%-34s %s\n' "$1" "$2"; }

ids=$(curl -sf --max-time 10 "${ENDPOINT%/}/v1/models" \
      | /home/mbelleau/venvs/bench/bin/python -c \
        'import json,sys;print(" ".join(m["id"] for m in json.load(sys.stdin)["data"]))' \
      2>/dev/null) || { say "/v1/models" "UNREACHABLE"; exit 1; }
say "/v1/models" "$ids"
case " $ids " in
  *" $MODEL "*) say "model name '$MODEL'" "OK" ;;
  *) say "model name '$MODEL'" "NOT SERVED -- fix MODEL= or --served-model-name"; rc=1 ;;
esac

if curl -sf --max-time 10 "${ENDPOINT%/}/metrics" >/dev/null 2>&1; then
  say "/metrics" "exposed (server-side cross-check available)"
else
  say "/metrics" "absent -- client-side numbers only, and swap_evidence.py scrape will not work"
  rc=1
fi

# The one property every throughput number here depends on: does a streamed
# response carry a cumulative completion_tokens on each chunk? Without it the
# tools fall back to counting chunks, which is an estimate, not a token count.
usage=$(curl -sf --max-time 120 "${ENDPOINT%/}/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Count to five.\"}],\"max_tokens\":24,\"stream\":true,\"stream_options\":{\"include_usage\":true,\"continuous_usage_stats\":true}}" \
  | grep -c '"completion_tokens"') || usage=0
if [ "${usage:-0}" -gt 2 ]; then
  say "continuous usage stats" "OK ($usage chunks carried usage)"
else
  say "continuous usage stats" "MISSING -- decode tok/s will be a chunk-count ESTIMATE"
  rc=1
fi

exit $rc
