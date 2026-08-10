#!/bin/bash
# bench-leg.sh <tag> — one A/B leg: warmup + 3 measured traffic runs +
# per-rank memory snapshot via the fq_reload worker-extension RPC.
# Traffic is deterministic (fixed seed/prompt set) so legs are comparable.
set -eu
RUN=/home/mbelleau/protensors-work/vllm-voipmonitor/research/fungible-quant/runs/m2-dryrun
TAG=$1
python3 "$RUN/traffic_bench.py" --n 40 --concurrency 4 --max-tokens 96 \
  --seed 9 --tag "$TAG-warmup" --out "$RUN/bench-$TAG-warmup.json"
for i in 1 2 3; do
  python3 "$RUN/traffic_bench.py" --n 160 --concurrency 4 --max-tokens 96 \
    --seed 0 --tag "$TAG-r$i" --out "$RUN/bench-$TAG-r$i.json"
done
curl -s -X POST http://127.0.0.1:8801/collective_rpc \
  -H 'Content-Type: application/json' \
  -d '{"method":"fq_expert_state"}' > "$RUN/mem-$TAG.json"
echo "leg $TAG done"
