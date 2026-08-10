#!/bin/bash
# One-glance health sweep of all FQ jobs. Used by the recurring check.
RUNS=/home/mbelleau/protensors-work/vllm-voipmonitor/research/fungible-quant/runs
for f in serve-baseline/serve.log t1-graph-freeze/t1.log 0c-campaign/capture.log \
         m0-assemble/verify.log m0-seed/upload.log; do
  p=$RUNS/$f
  [ -f "$p" ] || continue
  printf "%-28s %8sB  mtime %s  | %s\n" "$(basename $(dirname $f))" \
    "$(stat -c%s $p)" "$(date -d @$(stat -c%Y $p) +%H:%M:%S)" \
    "$(tail -c 2000 $p | tr '\r' '\n' | grep -vE '^$' | tail -1 | cut -c1-70)"
done
echo "--- GPUs:"; nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader | paste -sd' ' -
echo "--- disk: $(df -h /home | awk 'NR==2{print $4" free"}')  shm: $(df -h /dev/shm | awk 'NR==2{print $3" used"}')"
