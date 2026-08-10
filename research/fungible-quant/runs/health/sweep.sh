#!/bin/bash
# One-glance health sweep. Reports process-liveness + CR-cleaned fresh tails.
RUNS=/home/mbelleau/protensors-work/vllm-voipmonitor/research/fungible-quant/runs
report() { # name pattern logfile
  local alive; alive=$(pgrep -fc "$2" 2>/dev/null || echo 0)
  local tail=""; [ -f "$3" ] && tail=$(tr '\r' '\n' < "$3" | grep -vE '^$' | tail -1 | cut -c1-70)
  printf "%-14s alive=%s  | %s\n" "$1" "$alive" "$tail"
}
report ring        "ring-window1[.]sh|fruit_encode_driver.*glm52" "$RUNS/0c-campaign/ring-window1.log"
report publish     "publish-window1[.]sh|upload_large_folder"     "$RUNS/0c-campaign/publish-window1.log"
report mixed-serve "vllm.*fruit-mixed"                            "$RUNS/serve-baseline/fruit-mixed.log"
curl -fsS -m 3 http://127.0.0.1:8801/health >/dev/null 2>&1 && echo "mixed-serve  :8801 healthy" || echo "mixed-serve  :8801 DOWN"
echo "--- GPUs:"; nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader | paste -sd' ' -
echo "--- disk: $(df -h /home | awk 'NR==2{print $4" free"}')"
