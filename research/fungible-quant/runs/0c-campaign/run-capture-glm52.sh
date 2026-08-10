#!/bin/bash
# run-capture-glm52.sh — GLM-5.2 layer-streaming BF16 capture, one 8-layer window.
#
# Window: MoE layers 3-10 (streaming-ring campaign; fixed 3TB disk).  Shard 0 on
# GPU 6, shard 1 on GPU 7 (corpus halves, no cross-sample attention), then seal.
# The post-layer-10 boundary (boundary_011 shard files) and layer 10's DSA topk
# store are PRESERVED as the next window's (11-18) input; see
# capture_run_manifest.json -> layer_stream for the resume contract.
#
# Idempotent: rerunning skips sealed boundaries/layers and resumes mid-window.
# Next windows: bump LAYERS/STOP (11-18/18, 19-26/26, ...) after the campaign
# ring encodes+publishes and deletes this window's layer_* payloads.
set -u

GG=/home/mbelleau/protensors-work/vllm-voipmonitor/research/fungible-quant/runs/gg-env/gg-run.sh
CS=/home/mbelleau/protensors-work/vllm-voipmonitor/research/fungible-quant/tools/fruit-encoder/capture_stream.py
SRC=/home/mbelleau/.cache/huggingface/hub/models--zai-org--GLM-5.2/snapshots/b4734de4facf877f85769a911abafc5283eab3d9
CORPUS=/home/mbelleau/.cache/huggingface/hub/models--brandonmusic--GLM-5.2-EXL3-TR3-3.0bpw/snapshots/9297b9f1d53af5c67cffa01e30cc071a1ff7144b/calibration_encoder/calibration/reap_recall_calib.jsonl
PLAN=/home/mbelleau/fq-0c/capture_plan_glm52.json
ROOT=/home/mbelleau/glm52-capture
WORK=$ROOT/work
LAYERS=${LAYERS:-3-10}
STOP=${STOP:-10}
# EXACT MODE (mandatory): one sample per forward, one sample per MoE call.
# This reproduces the reference batch=1 shapes bit-for-bit; any batching change
# (packing or grouped MoE) is NOT row-stable on this GPU (cublas/grouped_mm pick
# shape-dependent reduction orders) and fails the 99.9% routing-id gate.
PACK=1
GROUP=1
# per-sample expert-weight read ~= 2000 token-equivalents; balances GPU 6/7 shards
COST_OVERHEAD=2000

mkdir -p "$ROOT" "$WORK"

free_gb=$(df --output=avail -BG /home/mbelleau | tail -1 | tr -dc 0-9)
if [ "$free_gb" -lt 200 ]; then
    echo "DISK GUARD: only ${free_gb}G free on /home; need 200G for an 8-layer window" >&2
    exit 1
fi
echo "$(date '+%F %T') | window layers=$LAYERS stop=$STOP pack=$PACK group=$GROUP free=${free_gb}G"

run_shard() {
    local idx=$1 gpu=$2
    CUDA_VISIBLE_DEVICES=$gpu $GG python $CS --run \
        --src "$SRC" --corpus "$CORPUS" --plan-file "$PLAN" \
        --work-dir "$WORK" --capture-dir "$ROOT" \
        --layers "$LAYERS" --stop-after-layer "$STOP" \
        --pack-tokens $PACK --moe-group $GROUP --shard-cost-overhead $COST_OVERHEAD \
        --shard-index "$idx" --shard-count 2 \
        --state-file "$ROOT/state.json" \
        --log "$ROOT/shard${idx}.log"
}

run_shard 0 6 &
PID0=$!
run_shard 1 7 &
PID1=$!
wait $PID0; RC0=$?
wait $PID1; RC1=$?
echo "$(date '+%F %T') | shard exits: shard0=$RC0 shard1=$RC1"
if [ $RC0 -ne 0 ] || [ $RC1 -ne 0 ]; then
    echo "SHARD FAILURE — not sealing; rerun this script to resume" >&2
    exit 1
fi

CUDA_VISIBLE_DEVICES=6 $GG python $CS --seal \
    --src "$SRC" --corpus "$CORPUS" --plan-file "$PLAN" \
    --work-dir "$WORK" --capture-dir "$ROOT" \
    --layers "$LAYERS" --stop-after-layer "$STOP" \
    --pack-tokens $PACK --moe-group $GROUP --shard-cost-overhead $COST_OVERHEAD --shard-count 2 \
    --log "$ROOT/seal.log"
RC=$?
echo "$(date '+%F %T') | seal exit=$RC"
if [ $RC -eq 0 ]; then
    echo "WINDOW $LAYERS SEALED: capture at $ROOT/layer_*, run manifest $ROOT/capture_run_manifest.json"
    echo "PRESERVED for next window: $WORK/shard{0,1}/boundary_$(printf '%03d' $((STOP+1))).bin + topk stores per manifest layer_stream section"
fi
exit $RC
