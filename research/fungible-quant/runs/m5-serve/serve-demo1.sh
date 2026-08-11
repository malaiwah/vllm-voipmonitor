#!/bin/bash
# serve-demo1.sh — GLM-5.2 booted DIRECTLY from Progressive Tensors segments
# plus a bitrate policy. No assembled mixed checkpoint on disk.
#
# This is demo 1: start from a flat 3.0bpw base, let the loader place the
# experts the policy asks for at K4, and — the actual point — let fragments
# that do NOT exist yet resolve down the K ladder and enqueue an on-the-fly
# encode instead of failing. Partial K4 coverage is not a blocker here, it is
# the thing being demonstrated.
#
# Usage: serve-demo1.sh <policy.json> [port] [extra vllm args...]
set -eu
BASE=/home/mbelleau/protensors-work/vllm-voipmonitor/research/fungible-quant
RUN=$BASE/runs/m5-serve
POLICY=${1:?usage: serve-demo1.sh <policy.json> [port] [extra vllm args...]}
PORT=${2:-8000}
# Consume both, or they leak into "$@" and vLLM's argparse rejects the port as
# an unrecognized positional.
shift $(( $# >= 2 ? 2 : 1 ))

GG=$BASE/runs/gg-env/gg-run.sh

# PRE-FLIGHT: refuse to boot onto occupied cards.
# A TP4 serve killed by signalling its process group can leave all four
# workers alive holding ~14.7 GiB each -- observed: four orphans still
# resident 46 minutes after the kill, while a fresh boot tried to start on
# top of them. Starting anyway either OOMs at KV sizing or silently halves
# the cache. Check the DEVICES WE WILL USE, not the whole box: GPUs 4-7 are
# the campaign's and are expected to be busy.
FQ_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
_busy=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
        -i "$FQ_DEVICES" 2>/dev/null | awk -F', *' '$2 > 1024 {printf "%s(%sMiB) ", $1, $2}')
if [ -n "$_busy" ] && [ "${FQ_REAP_STALE:-1}" = 1 ]; then
  # Reap first, then re-check. The workers are matched by the DEVICE they
  # hold, not by a command-line pattern: they exec through the rootfs
  # ld-linux shim, so their argv does not contain "vllm" or "VLLM::" and
  # every pattern-based reap silently matched nothing.
  for _pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader \
                -i "$FQ_DEVICES" 2>/dev/null); do
    echo "  reaping stale GPU process $_pid" >&2
    kill -9 "$_pid" 2>/dev/null
  done
  sleep 8
  _busy=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
          -i "$FQ_DEVICES" 2>/dev/null | awk -F', *' '$2 > 1024 {printf "%s(%sMiB) ", $1, $2}')
fi
if [ -n "$_busy" ]; then
  echo "FATAL: GPUs still occupied after reap: $_busy" >&2
  echo "  Stale workers from a previous serve? Check:" >&2
  echo "    nvidia-smi --query-compute-apps=pid,used_memory --format=csv" >&2
  echo "  Then kill them and retry. Set FQ_IGNORE_BUSY_GPUS=1 to override." >&2
  [ "${FQ_IGNORE_BUSY_GPUS:-0}" = 1 ] || exit 5
fi

# DEPLOY FIRST. The serve loads exl3_fungible from the extracted rootfs, NOT
# from the source tree, so editing and committing code has no effect on the
# next boot -- silently. That has now cost three boots: one without the histc
# fix, one without the composition table, and one 97-hour load that ran the
# pre-prefetch fetch path. Never trust that the rootfs matches; make it match.
"$BASE/runs/gg-env/deploy-fq.sh" || {
  echo "FATAL: deploy-fq.sh failed — refusing to boot possibly-stale code" >&2
  exit 4
}

# --- fragment sources, in resolution order -------------------------------
# Local segment dirs always resolve first; the HF repo is the fallback so a
# layer we have not encoded locally is a ranged fetch, not a failure.
export VLLM_FQ_MANIFEST_DIR=${VLLM_FQ_MANIFEST_DIR:-/home/mbelleau/glm52-segments}
export VLLM_FQ_SOURCES=${VLLM_FQ_SOURCES:-malaiwah/GLM-5.2-EXL3-FQ-segments}
export VLLM_FQ_SOURCES_MODE=append
export VLLM_FQ_POLICY=$POLICY
# Non-expert tensors (attention, router, shared experts, norms) come from the
# assembled K3 base byte-exact; only the routed experts are placed per policy.
export VLLM_FQ_DENSE_SOURCE=${VLLM_FQ_DENSE_SOURCE:-/home/mbelleau/glm52-k3-assembled}
export VLLM_FQ_CACHE=${VLLM_FQ_CACHE:-/home/mbelleau/cache/fq-demo1}
mkdir -p "$VLLM_FQ_CACHE"

# --- the point of the demo -----------------------------------------------
# A requested K with no fragment must degrade, never crash: walk DOWN the
# ladder (auto = nearest lower K), log it loudly, and enqueue the miss for an
# out-of-band encode. Boot never blocks on an encoder.
export VLLM_FQ_K_FALLBACK=${VLLM_FQ_K_FALLBACK:-auto}
# Whole-segment prefetch goes through hf_hub_download, which is the ONLY way
# to get parallel chunked transfer and Xet dedup -- a raw urllib stream gets
# neither, and measured 45 MiB/min on the per-expert path.
#
# NOT hf_transfer: this huggingface_hub deprecated it ("'hf_transfer' is not
# used anymore") in favour of Xet, so HF_HUB_ENABLE_HF_TRANSFER is now a no-op
# that only emits a FutureWarning. HF_XET_HIGH_PERFORMANCE is the live knob.
export HF_XET_HIGH_PERFORMANCE=${HF_XET_HIGH_PERFORMANCE:-1}
# Authenticate. The Hub said so itself on attempt 7, once per rank:
#   "You are sending unauthenticated requests to the HF Hub. Please set a
#    HF_TOKEN to enable higher rate limits and faster downloads."
# Sourced in a SUBSHELL that prints only the token, so no other secret in
# ~/.fq_env leaks into the server environment and nothing is ever echoed.
if [ -z "${HF_TOKEN:-}" ] && [ -r "$HOME/.fq_env" ]; then
  HF_TOKEN=$( . "$HOME/.fq_env" >/dev/null 2>&1
              printf '%s' "${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}" )
  export HF_TOKEN
fi
# Segments download DURING load, so a progressive boot is legitimately slower
# than reading a local checkpoint -- attempt 5 died on the 600 s default with
# the load still progressing normally. Prefetch depth+width cut the wall time,
# but the ceiling has to admit a first boot on a cold cache.
export VLLM_ENGINE_READY_TIMEOUT_S=${VLLM_ENGINE_READY_TIMEOUT_S:-5400}
# Layers ahead to prefetch, x2 concurrent objects each. Footprint stays
# bounded because release_layer() drops each layer once it hits the GPU.
export VLLM_FQ_PREFETCH_DEPTH=${VLLM_FQ_PREFETCH_DEPTH:-3}
# Keep whole downloaded layer objects so a later K3->K4 promotion slices from
# a local file instead of re-fetching ~2.5 GB. Off by default: this trades
# bounded-by-depth footprint for bounded-by-model (hundreds of GB).
export VLLM_FQ_KEEP_LAYERS=${VLLM_FQ_KEEP_LAYERS:-1}
export VLLM_FQ_ENCODE_QUEUE=${VLLM_FQ_ENCODE_QUEUE:-$RUN/results/demo1/encode-queue.jsonl}
mkdir -p "$(dirname "$VLLM_FQ_ENCODE_QUEUE")"

# --- trust ---------------------------------------------------------------
# An EMPTY signer list is WORSE than an unset one (the resolver branches on
# `is not None`, so "" disables filtering entirely). Fail rather than boot
# silently untrusted.
FQ_PUB=$HOME/.fq_keys/fq_signing.pub
if [ ! -s "$FQ_PUB" ] && [ "${FQ_ALLOW_UNTRUSTED:-0}" != 1 ]; then
  echo "FATAL: $FQ_PUB missing — refusing to boot with trust filtering off" >&2
  exit 3
fi
[ -s "$FQ_PUB" ] && export VLLM_FQ_TRUST_SIGNERS=$(tr -d ' \n' < "$FQ_PUB")
export VLLM_FQ_TRUST_PREDICATES=repack-of,encode-of
export VLLM_FQ_VERIFY=all

# --- the fungible loop ---------------------------------------------------
export VLLM_FQ_ENABLE=1
export VLLM_FQ_APPLY_MODE=${VLLM_FQ_APPLY_MODE:-atomic}
export VLLM_FQ_INTERVAL_STEPS=${FQ_INTERVAL:-100}
export VLLM_FQ_DWELL_STEPS=${FQ_DWELL:-1}
export VLLM_FQ_GATE_MASS=${VLLM_FQ_GATE_MASS:-0}
export VLLM_FQ_TABLE_EVERY_INTERVALS=${FQ_TABLE:-3}
export VLLM_FQ_ARTIFACT_DIR=$RUN/results/demo1/artifacts
export VLLM_FQ_CACHE_ROOT=$RUN/results/demo1/fq-cache
export VLLM_FQ_DUMP_STATS=${VLLM_FQ_DUMP_STATS:-$RUN/results/demo1/stats.jsonl}
mkdir -p "$VLLM_FQ_ARTIFACT_DIR" "$VLLM_FQ_CACHE_ROOT"

# --- runtime -------------------------------------------------------------
export PROMETHEUS_MULTIPROC_DIR=/home/mbelleau/fq-0c/fq-prom-demo1
rm -rf "$PROMETHEUS_MULTIPROC_DIR"; mkdir -p "$PROMETHEUS_MULTIPROC_DIR"
# Private JIT caches: sharing them with the encoder campaign killed every M2
# boot with an illegal memory access, including the FQ-disabled arm.
export CUDA_CACHE_PATH=/home/mbelleau/cache/jit-m5/cuda
export TRITON_CACHE_DIR=/home/mbelleau/cache/jit-m5/triton
export TORCHINDUCTOR_CACHE_DIR=/home/mbelleau/cache/jit-m5/inductor
mkdir -p "$CUDA_CACHE_PATH" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"
export CUDA_MODULE_LOADING=EAGER
export VLLM_SERVER_DEV_MODE=1

# --- FQ_FAST=1: loader-iteration mode ------------------------------------
# Trades serving throughput for turnaround. When the thing under test is the
# LOADER, ~9 min of CUDA JIT plus inductor compilation plus CUDA-graph capture
# is pure tax on every experiment -- and every one of those minutes is spent
# after the part we are trying to observe. Eager mode is 2-3x slower to decode
# and boots minutes sooner; that is the right trade until the loader is good.
#
# Deliberately NOT touched: the JIT caches (they only help) and the fragment
# cache (BT-2 needs it warm).
FQ_FAST=${FQ_FAST:-0}
FAST_ARGS=""
if [ "$FQ_FAST" = 1 ]; then
  FAST_ARGS="--enforce-eager -O0"
  # EAGER module loading was an M2 mitigation for a cuBLAS status-14 flake
  # that only appeared while the encoder campaign hammered GPUs 4-7. With the
  # campaign paused, LAZY skips loading kernels we never call.
  export CUDA_MODULE_LOADING=${CUDA_MODULE_LOADING_FAST:-LAZY}
  export VLLM_FQ_TABLE_EVERY_INTERVALS=${FQ_TABLE:-1}
  echo "    MODE         : FAST — eager, no inductor, no graph capture."
  echo "                   Decode throughput is NOT comparable to a normal run."
fi

# --- memory preflight -----------------------------------------------------
# Hand the loader the same utilization the engine will use, so its projection
# is the engine's arithmetic and not a guess that can drift out of sync.
export VLLM_FQ_BUDGET_UTIL=${FQ_GPUMEM:-0.92}
export VLLM_FQ_BUDGET_MIN_KV=${FQ_MIN_KV:-4g}
# Eager mode skips graph capture, so it leaves less residue to reserve for.
if [ "$FQ_FAST" = 1 ]; then
  export VLLM_FQ_BUDGET_OVERHEAD=${FQ_OVERHEAD:-4.5g}
else
  export VLLM_FQ_BUDGET_OVERHEAD=${FQ_OVERHEAD:-5.5g}
fi
export CUDA_VISIBLE_DEVICES=0,1,2,3
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_B12X_MOE=1
export VLLM_USE_B12X_SPARSE_INDEXER=1

# Synthesize the mixed hybrid_tr3_tail + quantization_config stub from the
# policy (the GG loader contract from runs/serve-baseline/fruit-mixed-report.md).
OV=$("$GG" python -m vllm.model_executor.layers.quantization.exl3_fungible.progressive \
  --extra-overrides '{"use_index_cache":true}')

echo "=== demo1: progressive boot from segments"
echo "    policy       : $POLICY"
echo "    segments     : $VLLM_FQ_MANIFEST_DIR (+ HF $VLLM_FQ_SOURCES)"
echo "    dense source : $VLLM_FQ_DENSE_SOURCE"
echo "    apply mode   : $VLLM_FQ_APPLY_MODE   K-fallback: $VLLM_FQ_K_FALLBACK"
echo "    encode queue : $VLLM_FQ_ENCODE_QUEUE"
# Presence only -- never the value.
if [ -n "${HF_TOKEN:-}" ]; then
  echo "    hf token     : present (authenticated Hub requests)"
else
  echo "    hf token     : ABSENT — downloads will be rate-limited and slower"
fi
echo "    prefetch     : depth=$VLLM_FQ_PREFETCH_DEPTH x 2 objects, Xet hi-perf=$HF_XET_HIGH_PERFORMANCE"
if [ "${VLLM_FQ_KEEP_LAYERS}" = 1 ]; then
  echo "    keep layers  : ON — whole layers retained for local re-tiering (disk grows to model size)"
else
  echo "    keep layers  : off — layers released after load (footprint O(prefetch depth))"
fi

exec "$GG" python -m vllm.entrypoints.openai.api_server \
  --model "$VLLM_FQ_DENSE_SOURCE" \
  --served-model-name GLM-5.2 \
  --host 127.0.0.1 --port "$PORT" \
  --trust-remote-code \
  --tensor-parallel-size 4 \
  --quantization exl3 \
  --load-format progressive \
  --attention-backend B12X_MLA_SPARSE \
  --moe-backend b12x \
  --max-model-len ${FQ_MAXLEN:-32768} \
  --max-num-seqs ${FQ_MAXSEQS:-32} \
  --gpu-memory-utilization ${FQ_GPUMEM:-0.92} \
  --kv-cache-dtype fp8_ds_mla \
  --hf-overrides "$OV" \
  --worker-extension-cls fq_reload.FqReloadWorker \
  $FAST_ARGS \
  "$@"
