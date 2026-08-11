#!/bin/bash
# serve-glm52.sh — GLM-5.2 3.0bpw TP4 serve for the M5 end-to-end evidence run.
#
# Usage: serve-glm52.sh <off|dryrun|live> [extra vllm args...]
#   off      VLLM_FQ_ENABLE=0                 — clean A/B baseline
#   dryrun   FQ on, decisions logged only     — safe first boot on the real model
#   live     FQ on, APPLY_MODE=atomic         — experts actually upgrade on the fly
#
# Runs on GPUs 0-3 while the quantization campaign keeps 4-7 (see the
# .reserved-gpus file the supervisor honours). That coexistence is deliberate
# and is also the reason for the JIT isolation below.
set -u
BASE=/home/mbelleau/protensors-work/vllm-voipmonitor/research/fungible-quant
RUN=$BASE/runs/m5-serve
MODE=${1:?usage: serve-glm52.sh <off|dryrun|live> [extra args]}; shift || true

MODEL=${FQ_MODEL:-/home/mbelleau/glm52-k3-assembled}
PORT=${FQ_PORT:-8000}
if [ ! -d "$MODEL" ]; then
  echo "no checkpoint at $MODEL — run the assembly step first" >&2; exit 2
fi

export PROMETHEUS_MULTIPROC_DIR=/home/mbelleau/fq-0c/fq-prom-m5
rm -rf "$PROMETHEUS_MULTIPROC_DIR"; mkdir -p "$PROMETHEUS_MULTIPROC_DIR"

# Private JIT caches. The gg-run defaults are SHARED with the encoder campaign
# on GPUs 4-7, and during M2 every serve boot that overlapped an encode died
# with an illegal memory access in its first traffic wave — including the
# FQ-disabled arm, which is what proved the cause was concurrent JIT cache
# mutation rather than our code. Private dirs cost a slow first JIT only.
export CUDA_CACHE_PATH=/home/mbelleau/cache/jit-m5/cuda
export TRITON_CACHE_DIR=/home/mbelleau/cache/jit-m5/triton
export TORCHINDUCTOR_CACHE_DIR=/home/mbelleau/cache/jit-m5/inductor
mkdir -p "$CUDA_CACHE_PATH" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

# Eager module loading: mitigation for the MLA query-BMM cuBLAS status-14
# flake seen whenever the encoder campaign is hammering the other four GPUs.
export CUDA_MODULE_LOADING=EAGER

# Segment sources for expert upgrades. Local dirs always resolve first; the
# HF repo is the fallback so a missing local K5 layer degrades to a ranged
# fetch instead of failing the swap.
FQ_LOCAL_SEGMENTS=${FQ_LOCAL_SEGMENTS:-/home/mbelleau/glm52-segments}
case "$MODE" in
  off)
    export VLLM_FQ_ENABLE=0
    ;;
  dryrun|live)
    export VLLM_FQ_ENABLE=1
    export VLLM_FQ_MANIFEST_DIR="$FQ_LOCAL_SEGMENTS"
    export VLLM_FQ_SOURCES=malaiwah/GLM-5.2-EXL3-FQ-segments
    export VLLM_FQ_SOURCES_MODE=append
    # Trust: only fragments signed by our pinned key, and only honest rungs.
    # An EMPTY value here is strictly WORSE than an unset one: the resolver
    # branches on `is not None` (fragments.py:495), so exporting "" turns
    # trust filtering OFF entirely — a fragment signed by an unrelated key is
    # ACCEPTED and VLLM_FQ_TRUST_PREDICATES is bypassed — whereas leaving it
    # unset falls back to the manifest's signer_pubkey. A missing key file
    # must therefore abort the boot, not silently downgrade it.
    FQ_PUB=$HOME/.fq_keys/fq_signing.pub
    if [ ! -s "$FQ_PUB" ]; then
      echo "FATAL: $FQ_PUB missing/empty — refusing to boot with trust" \
           "filtering silently disabled. Derive it with fq_repack.Signer," \
           "or set FQ_ALLOW_UNTRUSTED=1 to boot deliberately untrusted." >&2
      [ "${FQ_ALLOW_UNTRUSTED:-0}" = 1 ] || exit 3
    fi
    if [ -s "$FQ_PUB" ]; then
      export VLLM_FQ_TRUST_SIGNERS=$(tr -d ' \n' < "$FQ_PUB")
    else
      echo "WARNING: booting with trust filtering DISABLED (operator opt-in)" >&2
    fi
    export VLLM_FQ_TRUST_PREDICATES=repack-of,encode-of
    export VLLM_FQ_VERIFY=all
    # K5 coverage is partial (the campaign is still encoding). A layer with no
    # K5 fragment must fall back to the K3 base rather than fail the boot.
    export VLLM_FQ_K_FALLBACK=3
    export VLLM_FQ_CACHE_ROOT=/home/mbelleau/fq-0c/fq-cache-m5
    export VLLM_FQ_ARTIFACT_DIR=$RUN/artifacts
    mkdir -p "$VLLM_FQ_ARTIFACT_DIR"
    # Short interval so a demo-length run actually reaches decision points;
    # the spec default of 3000 steps would need hours of traffic. Guards
    # (dwell, hysteresis, jaccard floor) are left at spec defaults on purpose
    # — the point is to show swaps happening THROUGH the guards, not around.
    export VLLM_FQ_INTERVAL_STEPS=${FQ_INTERVAL:-200}
    export VLLM_FQ_DWELL_STEPS=${FQ_DWELL:-250}
    ;;
  *) echo "unknown mode: $MODE" >&2; exit 2 ;;
esac
[ "$MODE" = live ]   && export VLLM_FQ_APPLY_MODE=atomic
[ "$MODE" = dryrun ] && export VLLM_FQ_APPLY_MODE=dryrun

# Dev endpoints: /collective_rpc, /pause, /resume — needed by the reload path
# and by the memory probe.
export VLLM_SERVER_DEV_MODE=1

GG="$BASE/runs/gg-env/gg-run.sh"
export CUDA_VISIBLE_DEVICES=0,1,2,3
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_B12X_MOE=1
export VLLM_USE_B12X_SPARSE_INDEXER=1

echo "=== serve-glm52 mode=$MODE model=$MODEL port=$PORT"
env | grep -E '^VLLM_FQ_' | sort | sed 's/^/    /'

exec "$GG" python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --served-model-name GLM-5.2 \
  --host 127.0.0.1 \
  --port "$PORT" \
  --trust-remote-code \
  --tensor-parallel-size 4 \
  --quantization exl3 \
  --attention-backend B12X_MLA_SPARSE \
  --moe-backend b12x \
  `# 32k, not 8k: vLLM rejects prompt+max_tokens beyond the window, and a` \
  `# truncated GPQA reasoning trace scores as WRONG, so a small window` \
  `# quietly understates quality rather than failing visibly.` \
  --max-model-len ${FQ_MAXLEN:-32768} \
  --max-num-seqs ${FQ_MAXSEQS:-32} \
  --gpu-memory-utilization ${FQ_GPUMEM:-0.92} \
  --kv-cache-dtype fp8_ds_mla --hf-overrides '{"use_index_cache":true}' \
  --worker-extension-cls fq_reload.FqReloadWorker \
  "$@"
