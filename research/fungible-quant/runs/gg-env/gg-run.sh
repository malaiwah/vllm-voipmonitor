#!/bin/bash
# gg-run.sh — run a command inside the extracted GG v20-r33 rootfs env.
#
# No container runtime needed (JarvisAI managed container: namespaces disabled).
# Mechanism: the image's own dynamic loader + library path + PYTHONHOME, with
# the host driver libs appended last. Absolute symlinks inside the rootfs do
# NOT resolve into the image (no chroot) — always use real paths.
#
# Usage: gg-run.sh python -c "import torch"        (python -> image python3.12)
#        gg-run.sh env                              (inspect effective env)
set -u
ROOT=${GG_ROOT:-/home/mbelleau/rootfs/gg-v20-r33}
PY="$ROOT/usr/bin/python3.12"
LOADER="$ROOT/lib/x86_64-linux-gnu/ld-linux-x86-64.so.2"
NCCL="$ROOT/opt/libnccl.so.2.30.4"   # real file; /opt symlink is absolute+broken outside chroot

LIBP="$ROOT/lib/x86_64-linux-gnu"
LIBP="$LIBP:$ROOT/usr/lib/x86_64-linux-gnu"
LIBP="$LIBP:$ROOT/usr/local/cuda-13.2/lib64"
LIBP="$LIBP:$ROOT/usr/local/cuda-13.2/targets/x86_64-linux/lib"
LIBP="$LIBP:$ROOT/opt/venv/lib/python3.12/site-packages/torch/lib"
LIBP="$LIBP:/usr/lib/x86_64-linux-gnu"   # host driver: libcuda.so.1, libnvidia-ml

cmd=$1; shift
case "$cmd" in
  python|python3|python3.12) target="$PY" ;;
  *) target="$cmd" ;;
esac

# $PY is a shim that re-execs through the image loader with --argv0, so
# sys.executable-based child spawns (vLLM registry inspect, multiprocessing
# spawn, EngineCore) recurse correctly. Real binary: python3.12.real.
exec env \
  CUDA_HOME="$ROOT/usr/local/cuda-13.2" \
  PATH="$ROOT/usr/local/cuda-13.2/bin:$PATH" \
  CPATH="$ROOT/usr/include:$ROOT/usr/include/x86_64-linux-gnu" \
  PYTHONHOME="$ROOT/usr" \
  PYTHONPATH="$ROOT/opt/venv/lib/python3.12/site-packages:$ROOT/opt/venv/lib/python3.12/site-packages/nvidia_cutlass_dsl/dsl_packages:$ROOT/opt/exllamav3-python:$ROOT/opt/exllamav3:/home/mbelleau/gg-extra" \
  PYTHONDONTWRITEBYTECODE=1 \
  CUDA_DEVICE_ORDER=PCI_BUS_ID \
  TORCH_CUDA_ARCH_LIST=12.0a \
  FLASHINFER_CUDA_ARCH_LIST=12.0f \
  VLLM_EXL3_EXT_PATH="$ROOT/opt/exllamav3" \
  VLLM_NCCL_SO_PATH="$NCCL" \
  NCCL_LOCAL_INFERENCE_PATH="$NCCL" \
  CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-/home/mbelleau/cache/jit/cuda}" \
  TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/home/mbelleau/cache/jit/triton}" \
  "$target" "$@"
