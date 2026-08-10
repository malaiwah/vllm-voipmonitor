#!/bin/bash
# Rolling publish for GLM-5.2 window 1: repack K2/K5 encoder outputs into
# segments (encode-of lineage) and upload into the K3 seed's HF family.
set -u
source ~/.fq_env
for K in 2 5; do
  STAGE=/home/mbelleau/glm52-segments-stage/k$K
  mkdir -p "$STAGE"
  for f in /home/mbelleau/glm52-work-k$K/tr3-layer-*.safetensors; do
    L=$(basename "$f" | grep -oE '[0-9]+')
    ln -sf "$f" "$STAGE/model-layer-$L.safetensors"
  done
  /home/mbelleau/venvs/fq/bin/python \
    /home/mbelleau/protensors-work/vllm-voipmonitor/research/fungible-quant/tools/fq_repack.py \
    --snapshot "$STAGE" \
    --source-repo "local:glm52-k$K-encode-of-window1" \
    --revision "$(cat /home/mbelleau/glm52-capture/capture_run_manifest.json | /home/mbelleau/venvs/fq/bin/python -c 'import json,sys; print(json.load(sys.stdin).get("fingerprint","unknown"))')" \
    --base-model zai-org/GLM-5.2 --k $K \
    --out /home/mbelleau/glm52-segments || exit 1
done
/home/mbelleau/venvs/fq/bin/python - <<'PY'
from huggingface_hub import HfApi
api = HfApi()
api.upload_large_folder(repo_id="malaiwah/GLM-5.2-EXL3-FQ-segments",
    folder_path="/home/mbelleau/glm52-segments", repo_type="model",
    ignore_patterns=["state.json", "*.part", ".huggingface*"])
print("WINDOW-1 K2/K5 SEGMENTS PUBLISHED")
PY
