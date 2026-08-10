#!/bin/bash
# M0 seed phase 2: resumable bulk publish of the segment tree.
set -u
source ~/.fq_env
/home/mbelleau/venvs/fq/bin/python - <<'PY'
from huggingface_hub import HfApi
api = HfApi()
api.upload_large_folder(
    repo_id="malaiwah/GLM-5.2-EXL3-FQ-segments",
    folder_path="/home/mbelleau/fq-segments/GLM-5.2-EXL3-FQ",
    repo_type="model",
    ignore_patterns=["state.json", "*.part", ".huggingface/**"],
)
print("upload complete")
PY
