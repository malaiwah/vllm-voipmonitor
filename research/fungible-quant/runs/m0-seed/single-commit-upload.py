"""One-commit publish: Xet-dedupes already-uploaded chunks, 1 commit call."""
import time
from huggingface_hub import HfApi
api = HfApi()
for attempt in range(20):
    try:
        api.upload_folder(
            repo_id="malaiwah/GLM-5.2-EXL3-FQ-segments",
            folder_path="/home/mbelleau/fq-segments/GLM-5.2-EXL3-FQ",
            ignore_patterns=["state.json", "*.part", ".huggingface*"],
            commit_message="fq_repack: full K3 segment family + attestations + card (single commit)")
        print("UPLOAD COMMITTED")
        break
    except Exception as e:
        print(f"attempt {attempt}: {type(e).__name__}: {str(e)[:200]}")
        time.sleep(120)
