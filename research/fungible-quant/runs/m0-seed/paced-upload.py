"""Paced per-file publisher: guaranteed progress under free-tier API quota."""
import time
from pathlib import Path
from huggingface_hub import HfApi
REPO = "malaiwah/GLM-5.2-EXL3-FQ-segments"
ROOT = Path("/home/mbelleau/fq-segments/GLM-5.2-EXL3-FQ")
api = HfApi()
existing = set(api.list_repo_files(REPO))
todo = [p for p in sorted(ROOT.rglob("*")) if p.is_file()
        and not p.name.endswith((".part", ".tmp")) and p.name != "state.json"
        and not any(part.startswith(".") for part in p.relative_to(ROOT).parts)
        and str(p.relative_to(ROOT)) not in existing]
print(f"{len(todo)} files to publish (of {sum(1 for _ in ROOT.rglob('*') if _.is_file())})", flush=True)
for i, p in enumerate(todo):
    rel = str(p.relative_to(ROOT))
    for attempt in range(10):
        try:
            api.upload_file(path_or_fileobj=str(p), path_in_repo=rel,
                            repo_id=REPO, commit_message=f"fq: {rel}")
            print(f"[{i+1}/{len(todo)}] {rel}", flush=True)
            break
        except Exception as e:
            print(f"[{i+1}/{len(todo)}] {rel} retry {attempt}: {str(e)[:120]}", flush=True)
            time.sleep(300)
    time.sleep(65)
print("PACED UPLOAD COMPLETE", flush=True)
