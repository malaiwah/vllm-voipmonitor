# runs/ — on-box job state (ONBOX-BOOTSTRAP rule 3)

Session 2026-08-10, box: 8x RTX PRO 6000 (SM120), 224 cores, 1.5 TB RAM,
3.0 TB /home (persistent; everything else is ephemeral). We run INSIDE a
docker container: no nested runtime, no sudo — GG images are pulled as OCI
layouts by `dl-gg-images/ghcr_pull.py` and their rootfs will be extracted
for direct venv use.

| Job | What | Pinned revision | Dest |
|---|---|---|---|
| `dl-glm52-orig` | zai-org/GLM-5.2 (1506.7 GB, 282 shards) | `b4734de4facf877f85769a911abafc5283eab3d9` | HF cache `~/.cache/huggingface` |
| `dl-glm52-k3` | brandonmusic/GLM-5.2-EXL3-TR3-3.0bpw (316.6 GB) | `9297b9f1d53af5c67cffa01e30cc071a1ff7144b` | HF cache |
| `dl-gg-images` | ghcr.io/malaiwah/gilded-gnosis-v20:r12-field-review + glm52-exl3-vast:latest | by tag→digest at pull time | `/home/mbelleau/images/<name>/` (OCI layout) |

All jobs: tmux session `fq` (windows `orig`, `k3`, `images`), nohup, retry
loops, resumable after interruption (HF cache resume / sha256-verified blob
cache). Relaunch any job by re-running its `launch.sh`.

Known issue: GPU 4 phantom 100% util (P0, no processes). `nvidia-smi -r -i 4`
denied (non-root container). Will A/B-bench GPU 4 vs 5 once the GG env is
extracted; provider-side reset/reboot if it underperforms.
