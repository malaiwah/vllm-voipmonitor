#!/bin/bash
# Full M0 seed: repack all layers + incremental publish. Resumable (state.json).
set -u
source ~/.fq_env
exec /home/mbelleau/venvs/fq/bin/python \
  /home/mbelleau/protensors-work/vllm-voipmonitor/research/fungible-quant/tools/fq_repack.py \
  --snapshot "/home/mbelleau/.cache/huggingface/hub/models--brandonmusic--GLM-5.2-EXL3-TR3-3.0bpw/snapshots/9297b9f1d53af5c67cffa01e30cc071a1ff7144b/" \
  --source-repo brandonmusic/GLM-5.2-EXL3-TR3-3.0bpw \
  --revision 9297b9f1d53af5c67cffa01e30cc071a1ff7144b \
  --base-model zai-org/GLM-5.2 \
  --out /home/mbelleau/fq-segments/GLM-5.2-EXL3-FQ \
  --publish malaiwah/GLM-5.2-EXL3-FQ-segments
