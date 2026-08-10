#!/bin/bash
# M0 gate offline half at scale: assemble all 76 MoE layers all-K3, sha-compare vs source.
set -u
/home/mbelleau/venvs/fq/bin/python /home/mbelleau/protensors-work/vllm-voipmonitor/research/fungible-quant/tools/fq_assemble.py \
  --segments /home/mbelleau/fq-segments/GLM-5.2-EXL3-FQ --source "/home/mbelleau/.cache/huggingface/hub/models--brandonmusic--GLM-5.2-EXL3-TR3-3.0bpw/snapshots/9297b9f1d53af5c67cffa01e30cc071a1ff7144b/" \
  --policy /home/mbelleau/fq-0c/policy-all-k3.json --out /home/mbelleau/fq-0c/assembled-all-k3 || exit 1
fails=0
for f in /home/mbelleau/fq-0c/assembled-all-k3/model-layer-*.safetensors; do
  b=$(basename $f)
  a=$(sha256sum "/home/mbelleau/.cache/huggingface/hub/models--brandonmusic--GLM-5.2-EXL3-TR3-3.0bpw/snapshots/9297b9f1d53af5c67cffa01e30cc071a1ff7144b//$b" | cut -d' ' -f1); c=$(sha256sum "$f" | cut -d' ' -f1)
  if [ "$a" = "$c" ]; then echo "OK $b"; rm "$f"; else echo "MISMATCH $b"; fails=$((fails+1)); fi
done
echo "assembly verify complete: $fails mismatches"
