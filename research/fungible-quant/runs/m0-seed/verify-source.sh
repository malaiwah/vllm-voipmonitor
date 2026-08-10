#!/bin/bash
cd "/home/mbelleau/.cache/huggingface/hub/models--brandonmusic--GLM-5.2-EXL3-TR3-3.0bpw/snapshots/9297b9f1d53af5c67cffa01e30cc071a1ff7144b/" || exit 1
awk '{print $2}' MANIFEST.sha256 | xargs -P 48 -I{} sh -c \
  'h=$(sha256sum "{}" | cut -d" " -f1); want=$(awk -v f="{}" '"'"'$2==f{print $1}'"'"' MANIFEST.sha256); \
   if [ "$h" = "$want" ]; then echo "OK {}"; else echo "FAIL {} got $h want $want"; fi'
