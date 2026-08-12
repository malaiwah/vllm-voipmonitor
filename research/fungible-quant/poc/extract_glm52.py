#!/usr/bin/env python3
"""Extract GLM-5.2 expert weights from HuggingFace shards."""
import torch, json, os, sys
from safetensors import safe_open
from pathlib import Path

SHARD_L10 = "/hf_cache/hub/models--zai-org--GLM-5.2/snapshots/b4734de4facf877f85769a911abafc5283eab3d9/model-00002-of-00282.safetensors"
SHARD_L40 = "/hf_cache/hub/models--zai-org--GLM-5.2/snapshots/b4734de4facf877f85769a911abafc5283eab3d9/model-00120-of-00282.safetensors"
OUT_DIR = Path("/tmp/poc_residual/glm52_data")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def extract_layer(shard_path, layer_idx, max_experts=256):
    print(f"Extracting layer {layer_idx} from {os.path.basename(shard_path)}...", flush=True)
    f = safe_open(shard_path, framework="pt")
    keys = list(f.keys())
    prefix = f"model.layers.{layer_idx}.mlp.experts."
    expert_keys = [k for k in keys if k.startswith(prefix) and k.endswith(".weight")]
    print(f"  Found {len(expert_keys)} expert weight keys", flush=True)
    experts = {}
    for k in expert_keys:
        parts = k.split(".")
        eidx = int(parts[5])
        proj = parts[6]
        if eidx >= max_experts: continue
        if eidx not in experts: experts[eidx] = {}
        experts[eidx][proj] = f.get_tensor(k).float()
    print(f"  Extracted {len(experts)} experts", flush=True)
    for eidx in sorted(experts.keys()):
        shapes = {p: tuple(experts[eidx][p].shape) for p in experts[eidx]}
        print(f"    Expert {eidx}: {shapes}", flush=True)
    for eidx in experts:
        for proj in experts[eidx]:
            torch.save(experts[eidx][proj], OUT_DIR / f"layer{layer_idx}_exp{eidx}_{proj}.pt")
    all_experts = sorted(experts.keys())
    for proj in ["gate_proj", "up_proj", "down_proj"]:
        if proj not in experts[all_experts[0]]: continue
        stacked = torch.stack([experts[e][proj] for e in all_experts])
        torch.save(stacked, OUT_DIR / f"layer{layer_idx}_all_{proj}.pt")
        print(f"  Saved stacked {proj}: {tuple(stacked.shape)}", flush=True)
    return list(experts.keys())

if __name__ == "__main__":
    l10 = extract_layer(SHARD_L10, 10)
    l40 = extract_layer(SHARD_L40, 40)
    print(f"\nLayer 10: {len(l10)} experts: {l10}", flush=True)
    print(f"Layer 40: {len(l40)} experts: {l40}", flush=True)
    print("Done!", flush=True)
