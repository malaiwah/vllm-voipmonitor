#!/usr/bin/env python3
"""Extract GLM-5.2 expert weights for new layers (30, 50, 60, 70).
Extracts from already-downloaded shards only. 10 experts per layer.
One projection at a time to minimize RAM."""
import torch, json, os, gc
from safetensors import safe_open
from pathlib import Path

SHARD_DIR = "/hf_cache/hub/models--zai-org--GLM-5.2/snapshots/b4734de4facf877f85769a911abafc5283eab3d9"
OUT_DIR = Path("/tmp/poc_residual/glm52_data")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_LAYERS = [30, 50, 60, 70]
N_EXPERTS = 10  # v48 proved CV=0.11% across 70 experts — 10 is plenty

def get_shard_keys_for_layer(layer_idx):
    """Read index.json to find which shard has each expert's keys."""
    index_path = f"{SHARD_DIR}/model.safetensors.index.json"
    with open(index_path) as f:
        idx = json.load(f)
    wm = idx["weight_map"]
    prefix = f"model.layers.{layer_idx}.mlp.experts."
    # Map: shard_name -> {expert_idx: {proj: key}}
    shard_map = {}
    for k, v in wm.items():
        if k.startswith(prefix) and k.endswith(".weight"):
            parts = k.split(".")
            eidx = int(parts[5])
            proj = parts[6]
            shard_map.setdefault(v, {}).setdefault(eidx, {})[proj] = k
    return shard_map

def extract_layer(layer_idx, shard_map):
    """Extract 10 experts, one projection at a time."""
    # Find which shards we actually have on disk
    available_shards = {}
    for shard, experts in shard_map.items():
        if os.path.exists(f"{SHARD_DIR}/{shard}"):
            available_shards[shard] = experts
    
    # Collect all available expert indices
    all_expert_indices = set()
    for experts in available_shards.values():
        all_expert_indices.update(experts.keys())
    all_expert_indices = sorted(all_expert_indices)
    
    # Pick first N_EXPERTS
    selected = all_expert_indices[:N_EXPERTS]
    print(f"Layer {layer_idx}: {len(all_expert_indices)} available, selecting {len(selected)}: {selected}", flush=True)
    
    # For each projection, load from shards one at a time
    for proj in ["gate_proj", "up_proj", "down_proj"]:
        tensors = {}
        for shard_name, experts_in_shard in available_shards.items():
            # Only open this shard if it has any of our selected experts with this proj
            needed = [e for e in selected if e in experts_in_shard and proj in experts_in_shard[e]]
            if not needed:
                continue
            f = safe_open(f"{SHARD_DIR}/{shard_name}", framework="pt")
            for eidx in needed:
                key = experts_in_shard[eidx][proj]
                tensors[eidx] = f.get_tensor(key).float()
            del f
        
        # Stack in expert index order
        stacked = torch.stack([tensors[e] for e in selected])
        out_path = OUT_DIR / f"layer{layer_idx}_all_{proj}.pt"
        torch.save(stacked, out_path)
        print(f"  Saved {proj}: {tuple(stacked.shape)} -> {out_path.name}", flush=True)
        
        del tensors, stacked
        gc.collect()

if __name__ == "__main__":
    for layer in TARGET_LAYERS:
        shard_map = get_shard_keys_for_layer(layer)
        extract_layer(layer, shard_map)
    print("Done!", flush=True)
