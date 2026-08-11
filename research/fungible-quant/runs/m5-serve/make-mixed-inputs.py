#!/usr/bin/env python3
"""Build the inputs for the mixed K3/K5 assembly.

Two products, both cheap and reproducible:

  1. /home/mbelleau/fq-segments-mixed-k3k5 -- a combined segment family that
     SYMLINKS the K3 family (layers 3-78, repack-of brandonmusic@9297b9f1)
     and the K5 family (whichever layers are present locally, encode-of the
     z.ai BF16 base @b4734de4) into one directory, because fq_assemble takes
     a single --segments dir.  Symlinks cost zero disk and leave both source
     families untouched; the per-segment ed25519 attestations are carried
     over unchanged, so the assembler still authenticates every fragment.

  2. runs/m5-serve/policy-mixed-k3k5.json -- an fq-policy/2 document that
     puts N_K5 experts per K5-covered layer at K5 and everything else at K3.

Expert selection: the brandonmusic source ships per-expert reconstruction
error in tier_bitmap.json (`expert_rel_rt_mse`, 256 floats per layer -- the
encoder's own relative round-trip MSE at K3).  We take the highest-error
experts, i.e. the ones K3 damaged most, as the ones with the most to gain
from K5.  This is a static reconstruction-error proxy, NOT the routing-mass
weighted benefit the 0c campaign identified as the real signal; the campaign's
eps data does not cover these layers.  Documented, reproducible, and honest
about being an initial allocation rather than an optimized one.
"""
import json
import sys
from pathlib import Path

K3_FAM = Path("/home/mbelleau/fq-segments/GLM-5.2-EXL3-FQ")
K5_FAM = Path("/home/mbelleau/glm52-segments")
COMBINED = Path("/home/mbelleau/fq-segments-mixed-k3k5")
SRC = Path("/home/mbelleau/.cache/huggingface/hub/models--brandonmusic--GLM-5.2-EXL3-TR3-3.0bpw"
           "/snapshots/9297b9f1d53af5c67cffa01e30cc071a1ff7144b")
RUN = Path("/home/mbelleau/protensors-work/vllm-voipmonitor/research/fungible-quant/runs/m5-serve")
N_K5 = int(sys.argv[1]) if len(sys.argv) > 1 else 64

# ------------------------------------------------ discover coverage on disk
k3_layers = sorted(int(p.name[6:9]) for p in K3_FAM.glob("layer-*.k3.safetensors"))
k5_layers = sorted(int(p.name[6:9]) for p in K5_FAM.glob("layer-*.k5.safetensors"))
print(f"K3 segments present: {len(k3_layers)} layers {k3_layers[0]}-{k3_layers[-1]}")
print(f"K5 segments present: {len(k5_layers)} layers {k5_layers}")

# ------------------------------------------------------ combined family dir
COMBINED.mkdir(exist_ok=True)
(COMBINED / "attestations").mkdir(exist_ok=True)
n_link = 0
for fam, layers, k in ((K3_FAM, k3_layers, 3), (K5_FAM, k5_layers, 5)):
    for L in layers:
        for rel in (f"layer-{L:03d}.k{k}.safetensors",
                    f"attestations/layer-{L:03d}.k{k}.jsonl"):
            src, dst = fam / rel, COMBINED / rel
            if not src.exists():
                raise SystemExit(f"missing {src}")
            if dst.is_symlink() or dst.exists():
                dst.unlink()
            dst.symlink_to(src.resolve())
            n_link += 1
print(f"combined family: {n_link} symlinks in {COMBINED}")

k3_man = json.loads((K3_FAM / "fq-manifest.json").read_text())
k5_man = json.loads((K5_FAM / "fq-manifest.json").read_text())
for field in ("base_model", "layout", "num_experts", "signer_pubkey"):
    if k3_man[field] != k5_man[field]:
        raise SystemExit(f"family {field} mismatch: {k3_man[field]} vs {k5_man[field]}")
(COMBINED / "fq-manifest.json").write_text(json.dumps({
    "schema": "fq-manifest/1",
    "base_model": k3_man["base_model"],
    "layout": k3_man["layout"],
    "num_experts": k3_man["num_experts"],
    "signer_pubkey": k3_man["signer_pubkey"],
    "k_variants": [3, 5],
    "moe_layers": [min(k3_layers), max(k3_layers)],
    "predicate": "repack-of",
    "note": ("Union view: K3 fragments symlinked from fq-segments/GLM-5.2-EXL3-FQ "
             "(repack-of brandonmusic/GLM-5.2-EXL3-TR3-3.0bpw@9297b9f1); K5 "
             "fragments symlinked from glm52-segments (encode-of zai-org/GLM-5.2"
             "@b4734de4). Built by runs/m5-serve/make-mixed-inputs.py."),
    "per_k": {
        "3": {"layers": [min(k3_layers), max(k3_layers)], "segment_count": len(k3_layers),
              "source_repo": "brandonmusic/GLM-5.2-EXL3-TR3-3.0bpw",
              "source_revision": "9297b9f1d53af5c67cffa01e30cc071a1ff7144b"},
        "5": {"layers": [min(k5_layers), max(k5_layers)], "segment_count": len(k5_layers),
              "covered": k5_layers,
              "source_repo": "local:glm52-k5-encode-of",
              "source_revision": "b4734de4facf877f85769a911abafc5283eab3d9"},
    },
}, indent=1) + "\n")

# ------------------------------------------------------------------ policy
bitmap = json.loads((SRC / "tier_bitmap.json").read_text())
bpe, chosen = {}, {}
for L in k3_layers:
    bits = [3] * 256
    if L in k5_layers:
        mse = bitmap[str(L)]["expert_rel_rt_mse"]
        if len(mse) != 256:
            raise SystemExit(f"layer {L}: expert_rel_rt_mse has {len(mse)} entries")
        # highest reconstruction error first; index as the deterministic tiebreak
        top = sorted(range(256), key=lambda e: (-mse[e], e))[:N_K5]
        for e in top:
            bits[e] = 5
        chosen[str(L)] = sorted(top)
    bpe[str(L)] = bits

policy = {
    "schema": "fq-policy/2",
    "base_model": k3_man["base_model"],
    "layout": k3_man["layout"],
    "note": (f"Mixed K3/K5. Layers with local K5 segments ({k5_layers}) place the "
             f"{N_K5} highest-expert_rel_rt_mse experts at K5 and the remaining "
             f"{256 - N_K5} at K3; every other MoE layer is pure K3. Selection is "
             "the source encoder's static per-expert round-trip MSE, not a "
             "routing-mass-weighted benefit -- an initial allocation for the swap "
             "demo, not an optimized one."),
    "budget": {"mode": "fixed_cardinality",
               "n_k5_per_layer": {str(L): N_K5 for L in k5_layers}},
    "selection": {"criterion": "tier_bitmap.json:expert_rel_rt_mse desc",
                  "experts_per_covered_layer": N_K5,
                  "covered_layers": k5_layers,
                  "chosen_experts": chosen},
    "bits_per_expert": bpe,
}
out = RUN / "policy-mixed-k3k5.json"
out.write_text(json.dumps(policy, indent=1) + "\n")

k5_total = sum(b.count(5) for b in bpe.values())
print(f"policy: {len(bpe)} layers, {len(k5_layers)} mixed, {k5_total} K5 experts total")
print(f"wrote {out}")

# ---------------------------------------------------- memory budget (TP4)
PER_EXPERT_K3, PER_EXPERT_K5 = 14_315_568, 23_752_752
delta = (PER_EXPERT_K5 - PER_EXPERT_K3) * k5_total
src_bytes = sum(p.stat().st_size for p in SRC.glob("model-*.safetensors"))
GiB = 1024 ** 3
print(f"pure K3 total {src_bytes/1e9:.1f} GB -> per rank (TP4) {src_bytes/4/GiB:.2f} GiB "
      f"({src_bytes/4/GiB/95.6*100:.1f}% of 95.6 GiB)")
print(f"mixed adds {delta/1e9:.2f} GB -> per rank +{delta/4/GiB:.2f} GiB")
print(f"mixed total {(src_bytes+delta)/1e9:.1f} GB -> per rank {(src_bytes+delta)/4/GiB:.2f} GiB "
      f"({(src_bytes+delta)/4/GiB/95.6*100:.1f}% of 95.6 GiB)")
