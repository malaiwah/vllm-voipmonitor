#!/usr/bin/env python3
"""Finalize + verify an assembled GLM-5.2 checkpoint (pure K3 or mixed K3/K5).

Usage: finalize.py {k3|mixed}

Runs after the matching assemble-*.sh has materialized every shard.  Does:

  1. installs the assembler-corrected metadata (config.json with the exl3
     quantization_config, tier_bitmap.json, tokenizer, ...) into the output;
  2. regenerates model.safetensors.index.json and MANIFEST.sha256 from the
     bytes actually present, using fq_assemble's own functions;
  3. writes a merged full-model fq-assembly.json whose fragment list is
     rebuilt from the ed25519-signed segment attestations (re-verified here
     under the pinned signer) and whose product list is the per-batch
     evidence recorded at assembly time;
  4. checks the file set, index self-consistency, tensor/parameter counts,
     the declared bit allocation, and the TP4 per-rank memory footprint.
"""
import json
import re
import shutil
import sys
from pathlib import Path

TOOLS = Path("/home/mbelleau/protensors-work/vllm-voipmonitor/research/fungible-quant/tools")
sys.path.insert(0, str(TOOLS))

import fq_assemble as FA  # noqa: E402
import fq_trust as FT  # noqa: E402

MODE = (sys.argv[1] if len(sys.argv) > 1 else "k3").lower()
RUN = Path("/home/mbelleau/protensors-work/vllm-voipmonitor/research/fungible-quant/runs/m5-serve")
SRC = Path("/home/mbelleau/.cache/huggingface/hub/models--brandonmusic--GLM-5.2-EXL3-TR3-3.0bpw"
           "/snapshots/9297b9f1d53af5c67cffa01e30cc071a1ff7144b")
SIGNER = "a58b7bb79ba5845716aa6fee7d54e714ef243c2875f23a617e1ef3247c565525"
GiB = 1024 ** 3
CARD_GiB = 95.6

if MODE == "k3":
    OUT = Path("/home/mbelleau/glm52-k3-assembled")
    SEG = Path("/home/mbelleau/fq-segments/GLM-5.2-EXL3-FQ")
    EVID = RUN / "evidence"
    POLICY = RUN / "policy-k3-uniform.json"
elif MODE == "mixed":
    OUT = Path("/home/mbelleau/glm52-mixed-k3k5")
    SEG = Path("/home/mbelleau/fq-segments-mixed-k3k5")
    EVID = RUN / "evidence-mixed"
    POLICY = RUN / "policy-mixed-k3k5.json"
else:
    raise SystemExit("mode must be k3 or mixed")

policy = json.loads(POLICY.read_text())
bpe = policy["bits_per_expert"]
ks = sorted({int(k) for bits in bpe.values() for k in bits})
is_mixed = len(ks) > 1
report = {"mode": MODE, "out": str(OUT), "k_values": ks}

# ---------------------------------------------------------------- 1. metadata
installed = []
for f in sorted((EVID / "last-meta").iterdir()):
    if f.name in ("MANIFEST.sha256", FA.ASSEMBLY_RECORD, "model.safetensors.index.json"):
        continue  # regenerated below from the real bytes
    if f.name.endswith(".safetensors"):
        continue  # already materialized (as reflink clones) by the driver
    if f.is_file():
        shutil.copyfile(f, OUT / f.name)
        installed.append(f.name)
report["metadata_installed"] = installed

# ------------------------------------------- 2. index + manifest from bytes
FA.regenerate_shard_index(OUT)
shas = FA.hash_tree(OUT, skip=("MANIFEST.sha256", FA.ASSEMBLY_RECORD))
FA.regenerate_manifest(OUT, known=shas)

# ------------------------------------------------- 3. merged assembly record
fragments, frag_bad = [], []
for att in sorted((SEG / "attestations").glob("*.jsonl")):
    env = json.loads(att.read_text().splitlines()[0])
    try:
        payload = FT.verify_signature(env, SIGNER, where=att.name)
    except FT.TrustError as e:
        frag_bad.append(f"{att.name}: {e}")
        continue
    fragments.append({"file": payload["fragment"]["file"],
                      "sha256": payload["fragment"]["sha256"],
                      "size": payload["fragment"]["size"],
                      "predicate": payload["predicate"], "keyid": env["keyid"],
                      "materials": payload["materials"]})
report["fragments_signature_verified"] = len(fragments)
report["fragments_signature_failed"] = frag_bad

batches, products, divergent = [], [], []
for b in sorted(EVID.glob("batch-*.json")):
    d = json.loads(b.read_text())
    batches.append({"range": d["range"], "shards": len(d["shards"]),
                    "segments_verified": d["segments_verified"], "mode": d["mode"]})
    for s in d["shards"]:
        products.append(s)
        same = s.get("identical_to_source", s.get("bit_exact"))
        if not same:
            divergent.append({k: s[k] for k in
                              ("shard", "assembled_sha256", "source_sha256")})
report["batches"] = batches
report["shards_checked"] = len(products)
report["shards_differing_from_source"] = divergent
report["shards_identical_to_source"] = len(products) - len(divergent)

record = {
    "schema": FA.ATTESTATION_SCHEMA, "predicate": "assembly-of",
    "created_utc": FA.now_utc(),
    "tool": {"name": "fq_assemble", "version": FA.TOOL_VERSION,
             "driver": f"runs/m5-serve/assemble-{'mixed' if is_mixed else 'full'}.sh"},
    "recipe": {"file": POLICY.name, "sha256": FA.sha256_file(POLICY),
               "schema": policy.get("schema"), "k_values": ks,
               "layers": sorted(int(L) for L in bpe),
               "layers_assembled": sorted(int(L) for L in bpe)},
    "materials": {"segments": {"dir": SEG.name, "fragments": fragments},
                  "source": {"dir": SRC.name,
                             "manifest_sha256": FA.sha256_file(SRC / "MANIFEST.sha256")}},
    "products": [{"file": p["shard"], "sha256": p["assembled_sha256"]}
                 for p in sorted(products, key=lambda x: x["shard"])],
    "verification": {
        "mode": "verified", "trusted_signers": [SIGNER],
        "allowed_predicates": sorted(FA.DEFAULT_ALLOWED_PREDICATES),
        "segments_verified": sum(b["segments_verified"] for b in batches),
        "shards_identical_to_source": len(products) - len(divergent),
        "shards_differing_from_source": len(divergent),
        "note": ("Shards were assembled from segments by fq_assemble and hashed "
                 "by the tool from the bytes it wrote. A shard whose hash equals "
                 "the source shard is stored as a whole-file XFS reflink clone of "
                 "the byte-identical source blob; a shard that differs is the "
                 "assembler's own output, renamed into place. See "
                 "runs/m5-serve/assembly-report.md."),
    },
}
FA.atomic_write_json(OUT / FA.ASSEMBLY_RECORD, record)

# ------------------------------------------------------- 4. structural checks
TRELLIS = re.compile(r"\.experts\.(\d+)\.(\w+)\.rank(\d+)\.trellis$")


def survey(d: Path):
    tensors, params, tbytes = {}, 0, 0
    per_rank = {0: 0, 1: 0, 2: 0, 3: 0}
    shared = 0
    for shard in sorted(d.glob("model-*.safetensors")):
        hdr, _ = FA.read_header(shard)
        hdr.pop("__metadata__", None)
        for name, t in hdr.items():
            n = 1
            for x in t["shape"]:
                n *= x
            nbytes = t["data_offsets"][1] - t["data_offsets"][0]
            tensors[name] = shard.name
            params += n
            tbytes += nbytes
            m = re.search(r"\.rank(\d)\.", name)
            if m:
                per_rank[int(m.group(1))] += nbytes
            else:
                shared += nbytes
    return tensors, params, tbytes, per_rank, shared


a_t, a_p, a_b, a_rank, a_shared = survey(OUT)
s_t, s_p, s_b, s_rank, s_shared = survey(SRC)
report["tensor_count"] = {"assembled": len(a_t), "source": len(s_t),
                          "equal": len(a_t) == len(s_t)}
report["tensor_names_equal"] = (set(a_t) == set(s_t))
report["tensor_placement_equal"] = (a_t == s_t)
report["param_count"] = {"assembled": a_p, "source": s_p, "equal": a_p == s_p,
                         "delta": a_p - s_p}
report["tensor_bytes"] = {"assembled": a_b, "source": s_b, "delta": a_b - s_b}

# TP4 footprint.  Expert tensors carry an explicit .rankN. tag (the EXL3
# rank-sliced layout), so each rank loads exactly its own quarter of them.
# Everything else -- attention, shared experts, router, norms, embeddings,
# lm_head -- is sharded by vLLM's ordinary TP logic at load time, so it
# divides ~4 ways too; a little (norms, small replicated vectors) is genuinely
# replicated, which this model does not try to count.  Reporting the expert
# slice exactly and the remainder as /4 is the honest approximation and it
# agrees with the even split because the expert bytes divide exactly.
one_rank_experts = a_rank[0]
assert len(set(a_rank.values())) == 1, f"rank slices are not equal: {a_rank}"
per_rank = one_rank_experts + a_shared / 4
report["tp4_memory"] = {
    "expert_bytes_per_rank": one_rank_experts,
    "expert_bytes_all_ranks": sum(a_rank.values()),
    "non_expert_bytes_total": a_shared,
    "per_rank_bytes": int(per_rank),
    "per_rank_GiB": round(per_rank / GiB, 2),
    "pct_of_95.6GiB": round(per_rank / GiB / CARD_GiB * 100, 1),
    "model": ("expert bytes exact per rank + non-expert/4; excludes KV cache, "
              "activations, CUDA graphs and the small genuinely-replicated "
              "tensors (norms)"),
}

idx = json.loads((OUT / "model.safetensors.index.json").read_text())
wm = idx["weight_map"]
present = {p.name for p in OUT.glob("model-*.safetensors")}
dangling = sorted(set(wm.values()) - present)
missing = sorted(set(a_t) - set(wm))
report["index"] = {"entries": len(wm), "dangling_shard_refs": dangling,
                   "tensors_missing_from_index": missing,
                   "total_size": idx["metadata"]["total_size"],
                   "total_size_matches_headers": idx["metadata"]["total_size"] == a_b,
                   "self_consistent": not dangling and not missing and len(wm) == len(a_t)}

expected = ({f"model-layer-{i:03d}.safetensors" for i in range(0, 79)}
            | {"model-embed.safetensors", "model-head.safetensors"})
report["file_set"] = {"shards_present": len(present), "shards_expected": len(expected),
                      "missing": sorted(expected - present),
                      "unexpected": sorted(present - expected)}

cfg = json.loads((OUT / "config.json").read_text())
tail = cfg.get("hybrid_tr3_tail", {})
qc = cfg.get("quantization_config", {})
report["config"] = {
    "hybrid_tr3_tail.bits": tail.get("bits"),
    "hybrid_tr3_tail.k_values": tail.get("k_values"),
    "hybrid_tr3_tail.bits_per_expert": tail.get("bits_per_expert"),
    "quantization_config.quant_method": qc.get("quant_method"),
    "quantization_config.bits": qc.get("bits"),
    "quantization_config.codebook": qc.get("codebook"),
    "stale_modelopt_fields": sorted(k for k in qc
                                    if k in ("quant_algo", "config_groups", "producer")),
}
# the mixed contract, checked explicitly rather than assumed
if is_mixed:
    ref = tail.get("bits_per_expert")
    ok_ref = isinstance(ref, str) and ":" in ref
    bm_name, field = (ref.split(":", 1) if ok_ref else ("tier_bitmap.json", "bits_per_expert"))
    bm = json.loads((OUT / bm_name).read_text()) if (OUT / bm_name).exists() else {}
    bad_layers = []
    for L, bits in bpe.items():
        got = bm.get(L, {}).get(field)
        if got != [int(b) for b in bits]:
            bad_layers.append(L)
    report["mixed_contract"] = {
        "bits_is_mixed": tail.get("bits") == "mixed",
        "k_values": tail.get("k_values") == ks,
        "bits_per_expert_is_file_ref": ok_ref,
        "referenced_file": bm_name,
        "referenced_file_exists": (OUT / bm_name).exists(),
        "layers_with_wrong_bitmap": bad_layers,
        "quantization_config_bits_mixed": qc.get("bits") == "mixed",
        "quant_method_exl3": qc.get("quant_method") == "exl3",
        "all_ok": (tail.get("bits") == "mixed" and tail.get("k_values") == ks
                   and ok_ref and (OUT / bm_name).exists() and not bad_layers
                   and qc.get("quant_method") == "exl3"),
    }
    k5_per_layer = {L: bits.count(5) for L, bits in bpe.items() if 5 in bits}
    report["k5_allocation"] = {"covered_layers": sorted(int(L) for L in k5_per_layer),
                               "experts_per_layer": sorted(set(k5_per_layer.values())),
                               "total_k5_experts": sum(k5_per_layer.values())}
else:
    report["mixed_contract"] = {"uniform": True,
                                "bits_is_float_K": tail.get("bits") == float(ks[0]),
                                "quant_method_exl3": qc.get("quant_method") == "exl3"}

boot = ["config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
        "chat_template.jinja", "tier_bitmap.json", "model.safetensors.index.json",
        "MANIFEST.sha256"]
report["boot_files_present"] = {f: (OUT / f).exists() for f in boot}

(EVID / "finalize-report.json").write_text(json.dumps(report, indent=1) + "\n")
print(json.dumps(report, indent=1))
