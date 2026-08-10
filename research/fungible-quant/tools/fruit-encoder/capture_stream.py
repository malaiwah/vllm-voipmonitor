#!/usr/bin/env python3
"""Layer-streaming BF16 activation capture for GLM-5.2-family checkpoints.

Replaces whole-model residency (capture_hf.py) with layer-major streaming so the
full 78-layer GLM-5.2 fits one 96GB GPU: tokenize+embed all planned samples into a
boundary activation file, then for each transformer layer L load ONLY layer L's
weights, stream the previous boundary through it in packed batches, and write the
next boundary.  At most two boundaries live at once.

On-disk capture ABI is byte-compatible with capture_fruit.py / capture_hf.py so the
downstream LayerCalib encoder consumes it unchanged:

  <capture_dir>/layer_LLL/x.bin              bf16 [tokens, hidden] raw bytes
  <capture_dir>/layer_LLL/ids.bin            uint8 [tokens, 8] natural top-8 ids
  <capture_dir>/layer_LLL/layer_manifest.json  schema glm52-b300-layer-capture-v1
  <capture_dir>/capture_run_manifest.json      schema glm52-b300-capture-run-v1

Capture point and routing-id math are capture_hf.py's, verbatim: a
forward_pre_hook on layers[L].mlp.experts sees the post-attention-layernorm tensor
the router gate consumes, ids are recomputed as
  topk(sigmoid(F.linear(x.float32, gate_w.float32)) + e_score_correction_bias, 8)
and cross-checked against the model's own topk indices on every token.

Forward fidelity: the transformers GlmMoeDsa* modules are used for every op.  The
model skeleton is instantiated on the meta device; one decoder layer at a time is
materialized on GPU from the safetensors shards (per-expert gate/up/down weights
fused into gate_up_proj/down_proj exactly like transformers' qwen2_moe weight
converter; e_score_correction_bias upcast to fp32 per _keep_in_fp32_modules_strict;
everything else bf16 as from_pretrained(dtype=bf16) would produce).  The decoder
layer forward is decomposed into its own submodule calls (input_layernorm,
self_attn, post_attention_layernorm, mlp with the residual adds mirrored verbatim
from GlmMoeDsaDecoderLayer.forward) so that the MoE block can be invoked once per
group of packs, amortizing grouped_mm's full expert-weight read.

DSA cross-layer top-k sharing: GLM-5.2's indexer_types mixes "full" and "shared"
layers; shared layers reuse the previous full layer's top-k indices.  When layer L
is "full" and layer L+1 is "shared", the per-pack topk_indices returned by the
attention module are persisted to a topk store in the work dir and fed as
prev_topk_indices to the dependent shared layers.  Stores are retained across the
pass boundary when a later-pass layer still depends on them (layer 41 consumes
layer 38's store).

Batching: samples are packed in exact plan order into token-budgeted batches
([1, T] with per-sample position ids and an explicit block-diagonal additive
mask), so attention never crosses sample boundaries; single-sample packs go
through create_causal_mask exactly like the reference full-model forward.
Capture rows land in exact plan order.

Sharding: --shard-count contiguous corpus shards (split at pack boundaries) run
on separate GPUs with private boundary/topk files, pwriting their token ranges of
the shared x.bin/ids.bin.  --seal merges shard markers, hashes payloads, and
writes the ABI manifests.
"""

from __future__ import annotations

import argparse
from collections import Counter
import faulthandler
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import signal
import struct
import sys
import time
import uuid

try:
    faulthandler.register(signal.SIGUSR2, all_threads=True)
except (AttributeError, RuntimeError):
    pass

# ---- calibration-plan constants (verbatim from capture_fruit.py) --------------
CORPUS_SHA256 = "cf247acc7c5da9f0600c7d6ab3b7c2fcfc54ec30b794e3b6047559285fa44df4"
CORPUS_ROWS = 12_228
AXES = (
    "axis1_general",
    "axis2_legal",
    "axis3_code_agentic",
    "axis4_reasoning_termination",
)
AXIS_ROWS = 3_057
SEED = 20260711
TARGET_TOKENS = 1_048_576
MAX_SAMPLE_TOKENS = 4_096
MIN_SAMPLE_TOKENS = 8

# ---- capture ABI constants ----------------------------------------------------
NUM_EXPERTS = 256
TOPK = 8
CAPTURE_TP = 1
OUTPUT_TP = 4

ATTN_IMPL = "sdpa"
EXPERTS_IMPL = "grouped_mm"


def log(message: str, logfile: str | None = None) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} | {message}"
    print(line, flush=True)
    if logfile:
        with open(logfile, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def sha256_file(path: str | Path, chunk: int = 64 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp, path)


def canonical_hash(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _tokenizer_identity(src: Path, tokenizer: object) -> dict:
    files = {}
    for name in (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "chat_template.jinja",
    ):
        path = src / name
        if path.is_file():
            files[name] = sha256_file(path)
    return {
        "class": type(tokenizer).__name__,
        "vocab_size": int(getattr(tokenizer, "vocab_size", -1)),
        "files_sha256": files,
    }


# ---- model geometry -----------------------------------------------------------


class Geometry:
    """Layer/MoE/indexer facts derived from the target checkpoint's config."""

    def __init__(self, config):
        self.hidden = int(config.hidden_size)
        self.num_layers = int(config.num_hidden_layers)
        self.mlp_layer_types = list(config.mlp_layer_types)
        self.indexer_types = list(config.indexer_types)[: self.num_layers]
        self.moe_layers = [
            i for i, t in enumerate(self.mlp_layer_types[: self.num_layers]) if t == "sparse"
        ]
        routing = {
            "n_group": getattr(config, "n_group", None),
            "topk_group": getattr(config, "topk_group", None),
            "scoring_func": getattr(config, "scoring_func", None),
            "top_k": getattr(config, "num_experts_per_tok", None),
        }
        expected = {"n_group": 1, "topk_group": 1, "scoring_func": "sigmoid", "top_k": TOPK}
        if routing != expected:
            raise RuntimeError(f"routing config {routing} != {expected}")
        if int(config.n_routed_experts) != NUM_EXPERTS:
            raise RuntimeError(f"n_routed_experts {config.n_routed_experts} != {NUM_EXPERTS}")

    def indexer_source(self, layer: int) -> int:
        """For a 'shared' layer, the preceding 'full' layer whose topk it reuses."""
        if self.indexer_types[layer] != "shared":
            return layer
        probe = layer - 1
        while probe >= 0 and self.indexer_types[probe] != "full":
            probe -= 1
        if probe < 0:
            raise RuntimeError(f"layer {layer}: no preceding full indexer layer")
        return probe

    def store_needed(self, layer: int) -> bool:
        """Whether layer's topk indices must be persisted for later shared layers."""
        return (
            self.indexer_types[layer] == "full"
            and layer + 1 < self.num_layers
            and self.indexer_types[layer + 1] == "shared"
        )

    def store_consumers(self, layer: int) -> list[int]:
        out = []
        probe = layer + 1
        while probe < self.num_layers and self.indexer_types[probe] == "shared":
            out.append(probe)
            probe += 1
        return out


# ---- capture plan (verbatim selection logic from capture_fruit.py) ------------


def _waterfill_quotas(capacities: dict[str, int], target: int) -> dict[str, int]:
    if sum(capacities.values()) < target:
        raise RuntimeError(
            f"owner corpus has only {sum(capacities.values())} eligible truncated tokens, "
            f"below target {target}"
        )
    quotas = {axis: 0 for axis in capacities}
    remaining_axes = list(capacities)
    remaining = target
    while remaining_axes:
        share = math.ceil(remaining / len(remaining_axes))
        short = [axis for axis in remaining_axes if capacities[axis] < share]
        if not short:
            base, extra = divmod(remaining, len(remaining_axes))
            for index, axis in enumerate(remaining_axes):
                quotas[axis] = base + int(index < extra)
            remaining = 0
            break
        for axis in short:
            quotas[axis] = capacities[axis]
            remaining -= quotas[axis]
            remaining_axes.remove(axis)
    if remaining != 0 or sum(quotas.values()) != target:
        raise AssertionError((remaining, quotas, target))
    return quotas


def build_capture_plan(src: Path, corpus: Path, target_tokens: int, geom: Geometry, logfile: str) -> dict:
    """capture_fruit.build_capture_plan with HIDDEN / MoE-layer count taken from the
    target checkpoint's config instead of the Fruit proxy constants."""
    from transformers import AutoTokenizer

    corpus_digest = sha256_file(corpus)
    if corpus_digest != CORPUS_SHA256:
        raise RuntimeError(f"owner corpus sha256 mismatch: {corpus_digest}")

    records: list[dict] = []
    counts: Counter[str] = Counter()
    with corpus.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle):
            if not line.strip():
                continue
            record = json.loads(line)
            axis = record.get("axis")
            text = record.get("text")
            if axis not in AXES or not isinstance(text, str):
                raise RuntimeError(f"corpus line {line_no + 1}: invalid axis/text")
            records.append({"line": line_no, "axis": axis, "text": text})
            counts[axis] += 1
    if len(records) != CORPUS_ROWS or counts != Counter({axis: AXIS_ROWS for axis in AXES}):
        raise RuntimeError(f"owner corpus row/axis mismatch: rows={len(records)} axes={dict(counts)}")

    tokenizer = AutoTokenizer.from_pretrained(str(src), trust_remote_code=False)
    by_axis: dict[str, list[tuple[int, int]]] = {axis: [] for axis in AXES}
    skipped_short = 0
    t0 = time.time()
    for record in records:
        length = min(len(tokenizer.encode(record["text"])), MAX_SAMPLE_TOKENS)
        if length < MIN_SAMPLE_TOKENS:
            skipped_short += 1
            continue
        by_axis[record["axis"]].append((record["line"], length))
    capacities = {axis: sum(length for _, length in rows) for axis, rows in by_axis.items()}
    quotas = _waterfill_quotas(capacities, target_tokens)

    passes = []
    selected_lines: set[int] = set()
    total_tokens = 0
    for axis_index, axis in enumerate(AXES):
        candidates = list(by_axis[axis])
        random.Random(SEED + 1_000_003 * axis_index).shuffle(candidates)
        chosen = []
        pass_tokens = 0
        for line_no, length in candidates:
            if pass_tokens >= quotas[axis]:
                break
            if line_no in selected_lines:
                raise AssertionError(f"line selected twice: {line_no}")
            selected_lines.add(line_no)
            chosen.append({"line": line_no, "ntok": length})
            pass_tokens += length
        if pass_tokens < quotas[axis]:
            raise RuntimeError(f"axis {axis} exhausted at {pass_tokens} < quota {quotas[axis]}")
        passes.append(
            {
                "name": axis,
                "axis": axis,
                "quota_tokens": quotas[axis],
                "tokens": pass_tokens,
                "samples": chosen,
            }
        )
        total_tokens += pass_tokens

    source_identity = {
        "config_sha256": sha256_file(src / "config.json"),
        "index_sha256": sha256_file(src / "model.safetensors.index.json"),
    }
    plan = {
        "schema": "glm52-b300-capture-plan-v1",
        "selection_policy": "owner-corpus-axis-separated-luke-multipass-no-repeat-v1",
        "selection_note": (
            "Only the owner-pinned 12,228-row corpus is eligible. One natural-routing pass "
            "is sealed per owner axis; rows never repeat and no stock dataset is allowed."
        ),
        "owner_corpus_only": True,
        "calibration_baseline": True,
        "corpus_sha256": corpus_digest,
        "corpus_rows": len(records),
        "axis_rows": dict(sorted(counts.items())),
        "seed": SEED,
        "target_tokens": target_tokens,
        "max_sample_tokens": MAX_SAMPLE_TOKENS,
        "min_sample_tokens": MIN_SAMPLE_TOKENS,
        "skipped_short": skipped_short,
        "passes": passes,
        "total_tokens": total_tokens,
        "capture_bytes": total_tokens * len(geom.moe_layers) * (geom.hidden * 2 + TOPK),
        "source": source_identity,
        "tokenizer": _tokenizer_identity(src, tokenizer),
        "capture_tp": CAPTURE_TP,
        "output_tp": OUTPUT_TP,
        "routing": {
            "natural": True,
            "forced_expert_activation": False,
            "scoring_func": "sigmoid",
            "top_k": TOPK,
            "n_group": 1,
            "topk_group": 1,
        },
    }
    plan["capture_fingerprint"] = canonical_hash(plan)
    log(
        f"capture plan: {len(passes)} axis passes, {sum(len(p['samples']) for p in passes)} "
        f"samples, {total_tokens} tokens (global target {target_tokens}); "
        f"capacities={capacities}, quotas={quotas}, tokenize={time.time()-t0:.1f}s",
        logfile,
    )
    return plan


def validate_plan(plan: dict, src: Path, corpus: Path) -> None:
    """Ported verbatim from capture_hf.validate_plan."""
    if plan.get("schema") != "glm52-b300-capture-plan-v1":
        raise RuntimeError(f"unexpected capture plan schema: {plan.get('schema')!r}")
    claimed = plan.get("capture_fingerprint")
    canonical = dict(plan)
    canonical.pop("capture_fingerprint", None)
    actual = canonical_hash(canonical)
    if claimed != actual:
        raise RuntimeError(f"capture plan fingerprint mismatch: {claimed} != {actual}")
    if sha256_file(corpus) != plan.get("corpus_sha256") or claimed is None:
        raise RuntimeError("capture plan does not match the supplied owner corpus")
    if plan.get("corpus_sha256") != CORPUS_SHA256:
        raise RuntimeError("owner corpus sha256 mismatch vs pinned constant")
    current_source = {
        "config_sha256": sha256_file(src / "config.json"),
        "index_sha256": sha256_file(src / "model.safetensors.index.json"),
    }
    if current_source != plan.get("source"):
        raise RuntimeError(f"capture plan source mismatch: {current_source} != {plan.get('source')}")
    if int(plan.get("capture_tp", -1)) != CAPTURE_TP or int(plan.get("output_tp", -1)) != OUTPUT_TP:
        raise RuntimeError("capture/output TP in plan does not match TP1 capture / TP4 format")
    if plan.get("selection_policy") != "owner-corpus-axis-separated-luke-multipass-no-repeat-v1":
        raise RuntimeError("capture plan is not the owner-corpus Luke-style baseline")
    if plan.get("owner_corpus_only") is not True:
        raise RuntimeError("capture plan does not declare owner-corpus-only calibration")
    if plan.get("calibration_baseline") is not True:
        raise RuntimeError("capture plan does not declare mandatory baseline calibration")
    routing = plan.get("routing", {})
    expected_routing = {
        "natural": True,
        "forced_expert_activation": False,
        "scoring_func": "sigmoid",
        "top_k": TOPK,
        "n_group": 1,
        "topk_group": 1,
    }
    if routing != expected_routing:
        raise RuntimeError(f"capture plan routing {routing} != mandatory {expected_routing}")
    passes = plan.get("passes", [])
    if not passes:
        raise RuntimeError("capture plan contains no passes")
    if [item.get("axis") for item in passes] != list(AXES):
        raise RuntimeError("Luke-style baseline must contain one ordered pass per owner axis")
    lines = [sample["line"] for item in passes for sample in item["samples"]]
    if len(lines) != len(set(lines)):
        raise RuntimeError("capture plan repeats corpus rows")
    if sum(int(item["tokens"]) for item in passes) != int(plan["total_tokens"]):
        raise RuntimeError("capture plan per-pass token sum mismatch")


def load_plan_tokens(plan: dict, src: Path, corpus: Path, logfile: str) -> list[list[int]]:
    """capture_hf.load_plan_tokens flattened to one ordered sample list."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(src), trust_remote_code=False)
    identity = _tokenizer_identity(src, tokenizer)
    if identity != plan.get("tokenizer"):
        raise RuntimeError(
            f"tokenizer identity changed since capture plan was built: "
            f"{identity} != {plan.get('tokenizer')}"
        )
    raw_lines = corpus.read_text(encoding="utf-8").splitlines()
    out: list[list[int]] = []
    for pass_info in plan["passes"]:
        token_lists = []
        for sample in pass_info["samples"]:
            line_no = int(sample["line"])
            record = json.loads(raw_lines[line_no])
            ids = tokenizer.encode(record["text"])[:MAX_SAMPLE_TOKENS]
            if len(ids) != int(sample["ntok"]):
                raise RuntimeError(
                    f"tokenization drift at corpus line {line_no}: {len(ids)} != {sample['ntok']}"
                )
            token_lists.append(ids)
        tokens = sum(map(len, token_lists))
        if tokens != int(pass_info["tokens"]):
            raise RuntimeError(f"pass {pass_info['name']}: token sum drift {tokens}")
        log(f"prepared pass {pass_info['name']}: {len(token_lists)} samples, {tokens} tokens", logfile)
        out.extend(token_lists)
    return out


# ---- packing / sharding -------------------------------------------------------


class Pack:
    __slots__ = ("index", "sample_lens", "tokens", "token_offset")

    def __init__(self, index: int, sample_lens: list[int], token_offset: int):
        self.index = index
        self.sample_lens = sample_lens
        self.tokens = sum(sample_lens)
        self.token_offset = token_offset


def build_packs(sample_lens: list[int], budget: int) -> list[Pack]:
    packs: list[Pack] = []
    cur: list[int] = []
    offset = 0
    cur_offset = 0
    for n in sample_lens:
        if cur and sum(cur) + n > budget:
            packs.append(Pack(len(packs), cur, cur_offset))
            cur = []
            cur_offset = offset
        cur.append(n)
        offset += n
    if cur:
        packs.append(Pack(len(packs), cur, cur_offset))
    return packs


def shard_packs(
    packs: list[Pack], shard_count: int, overhead_tokens: int = 0
) -> list[tuple[int, int]]:
    """Contiguous [lo, hi) pack ranges with near-equal cost, where a pack's cost is
    tokens + overhead_tokens per sample (per-sample MoE calls pay a full
    expert-weight HBM read regardless of length, so short samples are not free)."""

    def cost(pack: Pack) -> int:
        return pack.tokens + overhead_tokens * len(pack.sample_lens)

    total = sum(cost(p) for p in packs)
    bounds = []
    lo = 0
    acc = 0
    for shard in range(shard_count):
        target = total * (shard + 1) / shard_count
        hi = lo
        while hi < len(packs) and (acc + cost(packs[hi]) <= target or hi == lo):
            acc += cost(packs[hi])
            hi += 1
        if shard == shard_count - 1:
            while hi < len(packs):
                acc += cost(packs[hi])
                hi += 1
        bounds.append((lo, hi))
        lo = hi
    if bounds[-1][1] != len(packs):
        raise AssertionError(bounds)
    return bounds


# ---- checkpoint access --------------------------------------------------------


class ShardIndex:
    def __init__(self, src: Path):
        self.src = src
        index = json.loads((src / "model.safetensors.index.json").read_text())
        self.weight_map: dict[str, str] = index["weight_map"]

    def load_prefix(self, prefix: str, device: str) -> dict:
        import torch
        from safetensors import safe_open

        keys = {k: v for k, v in self.weight_map.items() if k.startswith(prefix)}
        if not keys:
            raise RuntimeError(f"no checkpoint tensors under prefix {prefix!r}")
        by_shard: dict[str, list[str]] = {}
        for key, shard in keys.items():
            by_shard.setdefault(shard, []).append(key)
        out = {}
        for shard, shard_keys in sorted(by_shard.items()):
            with safe_open(str(self.src / shard), framework="pt", device=device) as handle:
                for key in shard_keys:
                    out[key[len(prefix):]] = handle.get_tensor(key)
        return out

    def load_tensor(self, name: str, device: str):
        from safetensors import safe_open

        shard = self.weight_map[name]
        with safe_open(str(self.src / shard), framework="pt", device=device) as handle:
            return handle.get_tensor(name)


def fuse_expert_weights(raw: dict, num_experts: int):
    """transformers qwen2_moe conversion: stack per-expert gate/up/down and concat
    gate|up along dim 1 -> mlp.experts.gate_up_proj [E, 2I, H], down_proj [E, H, I]."""
    import torch

    if "mlp.experts.0.gate_proj.weight" not in raw:
        return raw
    gates, ups, downs = [], [], []
    for expert in range(num_experts):
        gates.append(raw.pop(f"mlp.experts.{expert}.gate_proj.weight"))
        ups.append(raw.pop(f"mlp.experts.{expert}.up_proj.weight"))
        downs.append(raw.pop(f"mlp.experts.{expert}.down_proj.weight"))
    raw["mlp.experts.gate_up_proj"] = torch.cat([torch.stack(gates), torch.stack(ups)], dim=1)
    raw["mlp.experts.down_proj"] = torch.stack(downs)
    return raw


def materialize(module, raw: dict, expect_dtype) -> None:
    """Load a meta-device module's weights in place (assign=True), applying the
    from_pretrained dtype rules: e_score_correction_bias fp32 (strict), rest bf16."""
    import torch

    for name in list(raw):
        if name.endswith("e_score_correction_bias"):
            raw[name] = raw[name].to(torch.float32)
        elif raw[name].dtype != expect_dtype:
            raise RuntimeError(f"unexpected checkpoint dtype for {name}: {raw[name].dtype}")
    result = module.load_state_dict(raw, strict=True, assign=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(f"materialize mismatch: {result}")
    for param_name, param in module.named_parameters():
        if param.is_meta:
            raise RuntimeError(f"parameter still on meta after materialize: {param_name}")


def dematerialize(module) -> None:
    import torch

    module.to("meta")
    gc.collect()
    torch.cuda.empty_cache()


# ---- capture region writer ----------------------------------------------------


class RegionWriter:
    """One shard's token-range writer into the shared x.bin.partial/ids.bin.partial.

    Rows are pwritten at absolute offsets derived from the global plan token
    offset, so shard writes never overlap and rewrites (restart of an unsealed
    layer) are idempotent."""

    def __init__(self, layer_dir: Path, hidden: int, total_tokens: int, token_offset: int):
        layer_dir.mkdir(parents=True, exist_ok=True)
        self.hidden = hidden
        self.fd_x = os.open(layer_dir / "x.bin.partial", os.O_RDWR | os.O_CREAT, 0o644)
        self.fd_i = os.open(layer_dir / "ids.bin.partial", os.O_RDWR | os.O_CREAT, 0o644)
        for fd, row_bytes in ((self.fd_x, hidden * 2), (self.fd_i, TOPK)):
            want = total_tokens * row_bytes
            if os.fstat(fd).st_size < want:
                os.ftruncate(fd, want)
        self.token_offset = token_offset
        self.cursor = token_offset
        self.hx = hashlib.sha256()
        self.hi = hashlib.sha256()
        self.tokens = 0
        self.routed = [0] * NUM_EXPERTS

    def write(self, x_bytes: bytes, id_bytes: bytes, rows: int, bincount) -> None:
        os.pwrite(self.fd_x, x_bytes, self.cursor * self.hidden * 2)
        os.pwrite(self.fd_i, id_bytes, self.cursor * TOPK)
        self.hx.update(x_bytes)
        self.hi.update(id_bytes)
        self.cursor += rows
        self.tokens += rows
        for expert, value in enumerate(bincount):
            self.routed[expert] += int(value)

    def close(self) -> None:
        os.close(self.fd_x)
        os.close(self.fd_i)


def make_capture_hook(layer: int, gate_w, gate_b, writer: RegionWriter, verify: dict, hidden: int):
    """capture_hf.py's mlp.experts forward_pre_hook, verbatim math."""
    import torch
    import torch.nn.functional as torch_f

    def hook(module, args, kwargs):
        hidden_states = kwargs.get("hidden_states")
        if hidden_states is None and args:
            hidden_states = args[0]
        if hidden_states is None or hidden_states.dim() != 2 or hidden_states.shape[1] != hidden:
            raise RuntimeError(
                f"layer {layer}: unexpected MoE input "
                f"{None if hidden_states is None else tuple(hidden_states.shape)}"
            )
        rows = int(hidden_states.shape[0])
        logits = torch_f.linear(hidden_states.to(torch.float32), gate_w)
        scores = torch.sigmoid(logits) + gate_b
        ids = torch.topk(scores, TOPK, dim=-1, sorted=False).indices

        model_ids = kwargs.get("top_k_index")
        if model_ids is None and len(args) > 1:
            model_ids = args[1]
        if model_ids is not None and model_ids.dim() == 2 and tuple(model_ids.shape) == (rows, TOPK):
            bad = (ids.sort(dim=-1).values != model_ids.long().sort(dim=-1).values).any(dim=-1)
            verify["checked_tokens"] += rows
            verify["mismatch_tokens"] += int(bad.sum().item())
            verify["kwarg_logits_layers"] += 1

        ids_cpu = ids.to(torch.uint8).cpu().contiguous()
        hidden_cpu = hidden_states.detach().to(torch.bfloat16).cpu().contiguous()
        x_bytes = hidden_cpu.view(torch.int16).numpy().tobytes()
        id_bytes = ids_cpu.numpy().tobytes()
        bincount = torch.bincount(ids.flatten().to(torch.int64), minlength=NUM_EXPERTS).cpu().tolist()
        writer.write(x_bytes, id_bytes, rows, bincount)
        return None

    return hook


# ---- topk store ---------------------------------------------------------------


class TopkStoreWriter:
    def __init__(self, path: Path):
        self.path = path
        self.handle = open(str(path) + ".partial", "wb", buffering=8 << 20)
        self.records: dict[str, list[int]] = {}
        self.offset = 0

    def write(self, pack_index: int, topk) -> None:
        arr = topk.squeeze(0).to("cpu").numpy()  # [T, K] int32
        data = arr.tobytes()
        self.records[str(pack_index)] = [self.offset, int(arr.shape[0]), int(arr.shape[1])]
        self.handle.write(data)
        self.offset += len(data)

    def seal(self) -> None:
        self.handle.flush()
        self.handle.close()
        os.replace(str(self.path) + ".partial", self.path)
        atomic_json(Path(str(self.path) + ".json"), {"records": self.records, "bytes": self.offset})


class TopkStoreReader:
    def __init__(self, path: Path):
        meta = json.loads(Path(str(path) + ".json").read_text())
        self.records = meta["records"]
        self.handle = open(path, "rb")

    def read(self, pack_index: int, device: str):
        import numpy as np
        import torch

        offset, rows, k = self.records[str(pack_index)]
        self.handle.seek(offset)
        arr = np.frombuffer(self.handle.read(rows * k * 4), dtype=np.int32).reshape(rows, k)
        return torch.from_numpy(arr.copy()).unsqueeze(0).to(device)

    def close(self) -> None:
        self.handle.close()


# ---- streaming pipeline -------------------------------------------------------


class StreamRunner:
    def __init__(
        self,
        src: Path,
        samples: list[list[int]],
        work_dir: Path,
        capture_dir: Path,
        capture_layers: list[int],
        stop_after_layer: int,
        pack_tokens: int,
        moe_group: int,
        shard_index: int,
        shard_count: int,
        state_root: Path | None,
        logfile: str | None,
        capture_fingerprint: str = "selftest",
        shard_cost_overhead: int = 0,
    ):
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM

        self.src = src
        self.samples = samples
        self.sample_lens = [len(s) for s in samples]
        self.total_tokens = sum(self.sample_lens)
        self.capture_dir = capture_dir
        self.capture_layers = sorted(capture_layers)
        self.stop_after_layer = stop_after_layer
        self.moe_group = max(1, moe_group)
        self.shard_index = shard_index
        self.shard_count = shard_count
        self.state_root = state_root
        self.logfile = logfile
        self.fingerprint = capture_fingerprint
        self.device = "cuda:0"

        config = AutoConfig.from_pretrained(str(src))
        self.geom = Geometry(config)
        if stop_after_layer >= self.geom.num_layers:
            raise RuntimeError(f"stop_after_layer {stop_after_layer} >= {self.geom.num_layers}")
        bad = [l for l in self.capture_layers if l not in self.geom.moe_layers or l > stop_after_layer]
        if bad:
            raise RuntimeError(f"capture layers {bad} not MoE layers <= stop layer")

        with torch.device("meta"):
            self.model = AutoModelForCausalLM.from_config(
                config, attn_implementation=ATTN_IMPL, experts_implementation=EXPERTS_IMPL
            )
        self.model.eval()
        self.model.requires_grad_(False)
        self.config = self.model.config
        from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import GlmMoeDsaRotaryEmbedding

        # inv_freq MUST be computed on CPU and moved (as from_pretrained does):
        # CUDA pow() differs from CPU by ulps on some exponents, which perturbs
        # cos/sin enough to flip near-tie routings downstream.
        self.rotary = GlmMoeDsaRotaryEmbedding(config=self.config).to(self.device)

        self.ckpt = ShardIndex(src)

        packs = build_packs(self.sample_lens, pack_tokens)
        bounds = shard_packs(packs, shard_count, shard_cost_overhead)
        lo, hi = bounds[shard_index]
        self.all_packs = packs
        self.packs = packs[lo:hi]
        self.pack_range = (lo, hi)
        self.shard_token_offset = packs[lo].token_offset if self.packs else self.total_tokens
        self.shard_tokens = sum(p.tokens for p in self.packs)
        self.shard_dir = work_dir / f"shard{shard_index}"
        self.shard_dir.mkdir(parents=True, exist_ok=True)
        self.verify_totals = {"checked_tokens": 0, "mismatch_tokens": 0, "kwarg_logits_layers": 0}
        self._state_t = 0.0
        self._run_t0 = time.time()
        self._tokens_done_layers = 0
        log(
            f"shard {shard_index}/{shard_count}: packs [{lo},{hi}) of {len(packs)}, "
            f"{self.shard_tokens} tokens at plan offset {self.shard_token_offset}; "
            f"pack_tokens={pack_tokens} moe_group={self.moe_group} "
            f"layers 0..{stop_after_layer} capture={self.capture_layers}",
            logfile,
        )

    # -- paths ------------------------------------------------------------------

    def boundary_path(self, layer: int) -> Path:
        return self.shard_dir / f"boundary_{layer:03d}.bin"

    def boundary_marker(self, layer: int) -> Path:
        return self.shard_dir / f"boundary_{layer:03d}.json"

    def boundary_state(self, layer: int) -> str:
        """'sealed' (file present), 'consumed' (legitimately deleted after use),
        or 'absent'."""
        marker = self.boundary_marker(layer)
        if not marker.is_file():
            return "absent"
        meta = json.loads(marker.read_text())
        if meta.get("tokens") != self.shard_tokens or meta.get("fingerprint") != self.fingerprint:
            return "absent"
        if meta.get("consumed"):
            return "consumed"
        path = self.boundary_path(layer)
        if path.is_file() and path.stat().st_size == self.shard_tokens * self.geom.hidden * 2:
            return "sealed"
        return "absent"

    def boundary_sealed(self, layer: int) -> bool:
        return self.boundary_state(layer) in ("sealed", "consumed")

    def store_path(self, layer: int) -> Path:
        return self.shard_dir / f"topk_{layer:03d}.bin"

    def store_state(self, layer: int) -> str:
        meta_path = Path(str(self.store_path(layer)) + ".json")
        if not meta_path.is_file():
            return "absent"
        meta = json.loads(meta_path.read_text())
        if meta.get("consumed"):
            return "consumed"
        if self.store_path(layer).is_file():
            return "sealed"
        return "absent"

    def store_sealed(self, layer: int) -> bool:
        return self.store_state(layer) in ("sealed", "consumed")

    def done_marker(self, layer: int) -> Path:
        return self.shard_dir / f"layer_{layer:03d}.done.json"

    def capture_done(self, layer: int) -> bool:
        marker = self.done_marker(layer)
        if not marker.is_file():
            return False
        meta = json.loads(marker.read_text())
        return meta.get("tokens") == self.shard_tokens and meta.get("fingerprint") == self.fingerprint

    def layer_complete(self, layer: int) -> bool:
        if not self.boundary_sealed(layer + 1):
            return False
        if layer in self.capture_layers and not self.capture_done(layer):
            return False
        if self.geom.store_needed(layer) and any(
            c <= self.geom.num_layers - 1 for c in self.geom.store_consumers(layer)
        ) and not self.store_sealed(layer):
            return False
        return True

    # -- state ------------------------------------------------------------------

    def write_state(self, stage: str, layer: int, packs_done: int, force: bool = False) -> None:
        now = time.time()
        if not force and now - self._state_t < 120:
            return
        self._state_t = now
        payload = {
            "shard": self.shard_index,
            "shard_count": self.shard_count,
            "stage": stage,
            "layer": layer,
            "capture_window": [self.capture_layers[0], self.capture_layers[-1]],
            "stop_after_layer": self.stop_after_layer,
            "boundary_preserved_layer": self.stop_after_layer + 1,
            "boundary_preserved_file": str(self.boundary_path(self.stop_after_layer + 1)),
            "packs_done": packs_done,
            "packs_total": len(self.packs),
            "shard_tokens": self.shard_tokens,
            "elapsed_s": round(now - self._run_t0, 1),
            "updated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "pid": os.getpid(),
        }
        atomic_json(self.shard_dir / "state.json", payload)
        if self.state_root is not None:
            merged = {"shards": {}, "updated": payload["updated"]}
            for shard in range(self.shard_count):
                path = self.shard_dir.parent / f"shard{shard}" / "state.json"
                if shard == self.shard_index:
                    merged["shards"][str(shard)] = payload
                elif path.is_file():
                    try:
                        merged["shards"][str(shard)] = json.loads(path.read_text())
                    except Exception:
                        pass
            atomic_json(self.state_root, merged)

    # -- pack tensors -----------------------------------------------------------

    def pack_position_ids(self, pack: Pack):
        import torch

        ids = torch.cat([torch.arange(n, dtype=torch.long) for n in pack.sample_lens])
        return ids.unsqueeze(0).to(self.device)

    def pack_mask(self, pack: Pack, hidden_states):
        """Block-diagonal causal additive mask [1,1,T,T]; None for single-sample
        packs (create_causal_mask path, mirroring the reference full-model forward)."""
        import torch
        from transformers.masking_utils import create_causal_mask

        position_ids = self.pack_position_ids(pack)
        if len(pack.sample_lens) == 1:
            mask = create_causal_mask(
                config=self.config,
                inputs_embeds=hidden_states,
                attention_mask=None,
                past_key_values=None,
                position_ids=position_ids,
            )
            return position_ids, mask
        total = pack.tokens
        seg = torch.cat(
            [torch.full((n,), i, dtype=torch.long) for i, n in enumerate(pack.sample_lens)]
        ).to(self.device)
        pos = torch.arange(total, device=self.device)
        allowed = (seg.unsqueeze(1) == seg.unsqueeze(0)) & (pos.unsqueeze(0) <= pos.unsqueeze(1))
        mask = torch.zeros((total, total), dtype=hidden_states.dtype, device=self.device)
        mask = mask.masked_fill(~allowed, torch.finfo(hidden_states.dtype).min)
        return position_ids, mask.unsqueeze(0).unsqueeze(0)

    # -- stages -----------------------------------------------------------------

    def run(self) -> dict:
        import torch

        with torch.inference_mode():
            if not self.boundary_sealed(0):
                self.run_embed()
            for layer in range(0, self.stop_after_layer + 1):
                if self.layer_complete(layer):
                    log(f"shard {self.shard_index}: layer {layer} already complete; skip", self.logfile)
                    self.maybe_delete_inputs(layer)
                    continue
                if self.boundary_state(layer) != "sealed":
                    raise RuntimeError(
                        f"shard {self.shard_index}: input boundary {layer} not physically present "
                        f"(state={self.boundary_state(layer)}); cannot resume"
                    )
                self.run_layer(layer)
                self.maybe_delete_inputs(layer)
        self.write_state("done", self.stop_after_layer, len(self.packs), force=True)
        marker = {
            "shard": self.shard_index,
            "fingerprint": self.fingerprint,
            "tokens": self.shard_tokens,
            "verify": self.verify_totals,
            "boundary_preserved": self.stop_after_layer + 1,
            "finished": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        atomic_json(self.shard_dir / "shard.done.json", marker)
        return marker

    def run_embed(self) -> None:
        import torch

        t0 = time.time()
        embed = self.model.model.embed_tokens
        raw = self.ckpt.load_prefix("model.embed_tokens.", self.device)
        materialize(embed, raw, torch.bfloat16)
        out_path = self.boundary_path(0)
        tmp = Path(str(out_path) + ".partial")
        sample_offsets = []
        acc = 0
        for n in self.sample_lens:
            sample_offsets.append(acc)
            acc += n
        # map plan-order sample index per pack
        pack_first_sample = []
        seen = 0
        for pack in self.all_packs:
            pack_first_sample.append(seen)
            seen += len(pack.sample_lens)
        with open(tmp, "wb", buffering=32 << 20) as handle:
            for done, pack in enumerate(self.packs):
                first = pack_first_sample[pack.index]
                ids = [t for i in range(len(pack.sample_lens)) for t in self.samples[first + i]]
                input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
                rows = embed(input_ids).squeeze(0).to(torch.bfloat16).cpu().contiguous()
                handle.write(rows.view(torch.int16).numpy().tobytes())
                self.write_state("embed", -1, done + 1)
        os.replace(tmp, out_path)
        atomic_json(
            self.boundary_marker(0),
            {"layer": 0, "tokens": self.shard_tokens, "fingerprint": self.fingerprint,
             "pack_range": list(self.pack_range)},
        )
        dematerialize(embed)
        log(f"shard {self.shard_index}: embed boundary sealed in {time.time()-t0:.1f}s", self.logfile)

    def run_layer(self, layer: int) -> None:
        import torch

        t0 = time.time()
        module = self.model.model.layers[layer]
        raw = self.ckpt.load_prefix(f"model.layers.{layer}.", self.device)
        raw = fuse_expert_weights(raw, NUM_EXPERTS)
        materialize(module, raw, torch.bfloat16)
        del raw
        t_load = time.time() - t0

        capture_writer = None
        hook_handle = None
        verify = {"checked_tokens": 0, "mismatch_tokens": 0, "kwarg_logits_layers": 0}
        if layer in self.capture_layers:
            gate = module.mlp.gate
            weight = gate.weight
            bias = getattr(gate, "e_score_correction_bias", None)
            if tuple(weight.shape) != (NUM_EXPERTS, self.geom.hidden) or bias is None:
                raise RuntimeError(f"layer {layer}: unexpected router weight/bias shape")
            gate_w = weight.detach().to(torch.float32)
            gate_b = bias.detach().to(torch.float32).flatten()
            if tuple(gate_b.shape) != (NUM_EXPERTS,):
                raise RuntimeError(f"layer {layer}: router correction bias shape mismatch")
            capture_writer = RegionWriter(
                self.capture_dir / f"layer_{layer:03d}",
                self.geom.hidden,
                self.total_tokens,
                self.shard_token_offset,
            )
            hook_handle = module.mlp.experts.register_forward_pre_hook(
                make_capture_hook(layer, gate_w, gate_b, capture_writer, verify, self.geom.hidden),
                with_kwargs=True,
            )

        store_writer = None
        if self.geom.store_needed(layer):
            store_writer = TopkStoreWriter(self.store_path(layer))
        store_reader = None
        if self.geom.indexer_types[layer] == "shared":
            source = self.geom.indexer_source(layer)
            if self.store_state(source) != "sealed":
                raise RuntimeError(
                    f"layer {layer}: topk store for source layer {source} not physically present "
                    f"(state={self.store_state(source)})"
                )
            store_reader = TopkStoreReader(self.store_path(source))

        in_path = self.boundary_path(layer)
        out_tmp = Path(str(self.boundary_path(layer + 1)) + ".partial")
        row_bytes = self.geom.hidden * 2
        packs_done = 0
        with open(in_path, "rb", buffering=32 << 20) as fin, open(out_tmp, "wb", buffering=32 << 20) as fout:
            for group_start in range(0, len(self.packs), self.moe_group):
                group = self.packs[group_start:group_start + self.moe_group]
                attn_states = []
                for pack in group:
                    data = fin.read(pack.tokens * row_bytes)
                    if len(data) != pack.tokens * row_bytes:
                        raise RuntimeError(f"layer {layer}: boundary short read at pack {pack.index}")
                    h_cpu = torch.frombuffer(bytearray(data), dtype=torch.int16)
                    h = h_cpu.view(torch.bfloat16).reshape(1, pack.tokens, self.geom.hidden).to(self.device)
                    position_ids, mask = self.pack_mask(pack, h)
                    position_embeddings = self.rotary(h, position_ids)
                    prev_topk = None
                    if store_reader is not None:
                        prev_topk = store_reader.read(pack.index, self.device)
                    # GlmMoeDsaDecoderLayer.forward, attention half (verbatim ops):
                    residual = h
                    normed = module.input_layernorm(h)
                    attn_out, _, topk_indices = module.self_attn(
                        hidden_states=normed,
                        position_embeddings=position_embeddings,
                        attention_mask=mask,
                        past_key_values=None,
                        position_ids=position_ids,
                        prev_topk_indices=prev_topk,
                    )
                    attn_states.append(residual + attn_out)
                    if store_writer is not None:
                        store_writer.write(pack.index, topk_indices)
                    del h, normed, attn_out, prev_topk
                # GlmMoeDsaDecoderLayer.forward, MLP half, on the pack group:
                grouped = torch.cat(attn_states, dim=1) if len(attn_states) > 1 else attn_states[0]
                del attn_states
                residual = grouped
                normed = module.post_attention_layernorm(grouped)
                mlp_out = module.mlp(normed)
                out = residual + mlp_out
                del grouped, normed, mlp_out
                rows = out.squeeze(0).to(torch.bfloat16).cpu().contiguous()
                fout.write(rows.view(torch.int16).numpy().tobytes())
                del out, rows
                packs_done += len(group)
                self.write_state("layer", layer, packs_done)
        os.replace(out_tmp, self.boundary_path(layer + 1))
        atomic_json(
            self.boundary_marker(layer + 1),
            {"layer": layer + 1, "tokens": self.shard_tokens, "fingerprint": self.fingerprint,
             "pack_range": list(self.pack_range)},
        )
        if store_writer is not None:
            store_writer.seal()
        if store_reader is not None:
            store_reader.close()
        if hook_handle is not None:
            hook_handle.remove()
        if capture_writer is not None:
            if capture_writer.tokens != self.shard_tokens:
                raise RuntimeError(
                    f"layer {layer}: captured {capture_writer.tokens} != shard {self.shard_tokens}"
                )
            capture_writer.close()
            for key in self.verify_totals:
                self.verify_totals[key] += verify[key]
            atomic_json(
                self.done_marker(layer),
                {
                    "layer": layer,
                    "shard": self.shard_index,
                    "fingerprint": self.fingerprint,
                    "tokens": capture_writer.tokens,
                    "token_offset": self.shard_token_offset,
                    "routed": capture_writer.routed,
                    "sha256_x_region": capture_writer.hx.hexdigest(),
                    "sha256_ids_region": capture_writer.hi.hexdigest(),
                    "verify": verify,
                },
            )
        dematerialize(module)
        self.write_state("layer-done", layer, packs_done, force=True)
        log(
            f"shard {self.shard_index}: layer {layer} done in {time.time()-t0:.1f}s "
            f"(load {t_load:.1f}s), {self.shard_tokens} tokens"
            + (f", captured" if layer in self.capture_layers else ""),
            self.logfile,
        )

    def maybe_delete_inputs(self, layer: int) -> None:
        """After layer L is fully complete, its input boundary is dead; a topk store
        is dead once every consumer within this pass has a sealed output AND no
        consumer beyond the stop layer exists."""
        if not self.layer_complete(layer):
            return
        if self.boundary_state(layer) == "sealed":
            self.boundary_path(layer).unlink()
            meta = json.loads(self.boundary_marker(layer).read_text())
            meta["consumed"] = True
            atomic_json(self.boundary_marker(layer), meta)
        for full_layer in range(0, layer + 1):
            if not self.geom.store_needed(full_layer) or self.store_state(full_layer) != "sealed":
                continue
            consumers = self.geom.store_consumers(full_layer)
            if any(c > self.stop_after_layer for c in consumers):
                continue  # needed by a later window/pass (or later layer in this one)
            if all(self.boundary_sealed(c + 1) for c in consumers):
                self.store_path(full_layer).unlink()
                meta_path = Path(str(self.store_path(full_layer)) + ".json")
                meta = json.loads(meta_path.read_text())
                meta["consumed"] = True
                meta.pop("records", None)
                atomic_json(meta_path, meta)


# ---- seal ---------------------------------------------------------------------


def seal_capture(
    plan: dict,
    src: Path,
    work_dir: Path,
    capture_dir: Path,
    layers: list[int],
    shard_count: int,
    stop_after_layer: int,
    pack_tokens: int,
    moe_group: int,
    logfile: str,
    shard_cost_overhead: int = 0,
) -> dict:
    import torch
    import transformers
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(str(src))
    geom = Geometry(config)
    tokens = int(plan["total_tokens"])
    run_uuid = str(uuid.uuid4())
    verify = {"checked_tokens": 0, "mismatch_tokens": 0, "kwarg_logits_layers": 0}

    shard_markers: dict[int, dict] = {}
    for shard in range(shard_count):
        marker_path = work_dir / f"shard{shard}" / "shard.done.json"
        if not marker_path.is_file():
            raise RuntimeError(f"seal: shard {shard} not finished ({marker_path} missing)")
        shard_markers[shard] = json.loads(marker_path.read_text())
        if shard_markers[shard].get("fingerprint") != plan["capture_fingerprint"]:
            raise RuntimeError(f"seal: shard {shard} fingerprint mismatch")
        for key in verify:
            verify[key] += int(shard_markers[shard]["verify"][key])
    if sum(int(m["tokens"]) for m in shard_markers.values()) != tokens:
        raise RuntimeError("seal: shard token counts do not sum to plan total")

    for layer in layers:
        layer_dir = capture_dir / f"layer_{layer:03d}"
        routed = [0] * NUM_EXPERTS
        layer_tokens = 0
        for shard in range(shard_count):
            done_path = work_dir / f"shard{shard}" / f"layer_{layer:03d}.done.json"
            if not done_path.is_file():
                raise RuntimeError(f"seal: layer {layer} shard {shard} done marker missing")
            done = json.loads(done_path.read_text())
            if done.get("fingerprint") != plan["capture_fingerprint"]:
                raise RuntimeError(f"seal: layer {layer} shard {shard} fingerprint mismatch")
            layer_tokens += int(done["tokens"])
            for expert, value in enumerate(done["routed"]):
                routed[expert] += int(value)
        if layer_tokens != tokens:
            raise RuntimeError(f"seal: layer {layer} tokens {layer_tokens} != plan {tokens}")
        if len(routed) != NUM_EXPERTS or sum(routed) != tokens * TOPK:
            raise RuntimeError(f"seal: layer {layer} routed count audit failed")
        for name in ("x.bin", "ids.bin"):
            partial = layer_dir / (name + ".partial")
            final = layer_dir / name
            if partial.is_file():
                os.replace(partial, final)
            if not final.is_file():
                raise RuntimeError(f"seal: layer {layer} missing {name}")
        if (layer_dir / "x.bin").stat().st_size != tokens * geom.hidden * 2:
            raise RuntimeError(f"seal: layer {layer} x payload size mismatch")
        if (layer_dir / "ids.bin").stat().st_size != tokens * TOPK:
            raise RuntimeError(f"seal: layer {layer} ids payload size mismatch")
        t0 = time.time()
        sha_x = sha256_file(layer_dir / "x.bin")
        sha_ids = sha256_file(layer_dir / "ids.bin")
        manifest = {
            "schema": "glm52-b300-layer-capture-v1",
            "layer": layer,
            "capture_fingerprint": plan["capture_fingerprint"],
            "capture_run_uuid": run_uuid,
            "capture_tp": CAPTURE_TP,
            "owner_rank": 0,
            "tokens": tokens,
            "hidden": geom.hidden,
            "x_dtype": "bfloat16",
            "x_bytes": tokens * geom.hidden * 2,
            "ids_topk": TOPK,
            "ids_bytes": tokens * TOPK,
            "sha256_x": sha_x,
            "sha256_ids": sha_ids,
            "routed_counts": routed,
            "routed_min": min(routed),
            "routed_max": max(routed),
            "cold_experts_lt1024": [expert for expert, count in enumerate(routed) if count < 1024],
            "finished": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        atomic_json(layer_dir / "layer_manifest.json", manifest)
        log(f"sealed layer {layer}: sha256 in {time.time()-t0:.1f}s", logfile)

    if verify["checked_tokens"]:
        fraction = 1.0 - verify["mismatch_tokens"] / verify["checked_tokens"]
        if fraction < 0.99:
            raise RuntimeError(f"routing recompute vs model topk agreement only {fraction:.6f}")

    pass_audit = [
        {"index": i, "name": p["name"], "samples": len(p["samples"]), "tokens": int(p["tokens"])}
        for i, p in enumerate(plan["passes"])
    ]
    preserved_boundary = stop_after_layer + 1
    preserved_stores = sorted(
        full
        for full in range(0, stop_after_layer + 1)
        if geom.store_needed(full)
        and any(c > stop_after_layer for c in geom.store_consumers(full))
    )
    run_manifest = {
        "schema": "glm52-b300-capture-run-v1",
        "capture_engine": f"transformers-{transformers.__version__}-layer-stream",
        "capture_fingerprint": plan["capture_fingerprint"],
        "capture_run_uuid": run_uuid,
        "active_layers": sorted(layers),
        "tokens_per_layer": tokens,
        "capture_payload_bytes": tokens * len(layers) * (geom.hidden * 2 + TOPK),
        "pass_audit": pass_audit,
        "rank_assignment": {str(layer): 0 for layer in sorted(layers)},
        "verify_routing": verify,
        "capture_tp": CAPTURE_TP,
        "output_tp": OUTPUT_TP,
        "enforce_eager": True,
        "enable_prefix_caching": False,
        "max_tokens": 0,
        "prefill_only": True,
        "cpu_offload_gb": 0,
        "low_memory_mode": False,
        "model_snapshot": str(src),
        "corpus_sha256": plan["corpus_sha256"],
        "attn_implementation": ATTN_IMPL,
        "experts_implementation": EXPERTS_IMPL,
        "batching": (
            f"layer-major streaming; samples packed in exact plan order into <= {pack_tokens}-token "
            f"batches with block-diagonal causal masks and per-sample position ids (no cross-sample "
            f"attention); MoE invoked once per {moe_group}-pack group; {shard_count} contiguous corpus "
            f"shard(s) merged at byte offsets"
        ),
        "row_order_note": (
            "rows are written at absolute plan-order offsets (all tokens of sample 1, then sample 2, "
            "...). The downstream encoder accumulates order-independent Hessian sums (H += x^T x per "
            "expert), so only the set of (token, x, routing) rows and the per-layer counts are "
            "contractual."
        ),
        "verify_routing_note": (
            "verify_routing compares the reference fp32 gate recompute (sigmoid(x@W^T)+bias, top-8) "
            "against the transformers router's own topk indices at every captured token; "
            "kwarg_logits_layers counts hook invocations where the model's indices were visible."
        ),
        "layer_stream": {
            "stop_after_layer": stop_after_layer,
            "pack_tokens": pack_tokens,
            "moe_group": moe_group,
            "shard_count": shard_count,
            "shard_cost_overhead": shard_cost_overhead,
            "boundary_preserved_layer": preserved_boundary,
            "boundary_preserved_files": [
                str(work_dir / f"shard{s}" / f"boundary_{preserved_boundary:03d}.bin")
                for s in range(shard_count)
            ],
            "topk_stores_preserved": {
                str(full): [str(work_dir / f"shard{s}" / f"topk_{full:03d}.bin") for s in range(shard_count)]
                for full in preserved_stores
            },
            "resume_note": (
                f"next window: rerun with --layers starting at {stop_after_layer + 1} and a matching "
                f"--stop-after-layer; boundary_{preserved_boundary:03d} shard files above are its "
                f"input and the listed topk stores feed its leading shared-indexer layers. Identical "
                f"--pack-tokens/--shard-count/--shard-cost-overhead are required for pack/shard "
                f"coordinate consistency."
            ),
        },
        "transformers": transformers.__version__,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
        "finished": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    atomic_json(capture_dir / "capture_run_manifest.json", run_manifest)
    log(
        f"CAPTURE SEALED: {len(layers)} layer(s), {tokens} tokens/layer, "
        f"{run_manifest['capture_payload_bytes']/2**30:.2f} GiB payload, "
        f"fingerprint={plan['capture_fingerprint']}",
        logfile,
    )
    return run_manifest


def spot_check(capture_dir: Path, src: Path, layers: list[int], tokens: int, logfile: str) -> list[dict]:
    """capture_hf.validate_capture's gate-recompute spot check, gate weights read
    directly from the checkpoint shards."""
    import numpy as np
    import torch

    ckpt = ShardIndex(src)
    hidden = None
    rng = random.Random(1234)
    check_rows = 64
    tie_tolerance = 1e-5
    results = []
    for layer in layers:
        gate_w = ckpt.load_tensor(f"model.layers.{layer}.mlp.gate.weight", "cuda:0").to(torch.float32)
        gate_b = (
            ckpt.load_tensor(f"model.layers.{layer}.mlp.gate.e_score_correction_bias", "cuda:0")
            .to(torch.float32)
            .flatten()
        )
        hidden = gate_w.shape[1]
        layer_dir = capture_dir / f"layer_{layer:03d}"
        rows = sorted(rng.sample(range(tokens), check_rows))
        exact = 0
        near_tie = 0
        with open(layer_dir / "x.bin", "rb") as fx, open(layer_dir / "ids.bin", "rb") as fi:
            for row in rows:
                fx.seek(row * hidden * 2)
                x_np = np.frombuffer(fx.read(hidden * 2), dtype=np.int16).copy()
                fi.seek(row * TOPK)
                ids_disk = np.frombuffer(fi.read(TOPK), dtype=np.uint8)
                x = torch.from_numpy(x_np).view(torch.bfloat16).to("cuda:0")
                logits = torch.nn.functional.linear(x.to(torch.float32).unsqueeze(0), gate_w)
                scores = (torch.sigmoid(logits) + gate_b).squeeze(0)
                ids_re = torch.topk(scores, TOPK, dim=-1, sorted=False).indices.cpu().numpy()
                if set(ids_re.tolist()) == set(ids_disk.tolist()):
                    exact += 1
                else:
                    sorted_scores = torch.sort(scores, descending=True).values
                    boundary = sorted_scores[TOPK - 1].item()
                    disputed = set(ids_disk.tolist()) ^ set(ids_re.tolist())
                    gaps = [abs(scores[e].item() - boundary) for e in disputed]
                    if max(gaps) <= tie_tolerance:
                        near_tie += 1
                    else:
                        raise RuntimeError(
                            f"SPOT CHECK layer {layer} row {row}: ids mismatch beyond tie tolerance: "
                            f"disk={sorted(ids_disk.tolist())} recompute={sorted(ids_re.tolist())} "
                            f"max_gap={max(gaps)}"
                        )
        results.append({"layer": layer, "rows": check_rows, "exact": exact, "near_tie": near_tie})
        log(
            f"SPOT CHECK layer {layer}: {exact}/{check_rows} exact top-8, {near_tie} near-tie",
            logfile,
        )
    return results


# ---- compare (validation gate) ------------------------------------------------


def compare_captures(ref_dir: Path, new_dir: Path, layers: list[int], sample_rows: int, logfile: str) -> dict:
    import numpy as np

    rng = np.random.default_rng(20260810)
    report = {"layers": [], "gate": {"ids_min_match": 1.0, "x_min_row_pass": 1.0}}
    for layer in layers:
        ref_l = ref_dir / f"layer_{layer:03d}"
        new_l = new_dir / f"layer_{layer:03d}"
        ref_m = json.loads((ref_l / "layer_manifest.json").read_text())
        new_m = json.loads((new_l / "layer_manifest.json").read_text())
        tokens = int(ref_m["tokens"])
        hidden = int(ref_m["hidden"])
        if int(new_m["tokens"]) != tokens or int(new_m["hidden"]) != hidden:
            raise RuntimeError(f"layer {layer}: manifest geometry mismatch")

        ids_ref = np.fromfile(ref_l / "ids.bin", dtype=np.uint8).reshape(tokens, TOPK)
        ids_new = np.fromfile(new_l / "ids.bin", dtype=np.uint8).reshape(tokens, TOPK)
        rows_equal = (np.sort(ids_ref, axis=1) == np.sort(ids_new, axis=1)).all(axis=1)
        ids_match = float(rows_equal.mean())

        n_sample = min(sample_rows, tokens)
        rows = np.sort(rng.choice(tokens, size=n_sample, replace=False))
        row_bytes = hidden * 2
        x_ref = np.empty((n_sample, hidden), dtype=np.float32)
        x_new = np.empty((n_sample, hidden), dtype=np.float32)

        def read_rows(path, out):
            import torch

            with open(path, "rb") as handle:
                for i, row in enumerate(rows):
                    handle.seek(int(row) * row_bytes)
                    buf = np.frombuffer(handle.read(row_bytes), dtype=np.int16).copy()
                    out[i] = torch.from_numpy(buf).view(torch.bfloat16).to(torch.float32).numpy()

        read_rows(ref_l / "x.bin", x_ref)
        read_rows(new_l / "x.bin", x_new)
        close = np.abs(x_new - x_ref) <= (0.02 + 0.02 * np.abs(x_ref))
        row_pass = close.all(axis=1)
        denom = np.abs(x_ref) + 1e-6
        rel = np.abs(x_new - x_ref) / denom
        bitwise = float((x_new == x_ref).all(axis=1).mean())
        entry = {
            "layer": layer,
            "tokens": tokens,
            "ids_row_match": ids_match,
            "x_rows_sampled": n_sample,
            "x_rows_allclose_frac": float(row_pass.mean()),
            "x_elems_within_tol_frac": float(close.mean()),
            "x_rows_bitwise_frac": bitwise,
            "x_rel_p50": float(np.median(rel)),
            "x_rel_p999": float(np.quantile(rel, 0.999)),
            "routed_counts_equal": ref_m["routed_counts"] == new_m["routed_counts"],
            "sha_x_equal": ref_m["sha256_x"] == new_m["sha256_x"],
        }
        report["layers"].append(entry)
        report["gate"]["ids_min_match"] = min(report["gate"]["ids_min_match"], ids_match)
        report["gate"]["x_min_row_pass"] = min(report["gate"]["x_min_row_pass"], entry["x_rows_allclose_frac"])
        log(
            f"COMPARE layer {layer}: ids row match {ids_match*100:.4f}%, "
            f"x rows allclose {entry['x_rows_allclose_frac']*100:.4f}% "
            f"(bitwise {bitwise*100:.2f}%, rel p50 {entry['x_rel_p50']:.2e}, "
            f"p99.9 {entry['x_rel_p999']:.2e}), routed_equal={entry['routed_counts_equal']}",
            logfile,
        )
    report["gate"]["pass"] = (
        report["gate"]["ids_min_match"] >= 0.999 and report["gate"]["x_min_row_pass"] >= 0.999
    )
    log(f"COMPARE GATE: {json.dumps(report['gate'])}", logfile)
    return report


# ---- selftest -----------------------------------------------------------------


def selftest(scratch: Path, logfile: str) -> None:
    """Build a small random GlmMoeDsa model with mixed full/shared indexer layers,
    save an unfused checkpoint, and verify:
      A) streaming with single-sample packs + moe_group=1 is BITWISE equal to the
         reference full-model forward (boundaries, capture x and ids);
      B) packed multi-sample batches + moe_group>1 agree within bf16 tolerance.
    """
    import numpy as np
    import torch
    from safetensors.torch import save_file
    from transformers import AutoModelForCausalLM
    from transformers.models.glm_moe_dsa.configuration_glm_moe_dsa import GlmMoeDsaConfig

    scratch.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(20260810)
    num_layers = 8
    cfg = GlmMoeDsaConfig(
        hidden_size=128,
        intermediate_size=256,
        moe_intermediate_size=64,
        num_hidden_layers=num_layers,
        first_k_dense_replace=2,
        mlp_layer_types=["dense", "dense"] + ["sparse"] * (num_layers - 2),
        indexer_types=["full", "shared", "full", "shared", "shared", "full", "shared", "shared"],
        n_routed_experts=NUM_EXPERTS,
        num_experts_per_tok=TOPK,
        n_shared_experts=1,
        n_group=1,
        topk_group=1,
        norm_topk_prob=True,
        routed_scaling_factor=1.5,
        scoring_func="sigmoid",
        num_attention_heads=4,
        num_key_value_heads=4,
        qk_nope_head_dim=24,
        qk_rope_head_dim=8,
        qk_head_dim=32,
        v_head_dim=32,
        head_dim=24,
        kv_lora_rank=32,
        q_lora_rank=48,
        index_n_heads=4,
        index_head_dim=16,
        index_topk=24,
        vocab_size=512,
        max_position_embeddings=4096,
        num_nextn_predict_layers=0,
        pad_token_id=0,
    )
    with torch.device("cuda:0"):
        init_model = AutoModelForCausalLM.from_config(
            cfg, attn_implementation=ATTN_IMPL, experts_implementation=EXPERTS_IMPL
        )
    init_model = init_model.to(torch.bfloat16)
    for _, module in init_model.named_modules():
        bias = getattr(module, "e_score_correction_bias", None)
        if bias is not None:
            module.e_score_correction_bias = torch.nn.init.normal_(bias.to(torch.float32), std=0.1)

    # save unfused checkpoint (per-expert keys, e_score bias fp32 like GLM-5.2)
    sd = {}
    for key, value in init_model.state_dict().items():
        if key.endswith("mlp.experts.gate_up_proj"):
            base = key[: -len("gate_up_proj")]
            inter = cfg.moe_intermediate_size
            for e in range(NUM_EXPERTS):
                sd[f"{base}{e}.gate_proj.weight"] = value[e, :inter].contiguous()
                sd[f"{base}{e}.up_proj.weight"] = value[e, inter:].contiguous()
        elif key.endswith("mlp.experts.down_proj"):
            base = key[: -len("down_proj")]
            for e in range(NUM_EXPERTS):
                sd[f"{base}{e}.down_proj.weight"] = value[e].contiguous()
        else:
            sd[key] = value.contiguous()
    sd = {k: v.cpu() for k, v in sd.items()}
    ckpt_dir = scratch / "ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    save_file(sd, str(ckpt_dir / "model.safetensors"))
    atomic_json(
        ckpt_dir / "model.safetensors.index.json",
        {"metadata": {}, "weight_map": {k: "model.safetensors" for k in sd}},
    )
    cfg.save_pretrained(str(ckpt_dir))
    del init_model
    gc.collect()
    torch.cuda.empty_cache()

    # reference model exactly as capture_hf.build_model constructs it
    ref_model = AutoModelForCausalLM.from_pretrained(
        str(ckpt_dir),
        dtype=torch.bfloat16,
        attn_implementation=ATTN_IMPL,
        experts_implementation=EXPERTS_IMPL,
        trust_remote_code=False,
    ).to("cuda:0")
    ref_model.eval()
    ref_model.requires_grad_(False)

    rng = random.Random(7)
    samples = [[rng.randrange(cfg.vocab_size) for _ in range(rng.randrange(9, 61))] for _ in range(14)]
    sample_lens = [len(s) for s in samples]
    total = sum(sample_lens)
    moe_layers = list(range(2, num_layers))
    hidden = cfg.hidden_size

    # reference: full-model forward, batch=1 per sample, hooks on layer inputs + experts
    ref_bound = {L: [] for L in range(num_layers + 1)}
    ref_x = {L: [] for L in moe_layers}
    ref_ids = {L: [] for L in moe_layers}
    handles = []
    for L in range(num_layers):
        def bhook(mod, args, kwargs, L=L):
            ref_bound[L].append(args[0].detach().squeeze(0).to(torch.bfloat16).cpu())
        handles.append(ref_model.model.layers[L].register_forward_pre_hook(bhook, with_kwargs=True))
    for L in moe_layers:
        gate = ref_model.model.layers[L].mlp.gate
        gate_w = gate.weight.detach().to(torch.float32)
        gate_b = gate.e_score_correction_bias.detach().to(torch.float32).flatten()
        def xhook(mod, args, kwargs, L=L, gate_w=gate_w, gate_b=gate_b):
            hs = kwargs.get("hidden_states")
            if hs is None:
                hs = args[0]
            logits = torch.nn.functional.linear(hs.to(torch.float32), gate_w)
            scores = torch.sigmoid(logits) + gate_b
            ids = torch.topk(scores, TOPK, dim=-1, sorted=False).indices
            ref_x[L].append(hs.detach().to(torch.bfloat16).cpu())
            ref_ids[L].append(ids.to(torch.uint8).cpu())
        handles.append(
            ref_model.model.layers[L].mlp.experts.register_forward_pre_hook(xhook, with_kwargs=True)
        )
    with torch.inference_mode():
        for s in samples:
            ref_model.model(
                input_ids=torch.tensor([s], dtype=torch.long, device="cuda:0"), use_cache=False
            )
    for h in handles:
        h.remove()
    ref_bound = {L: torch.cat(v, dim=0) for L, v in ref_bound.items() if v}
    ref_x = {L: torch.cat(v, dim=0) for L, v in ref_x.items()}
    ref_ids = {L: torch.cat(v, dim=0) for L, v in ref_ids.items()}
    del ref_model
    gc.collect()
    torch.cuda.empty_cache()

    def run_mode(tag: str, pack_tokens: int, moe_group: int) -> tuple[dict, dict, dict]:
        work = scratch / f"work-{tag}"
        cap = scratch / f"cap-{tag}"
        for d in (work, cap):
            if d.exists():
                shutil.rmtree(d)
        runner = StreamRunner(
            src=ckpt_dir,
            samples=samples,
            work_dir=work,
            capture_dir=cap,
            capture_layers=moe_layers,
            stop_after_layer=num_layers - 1,
            pack_tokens=pack_tokens,
            moe_group=moe_group,
            shard_index=0,
            shard_count=1,
            state_root=None,
            logfile=logfile,
        )
        # keep boundaries for comparison
        runner.maybe_delete_inputs = lambda layer: None
        runner.run()
        bounds = {}
        for L in range(num_layers):
            data = np.fromfile(work / "shard0" / f"boundary_{L:03d}.bin", dtype=np.int16)
            bounds[L] = torch.from_numpy(data.copy()).view(torch.bfloat16).reshape(total, hidden)
        xs, idss = {}, {}
        for L in moe_layers:
            xd = np.fromfile(cap / f"layer_{L:03d}" / "x.bin.partial", dtype=np.int16)
            xs[L] = torch.from_numpy(xd.copy()).view(torch.bfloat16).reshape(total, hidden)
            idss[L] = torch.from_numpy(
                np.fromfile(cap / f"layer_{L:03d}" / "ids.bin.partial", dtype=np.uint8).copy()
            ).reshape(total, TOPK)
        return bounds, xs, idss

    # Mode A: single-sample packs (budget forces 1 sample/pack), moe_group=1 -> bitwise
    bounds, xs, idss = run_mode("exact", pack_tokens=1, moe_group=1)
    for L in range(num_layers):
        if not torch.equal(bounds[L], ref_bound[L]):
            diff = (bounds[L].float() - ref_bound[L].float()).abs().max().item()
            raise RuntimeError(f"selftest A: boundary {L} not bitwise (max abs {diff})")
    for L in moe_layers:
        if not torch.equal(xs[L], ref_x[L]):
            raise RuntimeError(f"selftest A: capture x layer {L} not bitwise")
        if not torch.equal(
            idss[L].to(torch.int16).sort(dim=1).values, ref_ids[L].to(torch.int16).sort(dim=1).values
        ):
            raise RuntimeError(f"selftest A: capture ids layer {L} mismatch")
    log("selftest A PASS: single-sample streaming bitwise-equal to full-model forward", logfile)

    # Mode B: packed multi-sample + grouped MoE -> tolerance.  A random-weight
    # model has ~half its indexer relu scores exactly 0, so the DSA top-k
    # boundary sits in a huge tie region and packed vs per-sample selection
    # legitimately flips ties; thresholds here only catch gross orchestration
    # bugs (wrong masks/offsets give ~0 agreement).  The strict gate is the
    # Fruit comparison against the sealed reference capture.
    bounds, xs, idss = run_mode("packed", pack_tokens=96, moe_group=3)
    worst_ids = 1.0
    for L in moe_layers:
        ref = ref_x[L].float()
        new = xs[L].float()
        close = (new - ref).abs() <= (0.05 + 0.05 * ref.abs())
        frac = close.all(dim=1).float().mean().item()
        ids_match = (
            (idss[L].to(torch.int16).sort(dim=1).values == ref_ids[L].to(torch.int16).sort(dim=1).values)
            .all(dim=1)
            .float()
            .mean()
            .item()
        )
        worst_ids = min(worst_ids, ids_match)
        log(
            f"selftest B layer {L}: x rows allclose {frac:.4f}, ids match {ids_match:.4f}",
            logfile,
        )
        if frac < 0.90:
            raise RuntimeError(f"selftest B: layer {L} x rows allclose only {frac}")
        if ids_match < 0.85:
            raise RuntimeError(f"selftest B: layer {L} ids match only {ids_match}")
    log(f"selftest B PASS: packed/grouped within tolerance (worst ids match {worst_ids:.4f})", logfile)
    shutil.rmtree(scratch)


# ---- main ---------------------------------------------------------------------


def parse_layers(spec: str, moe_layers: list[int]) -> list[int]:
    layers: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            layers.extend(range(int(start), int(end) + 1))
        else:
            layers.append(int(part))
    out = sorted(set(layers))
    if not out or any(layer not in moe_layers for layer in out):
        raise ValueError(f"capture layers must be a nonempty subset of MoE layers, got {spec!r}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Layer-streaming BF16 GLM-5.2 capture")
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--make-plan", action="store_true")
    modes.add_argument("--run", action="store_true", help="run one streaming shard")
    modes.add_argument("--seal", action="store_true", help="merge shard results, write ABI manifests")
    modes.add_argument("--selftest", action="store_true")
    modes.add_argument("--compare", action="store_true", help="validation gate vs a reference capture")
    parser.add_argument("--src", type=Path)
    parser.add_argument("--corpus", type=Path)
    parser.add_argument("--plan-file", type=Path)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--capture-dir", type=Path)
    parser.add_argument("--layers", default=None)
    parser.add_argument("--stop-after-layer", type=int, default=None)
    parser.add_argument("--pack-tokens", type=int, default=8192)
    parser.add_argument("--moe-group", type=int, default=8)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument(
        "--shard-cost-overhead",
        type=int,
        default=0,
        help="token-equivalent per-sample cost used to balance shards (per-sample "
        "grouped_mm pays a full expert-weight read; ~2000 for GLM-5.2). Must be "
        "identical across shards and across resumed windows.",
    )
    parser.add_argument("--target-tokens", type=int, default=TARGET_TOKENS)
    parser.add_argument("--state-file", type=Path, default=None)
    parser.add_argument("--ref-dir", type=Path)
    parser.add_argument("--new-dir", type=Path)
    parser.add_argument("--sample-rows", type=int, default=4096)
    parser.add_argument("--scratch", type=Path, default=Path("/tmp/capture-stream-selftest"))
    parser.add_argument("--log", default=None)
    args = parser.parse_args()
    logfile = args.log

    if args.selftest:
        selftest(args.scratch.resolve(), logfile)
        log("SELFTEST PASSED", logfile)
        return

    if args.compare:
        ref_dir = args.ref_dir.resolve()
        new_dir = args.new_dir.resolve()
        layers = sorted(
            int(p.name.split("_")[1])
            for p in ref_dir.glob("layer_*")
            if (p / "layer_manifest.json").is_file() and (new_dir / p.name / "layer_manifest.json").is_file()
        )
        if args.layers:
            wanted = set()
            for part in args.layers.split(","):
                if "-" in part:
                    a, b = part.split("-")
                    wanted.update(range(int(a), int(b) + 1))
                elif part.strip():
                    wanted.add(int(part))
            layers = [l for l in layers if l in wanted]
        if not layers:
            raise SystemExit("no common sealed layers to compare")
        report = compare_captures(ref_dir, new_dir, layers, args.sample_rows, logfile)
        out = new_dir / "compare_report.json"
        atomic_json(out, report)
        log(f"compare report written: {out}", logfile)
        raise SystemExit(0 if report["gate"]["pass"] else 1)

    from transformers import AutoConfig

    src = args.src.resolve()
    corpus = args.corpus.resolve()
    plan_file = args.plan_file.resolve()
    config = AutoConfig.from_pretrained(str(src))
    geom = Geometry(config)

    if args.make_plan:
        plan = build_capture_plan(src, corpus, args.target_tokens, geom, logfile)
        if plan_file.exists():
            existing = json.loads(plan_file.read_text())
            validate_plan(existing, src, corpus)
            if existing["capture_fingerprint"] != plan["capture_fingerprint"]:
                raise SystemExit(
                    f"existing plan {plan_file} has a different fingerprint; move it explicitly "
                    "before changing calibration selection"
                )
            log(f"capture plan already exists and matches: {plan_file}", logfile)
        else:
            atomic_json(plan_file, plan)
            log(f"wrote capture plan: {plan_file}", logfile)
        return

    plan = json.loads(plan_file.read_text())
    validate_plan(plan, src, corpus)
    layers = parse_layers(args.layers, geom.moe_layers)
    stop_after = args.stop_after_layer if args.stop_after_layer is not None else max(layers)
    work_dir = args.work_dir.resolve()
    capture_dir = args.capture_dir.resolve()
    log(
        f"plan validated: fingerprint={plan['capture_fingerprint']} "
        f"total_tokens={plan['total_tokens']} layers={layers} stop_after={stop_after}",
        logfile,
    )

    if args.seal:
        run_manifest = seal_capture(
            plan, src, work_dir, capture_dir, layers, args.shard_count,
            stop_after, args.pack_tokens, args.moe_group, logfile,
            shard_cost_overhead=args.shard_cost_overhead,
        )
        results = spot_check(capture_dir, src, layers, int(plan["total_tokens"]), logfile)
        log(f"SEAL VALIDATION PASSED: {json.dumps(results)}", logfile)
        return

    # --run
    samples = load_plan_tokens(plan, src, corpus, logfile)
    if sum(map(len, samples)) != int(plan["total_tokens"]):
        raise RuntimeError("sample token sum != plan total")
    runner = StreamRunner(
        src=src,
        samples=samples,
        work_dir=work_dir,
        capture_dir=capture_dir,
        capture_layers=layers,
        stop_after_layer=stop_after,
        pack_tokens=args.pack_tokens,
        moe_group=args.moe_group,
        shard_index=args.shard_index,
        shard_count=args.shard_count,
        state_root=args.state_file,
        logfile=logfile,
        capture_fingerprint=plan["capture_fingerprint"],
        shard_cost_overhead=args.shard_cost_overhead,
    )
    marker = runner.run()
    log(f"shard {args.shard_index} finished: {json.dumps(marker['verify'])}", logfile)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
        sys.exit(1)
