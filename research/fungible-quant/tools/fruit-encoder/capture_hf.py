#!/usr/bin/env python3
"""Transformers-based BF16 activation capture for the GLM-5.2 SIQ-Fruit proxy model.

Drop-in replacement for the vLLM capture path in capture_fruit.py (which crashes on
attention-backend incompatibilities in this environment).  The on-disk capture ABI is
byte-compatible with capture_fruit.py so the downstream LayerCalib encoder consumes it
unchanged:

  <capture_dir>/layer_LLL/x.bin              bf16 [tokens, 1024] raw bytes (int16 view)
  <capture_dir>/layer_LLL/ids.bin            uint8 [tokens, 8] natural top-8 routed ids
  <capture_dir>/layer_LLL/layer_manifest.json  schema glm52-b300-layer-capture-v1
  <capture_dir>/capture_run_manifest.json      schema glm52-b300-capture-run-v1

Capture point: forward_pre_hook on model.layers[L].mlp.experts (GlmMoeDsaExperts).
In transformers' GlmMoeDsaMoE.forward, the SAME post-attention-layernorm tensor is fed
to the router gate (`self.gate(hidden_states)`) and, viewed as [tokens, hidden], to
`self.experts(hidden_states, topk_indices, topk_weights)`.  The hook therefore sees
exactly the tensor the routing gate sees, plus the model's own topk indices, which are
cross-checked against the reference recompute on every token.

Routing ids are recomputed with capture_fruit.py's exact math and dtypes:
  logits = F.linear(x.to(float32), gate.weight.float32)
  scores = sigmoid(logits) + e_score_correction_bias.float32
  ids    = topk(scores, 8, sorted=False).indices.to(uint8)
(n_group == topk_group == 1, so the transformers group masking is a no-op.)

Row order: samples are forwarded one at a time (batch=1, no padding) in exact plan
order, so rows land as: all tokens of sample 1, then sample 2, ...  This may differ
from the vLLM engine's internal scheduling order; the downstream encoder builds
Hessians as order-independent sums (H += x^T x per expert), so only the SET of
(token, x, routing) rows and the manifest counts must match, which they do.

Capture is PREFILL ONLY (the vLLM reference generated max_tokens=1, i.e. its captured
rows are exactly the prompt tokens).
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time
import uuid

CORPUS_SHA256 = "cf247acc7c5da9f0600c7d6ab3b7c2fcfc54ec30b794e3b6047559285fa44df4"
AXES = (
    "axis1_general",
    "axis2_legal",
    "axis3_code_agentic",
    "axis4_reasoning_termination",
)

MAX_SAMPLE_TOKENS = 4_096
HIDDEN = 1_024
NUM_EXPERTS = 256
TOPK = 8
FIRST_MOE_LAYER = 3
NUM_LAYERS = 13
MOE_LAYERS = list(range(FIRST_MOE_LAYER, NUM_LAYERS))
CAPTURE_TP = 1
OUTPUT_TP = 4

DEFAULT_SRC = (
    "/home/mbelleau/.cache/huggingface/hub/"
    "models--malaiwah--GLM-5.2-SIQ-Fruit-Instruct-bf16/snapshots/"
    "678954f65e056a0f508e21eeb9251c655bb9463f"
)
DEFAULT_CORPUS = (
    "/home/mbelleau/.cache/huggingface/hub/"
    "models--brandonmusic--GLM-5.2-EXL3-TR3-3.0bpw/snapshots/"
    "9297b9f1d53af5c67cffa01e30cc071a1ff7144b/calibration_encoder/calibration/"
    "reap_recall_calib.jsonl"
)


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
    tmp = path.with_name(path.name + ".tmp")
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


def parse_layers(spec: str) -> list[int]:
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
    if not out or any(layer not in MOE_LAYERS for layer in out):
        raise ValueError(f"capture layers must be a nonempty subset of {MOE_LAYERS}, got {spec!r}")
    return out


def validate_plan(plan: dict, src: Path, corpus: Path) -> None:
    """Ported verbatim (modulo formatting) from capture_fruit.validate_plan."""
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


def load_plan_tokens(
    plan: dict, src: Path, corpus: Path, logfile: str
) -> list[tuple[str, list[list[int]]]]:
    """Ported from capture_fruit.load_plan_tokens: identical selection + drift guards."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(src), trust_remote_code=False)
    identity = _tokenizer_identity(src, tokenizer)
    if identity != plan.get("tokenizer"):
        raise RuntimeError(
            f"tokenizer identity changed since capture plan was built: "
            f"{identity} != {plan.get('tokenizer')}"
        )
    raw_lines = corpus.read_text(encoding="utf-8").splitlines()
    out = []
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
        out.append((str(pass_info["name"]), token_lists))
    return out


class LayerCapture:
    """Streaming writer for one MoE layer: x.bin + ids.bin + hashes + routed counts."""

    def __init__(self, layer: int, out_dir: Path):
        self.layer = layer
        self.dir = out_dir / f"layer_{layer:03d}"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.fx = open(self.dir / "x.bin.partial", "wb", buffering=16 << 20)
        self.fi = open(self.dir / "ids.bin.partial", "wb", buffering=1 << 20)
        self.hx = hashlib.sha256()
        self.hi = hashlib.sha256()
        self.tokens = 0
        self.routed = [0] * NUM_EXPERTS

    def write(self, x_bytes: bytes, id_bytes: bytes, rows: int, bincount) -> None:
        self.hx.update(x_bytes)
        self.hi.update(id_bytes)
        self.fx.write(x_bytes)
        self.fi.write(id_bytes)
        self.tokens += rows
        for expert, value in enumerate(bincount):
            self.routed[expert] += int(value)

    def finalize(self) -> None:
        for handle in (self.fx, self.fi):
            handle.flush()
            handle.close()
        os.replace(self.dir / "x.bin.partial", self.dir / "x.bin")
        os.replace(self.dir / "ids.bin.partial", self.dir / "ids.bin")


def build_model(src: Path, attn_impl: str, experts_impl: str, logfile: str):
    import torch
    from transformers import AutoModelForCausalLM

    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        str(src),
        dtype=torch.bfloat16,
        attn_implementation=attn_impl,
        experts_implementation=experts_impl,
        trust_remote_code=False,
    )
    model = model.to("cuda:0")
    model.eval()
    model.requires_grad_(False)
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    log(
        f"model loaded: {type(model).__name__} attn={attn_impl} experts={experts_impl} "
        f"in {time.time()-t0:.1f}s; GPU {torch.cuda.get_device_name()} "
        f"free={free_bytes/2**30:.1f}/{total_bytes/2**30:.1f} GiB",
        logfile,
    )
    return model


def install_hooks(model, layers: list[int], out_dir: Path, logfile: str):
    """Register forward_pre_hooks on layers[L].mlp.experts; return capture state."""
    import torch
    import torch.nn.functional as torch_f

    config = model.config
    routing = {
        "n_group": getattr(config, "n_group", None),
        "topk_group": getattr(config, "topk_group", None),
        "scoring_func": getattr(config, "scoring_func", None),
        "top_k": getattr(config, "num_experts_per_tok", None),
    }
    expected_routing = {"n_group": 1, "topk_group": 1, "scoring_func": "sigmoid", "top_k": TOPK}
    if routing != expected_routing:
        raise RuntimeError(f"routing config {routing} != {expected_routing}")

    decoder_layers = model.model.layers
    state = {
        "captures": {},
        "gate_w": {},
        "gate_b": {},
        "handles": [],
        "verify": {"checked_tokens": 0, "mismatch_tokens": 0, "kwarg_logits_layers": 0},
    }
    for layer in layers:
        mlp = decoder_layers[layer].mlp
        gate = getattr(mlp, "gate", None)
        experts = getattr(mlp, "experts", None)
        if gate is None or experts is None or not hasattr(gate, "weight"):
            raise RuntimeError(f"layer {layer}: mlp.gate / mlp.experts absent (not a MoE layer?)")
        weight = gate.weight
        bias = getattr(gate, "e_score_correction_bias", None)
        if tuple(weight.shape) != (NUM_EXPERTS, HIDDEN) or bias is None:
            raise RuntimeError(f"layer {layer}: unexpected router weight/bias shape")
        gate_w = weight.detach().to(torch.float32)
        gate_b = bias.detach().to(torch.float32).flatten()
        if tuple(gate_b.shape) != (NUM_EXPERTS,):
            raise RuntimeError(f"layer {layer}: router correction bias shape mismatch")
        state["gate_w"][layer] = gate_w
        state["gate_b"][layer] = gate_b
        state["captures"][layer] = LayerCapture(layer, out_dir)

    def make_hook(layer: int):
        gate_w = state["gate_w"][layer]
        gate_b = state["gate_b"][layer]
        capture = state["captures"][layer]
        verify = state["verify"]

        def hook(module, args, kwargs):
            hidden = kwargs.get("hidden_states")
            if hidden is None and args:
                hidden = args[0]
            if hidden is None or hidden.dim() != 2 or hidden.shape[1] != HIDDEN:
                raise RuntimeError(
                    f"layer {layer}: unexpected MoE input "
                    f"{None if hidden is None else tuple(hidden.shape)}"
                )
            rows = int(hidden.shape[0])
            # EXACT reference math (capture_fruit.py hook): fp32 gate recompute.
            logits = torch_f.linear(hidden.to(torch.float32), gate_w)
            scores = torch.sigmoid(logits) + gate_b
            ids = torch.topk(scores, TOPK, dim=-1, sorted=False).indices

            # Cross-check against the model's own router output (args[1] of
            # GlmMoeDsaExperts.forward(hidden_states, top_k_index, top_k_weights)).
            model_ids = kwargs.get("top_k_index")
            if model_ids is None and len(args) > 1:
                model_ids = args[1]
            if (
                model_ids is not None
                and model_ids.dim() == 2
                and tuple(model_ids.shape) == (rows, TOPK)
            ):
                bad = (ids.sort(dim=-1).values != model_ids.long().sort(dim=-1).values).any(dim=-1)
                verify["checked_tokens"] += rows
                verify["mismatch_tokens"] += int(bad.sum().item())
                verify["kwarg_logits_layers"] += 1

            ids_cpu = ids.to(torch.uint8).cpu().contiguous()
            hidden_cpu = hidden.detach().to(torch.bfloat16).cpu().contiguous()
            x_bytes = hidden_cpu.view(torch.int16).numpy().tobytes()
            id_bytes = ids_cpu.numpy().tobytes()
            bincount = torch.bincount(ids.flatten().to(torch.int64), minlength=NUM_EXPERTS).cpu().tolist()
            capture.write(x_bytes, id_bytes, rows, bincount)
            return None

        return hook

    for layer in layers:
        experts = decoder_layers[layer].mlp.experts
        state["handles"].append(
            experts.register_forward_pre_hook(make_hook(layer), with_kwargs=True)
        )
    log(f"hooks installed on layers {layers} (mlp.experts forward_pre_hook)", logfile)
    return state


def run_forwards(model, state, passes, layers, total_tokens: int, logfile: str) -> list[dict]:
    import torch

    device = "cuda:0"
    cumulative = 0
    samples_done = 0
    total_samples = sum(len(token_lists) for _, token_lists in passes)
    pass_audit = []
    t_start = time.time()
    t_log = t_start
    with torch.inference_mode():
        for pass_index, (pass_name, token_lists) in enumerate(passes):
            t_pass = time.time()
            for ids in token_lists:
                input_ids = torch.tensor([ids], dtype=torch.long, device=device)
                model.model(input_ids=input_ids, use_cache=False)
                cumulative += len(ids)
                samples_done += 1
                bad = {
                    layer: cap.tokens
                    for layer, cap in state["captures"].items()
                    if cap.tokens != cumulative
                }
                if bad:
                    raise RuntimeError(
                        f"token accounting: expected cumulative {cumulative}, got {bad}"
                    )
                now = time.time()
                if now - t_log > 15 or samples_done == total_samples:
                    rate = cumulative / max(now - t_start, 1e-9)
                    eta = (total_tokens - cumulative) / max(rate, 1e-9)
                    log(
                        f"progress: {cumulative}/{total_tokens} tokens "
                        f"({samples_done}/{total_samples} samples), "
                        f"{rate:.0f} tok/s, eta {eta/60:.1f} min",
                        logfile,
                    )
                    t_log = now
            expected = sum(map(len, token_lists))
            pass_audit.append(
                {
                    "index": pass_index,
                    "name": pass_name,
                    "samples": len(token_lists),
                    "tokens": expected,
                }
            )
            log(
                f"pass {pass_index + 1}/{len(passes)} {pass_name}: {expected} tokens, "
                f"cumulative={cumulative}, {time.time()-t_pass:.1f}s",
                logfile,
            )
    if cumulative != total_tokens:
        raise RuntimeError(f"captured {cumulative} tokens != plan {total_tokens}")
    return pass_audit


def write_manifests(
    out_dir: Path,
    plan: dict,
    state,
    pass_audit: list[dict],
    tokens: int,
    layers: list[int],
    args,
    logfile: str,
) -> dict:
    import torch
    import transformers

    run_uuid = str(uuid.uuid4())
    for layer in layers:
        capture = state["captures"][layer]
        x_path = capture.dir / "x.bin"
        ids_path = capture.dir / "ids.bin"
        if capture.tokens != tokens:
            raise RuntimeError(f"layer {layer}: finalized token count mismatch")
        if x_path.stat().st_size != tokens * HIDDEN * 2:
            raise RuntimeError(f"layer {layer}: x payload size mismatch")
        if ids_path.stat().st_size != tokens * TOPK:
            raise RuntimeError(f"layer {layer}: ids payload size mismatch")
        routed = [int(value) for value in capture.routed]
        if len(routed) != NUM_EXPERTS or sum(routed) != tokens * TOPK:
            raise RuntimeError(f"layer {layer}: routed count audit failed")
        manifest = {
            "schema": "glm52-b300-layer-capture-v1",
            "layer": layer,
            "capture_fingerprint": plan["capture_fingerprint"],
            "capture_run_uuid": run_uuid,
            "capture_tp": CAPTURE_TP,
            "owner_rank": 0,
            "tokens": tokens,
            "hidden": HIDDEN,
            "x_dtype": "bfloat16",
            "x_bytes": tokens * HIDDEN * 2,
            "ids_topk": TOPK,
            "ids_bytes": tokens * TOPK,
            "sha256_x": capture.hx.hexdigest(),
            "sha256_ids": capture.hi.hexdigest(),
            "routed_counts": routed,
            "routed_min": min(routed),
            "routed_max": max(routed),
            "cold_experts_lt1024": [expert for expert, count in enumerate(routed) if count < 1024],
            "finished": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        atomic_json(capture.dir / "layer_manifest.json", manifest)

    verify = state["verify"]
    if verify["checked_tokens"]:
        fraction = 1.0 - verify["mismatch_tokens"] / verify["checked_tokens"]
        if fraction < 0.99:
            raise RuntimeError(f"routing recompute vs model topk agreement only {fraction:.6f}")

    run_manifest = {
        "schema": "glm52-b300-capture-run-v1",
        "capture_engine": "transformers-5.14",
        "capture_fingerprint": plan["capture_fingerprint"],
        "capture_run_uuid": run_uuid,
        "active_layers": sorted(layers),
        "tokens_per_layer": tokens,
        "capture_payload_bytes": tokens * len(layers) * (HIDDEN * 2 + TOPK),
        "pass_audit": pass_audit,
        "rank_assignment": {str(layer): 0 for layer in sorted(layers)},
        "verify_routing": dict(verify),
        "capture_tp": CAPTURE_TP,
        "output_tp": OUTPUT_TP,
        "enforce_eager": True,
        "enable_prefix_caching": False,
        "max_tokens": 0,
        "prefill_only": True,
        "cpu_offload_gb": 0,
        "low_memory_mode": False,
        "model_snapshot": str(args.src),
        "corpus_sha256": plan["corpus_sha256"],
        "attn_implementation": args.attn_impl,
        "experts_implementation": args.experts_impl,
        "batching": "one sample per forward (batch=1, no padding), exact plan order",
        "row_order_note": (
            "rows are appended in plan order (all tokens of sample 1, then sample 2, ...). "
            "This can differ from the vLLM engine's internal scheduling order; the downstream "
            "encoder accumulates order-independent Hessian sums (H += x^T x per expert), so "
            "only the set of (token, x, routing) rows and the per-layer counts are contractual."
        ),
        "verify_routing_note": (
            "verify_routing compares the reference fp32 gate recompute (sigmoid(x@W^T)+bias, "
            "top-8) against the transformers router's own topk indices at every captured token; "
            "kwarg_logits_layers counts hook invocations where the model's indices were visible."
        ),
        "transformers": transformers.__version__,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
        "finished": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    atomic_json(out_dir / "capture_run_manifest.json", run_manifest)
    log(
        f"CAPTURE SEALED: {len(layers)} layer(s), {tokens} tokens/layer, "
        f"{run_manifest['capture_payload_bytes']/2**30:.2f} GiB payload, "
        f"fingerprint={plan['capture_fingerprint']}",
        logfile,
    )
    return run_manifest


def validate_capture(final_dir: Path, state, tokens: int, layers: list[int], logfile: str) -> dict:
    """Post-seal validation: sizes, one full sha256 recompute, routed sums, id spot checks."""
    import numpy as np
    import torch

    results = {"layers_checked": [], "sha_recompute": {}, "spot_checks": []}
    for layer in layers:
        layer_dir = final_dir / f"layer_{layer:03d}"
        manifest = json.loads((layer_dir / "layer_manifest.json").read_text())
        if int(manifest["tokens"]) != tokens:
            raise RuntimeError(f"VALIDATE layer {layer}: tokens {manifest['tokens']} != {tokens}")
        if (layer_dir / "x.bin").stat().st_size != tokens * HIDDEN * 2:
            raise RuntimeError(f"VALIDATE layer {layer}: x.bin size mismatch")
        if (layer_dir / "ids.bin").stat().st_size != tokens * TOPK:
            raise RuntimeError(f"VALIDATE layer {layer}: ids.bin size mismatch")
        routed = manifest["routed_counts"]
        if len(routed) != NUM_EXPERTS or sum(routed) != tokens * TOPK:
            raise RuntimeError(f"VALIDATE layer {layer}: routed_counts sum != tokens*8")
        results["layers_checked"].append(layer)

    # (b) full sha256 recompute on the first and last captured layers.
    for layer in (layers[0], layers[-1]):
        layer_dir = final_dir / f"layer_{layer:03d}"
        manifest = json.loads((layer_dir / "layer_manifest.json").read_text())
        for name, key in (("x.bin", "sha256_x"), ("ids.bin", "sha256_ids")):
            actual = sha256_file(layer_dir / name)
            if actual != manifest[key]:
                raise RuntimeError(
                    f"VALIDATE layer {layer}: {name} sha256 {actual} != manifest {manifest[key]}"
                )
            results["sha_recompute"][f"layer_{layer:03d}/{name}"] = actual
    log("VALIDATE: sha256 recompute matches manifests", logfile)

    # (d) spot-check routing ids against a manual gate recompute from captured x rows.
    rng = random.Random(1234)
    check_rows = 64
    tie_tolerance = 1e-5
    for layer in layers:
        gate_w = state["gate_w"][layer]
        gate_b = state["gate_b"][layer]
        layer_dir = final_dir / f"layer_{layer:03d}"
        rows = sorted(rng.sample(range(tokens), check_rows))
        exact = 0
        near_tie = 0
        with open(layer_dir / "x.bin", "rb") as fx, open(layer_dir / "ids.bin", "rb") as fi:
            for row in rows:
                fx.seek(row * HIDDEN * 2)
                x_np = np.frombuffer(fx.read(HIDDEN * 2), dtype=np.int16).copy()
                fi.seek(row * TOPK)
                ids_disk = np.frombuffer(fi.read(TOPK), dtype=np.uint8)
                x = torch.from_numpy(x_np).view(torch.bfloat16).to("cuda:0")
                logits = torch.nn.functional.linear(x.to(torch.float32).unsqueeze(0), gate_w)
                scores = (torch.sigmoid(logits) + gate_b).squeeze(0)
                ids_re = torch.topk(scores, TOPK, dim=-1, sorted=False).indices.cpu().numpy()
                if set(ids_re.tolist()) == set(ids_disk.tolist()):
                    exact += 1
                else:
                    # near-tie tolerance: disputed experts must sit within tie_tolerance
                    # of the top-8 score boundary (fp32 GEMM shape differs from capture).
                    sorted_scores = torch.sort(scores, descending=True).values
                    boundary = sorted_scores[TOPK - 1].item()
                    disputed = set(ids_disk.tolist()) ^ set(ids_re.tolist())
                    gaps = [abs(scores[e].item() - boundary) for e in disputed]
                    if max(gaps) <= tie_tolerance:
                        near_tie += 1
                    else:
                        raise RuntimeError(
                            f"VALIDATE layer {layer} row {row}: ids mismatch beyond tie "
                            f"tolerance: disk={sorted(ids_disk.tolist())} "
                            f"recompute={sorted(ids_re.tolist())} max_gap={max(gaps)}"
                        )
        results["spot_checks"].append(
            {"layer": layer, "rows": check_rows, "exact": exact, "near_tie": near_tie}
        )
        log(
            f"VALIDATE layer {layer}: {exact}/{check_rows} spot rows exact top-8 match, "
            f"{near_tie} within tie tolerance",
            logfile,
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Transformers-based GLM-5.2 SIQ-Fruit BF16 capture")
    parser.add_argument("--src", default=DEFAULT_SRC, type=Path)
    parser.add_argument("--corpus", default=DEFAULT_CORPUS, type=Path)
    parser.add_argument("--plan-file", default="/home/mbelleau/fq-0c/capture_plan.json", type=Path)
    parser.add_argument("--capture-dir", default="/dev/shm/fruit-capture", type=Path)
    parser.add_argument("--seal-dir", default="/home/mbelleau/fq-0c/capture", type=Path)
    parser.add_argument("--layers", default="3-12")
    parser.add_argument("--attn-impl", default="sdpa", choices=["sdpa", "eager"])
    parser.add_argument(
        "--experts-impl",
        default="grouped_mm",
        choices=["eager", "grouped_mm", "batched_mm", "deepgemm", "sonicmoe"],
    )
    parser.add_argument(
        "--smoke",
        type=int,
        default=0,
        help="capture only the first N samples of each pass into --capture-dir; no sealing",
    )
    parser.add_argument("--no-seal", action="store_true", help="leave the capture in --capture-dir")
    parser.add_argument("--log", default="/home/mbelleau/fq-0c/capture_hf.capture.log")
    args = parser.parse_args()

    args.src = args.src.resolve()
    args.corpus = args.corpus.resolve()
    args.plan_file = args.plan_file.resolve()
    logfile = args.log

    plan = json.loads(args.plan_file.read_text())
    validate_plan(plan, args.src, args.corpus)
    layers = parse_layers(args.layers)
    log(
        f"plan validated: fingerprint={plan['capture_fingerprint']} "
        f"total_tokens={plan['total_tokens']} layers={layers}",
        logfile,
    )

    passes = load_plan_tokens(plan, args.src, args.corpus, logfile)
    if args.smoke:
        passes = [(name, token_lists[: args.smoke]) for name, token_lists in passes]
        total_tokens = sum(sum(map(len, tl)) for _, tl in passes)
        log(f"SMOKE MODE: {args.smoke} samples/pass, {total_tokens} tokens; no sealing", logfile)
    else:
        total_tokens = int(plan["total_tokens"])

    payload_bytes = total_tokens * len(layers) * (HIDDEN * 2 + TOPK)
    free = shutil.disk_usage(args.capture_dir.parent).free
    if free < payload_bytes + (8 << 30):
        raise RuntimeError(
            f"capture target {args.capture_dir.parent}: {free} bytes free < payload "
            f"{payload_bytes} + 8 GiB reserve"
        )

    if args.capture_dir.exists():
        shutil.rmtree(args.capture_dir)
    args.capture_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(args.src, args.attn_impl, args.experts_impl, logfile)
    state = install_hooks(model, layers, args.capture_dir, logfile)
    pass_audit = run_forwards(model, state, passes, layers, total_tokens, logfile)

    for handle in state["handles"]:
        handle.remove()
    for capture in state["captures"].values():
        capture.finalize()
    verify = state["verify"]
    log(
        f"routing verify vs model topk: checked={verify['checked_tokens']} "
        f"mismatch={verify['mismatch_tokens']} hook_invocations={verify['kwarg_logits_layers']}",
        logfile,
    )
    write_manifests(args.capture_dir, plan, state, pass_audit, total_tokens, layers, args, logfile)

    if args.smoke or args.no_seal:
        final_dir = args.capture_dir
        log(f"capture left unsealed at {final_dir}", logfile)
    else:
        final_dir = args.seal_dir.resolve()
        if final_dir.exists():
            shutil.rmtree(final_dir)
        t0 = time.time()
        shutil.move(str(args.capture_dir), str(final_dir))
        log(f"sealed capture moved to {final_dir} in {time.time()-t0:.1f}s", logfile)

    results = validate_capture(final_dir, state, total_tokens, layers, logfile)
    log(f"VALIDATION PASSED: {json.dumps(results['spot_checks'])}", logfile)

    del model
    gc.collect()
    log("done", logfile)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
        sys.exit(1)
