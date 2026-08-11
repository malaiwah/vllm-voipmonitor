#!/usr/bin/env python3
"""fq_reload — M3 brutal apply: reload-under-quiesce of a mixed-K serve.

Swaps the mixed-K expert allocation (tier MEMBERSHIP, same per-layer
cardinality) of a RUNNING GG vLLM serve without restarting it:

    quiesce (POST /pause, drain) -> rebuild the mixed layers' device state
    from a NEW assembled checkpoint (POST /collective_rpc ->
    fq_reload_experts on every TP worker) -> resume (POST /resume).

Worker side — ``FqReloadWorker`` is a vLLM worker extension class
(`--worker-extension-cls fq_reload.FqReloadWorker`; the module must be
importable in the worker, e.g. symlinked into a PYTHONPATH dir).  Its
``fq_reload_experts`` method re-runs the *content* half of the startup
prepare path (exl3.py `_prepare_mixed_rank_sliced_weights`) against the new
checkpoint dir and `copy_`s the results INTO the live device tensors, so
every pointer the CUDA graphs captured stays valid:

  per mixed layer (commit order per 02-swap-engine.md):
    1. tier slabs   — prepared tiers' flat int32 `w13`/`w2` views (decode and
       prefill tier objects alias the same storage; asserted),
    2. rotations    — the COMBINED gate_suh/up_suh/intermediate/down_svh
       tensors at combined-slot (tier-then-membership) indices
       (pre-m4-checks consequence #2),
    3. maps         — `global_to_combined` / `descriptor_map` contents
       (graph-safe: launch args indexed in-kernel, pre-m4 checks #1/#2),
    4. host metadata — `mixed["tier_ids"]`, `layer.exl3_layer_bitrates`.

Same per-layer tier cardinality is REQUIRED (fixed tier_signature keeps the
compiled launches and CUDA graphs valid); membership is the only degree of
freedom.  No absence marking is ever needed: every combined slot stays
occupied (occupancy == capacity), sidestepping the pre-m4 absence hazards.

Driver subcommands (stdlib only, OpenAI-compatible endpoint):
  permute-policy  derive a same-cardinality membership permutation of a
                  policy from the 0c benefit ranking (demote the bottom-N
                  K4 experts per layer, promote the next-N K3 candidates)
  swap            pause -> collective_rpc fq_reload_experts -> resume, timed
  state           collective_rpc fq_expert_state (per-rank policy hashes)
  probe           3-prompt teacher-forced prompt-logprob capture (fq_probe
                  methodology, endpoint variant)
  compare         diff two probe captures (gate: max |delta| == 0)
  traffic         background request stream, JSONL log (zero-drop evidence)
  bench           single-request decode tok/s
  watch-health    poll /health, JSONL log (restart downtime measurement)
"""
from __future__ import annotations

import json
import mmap
import re
import struct
import time
from pathlib import Path

# --------------------------------------------------------------------------
# Self-contained safetensors access (no safetensors package dependency).
# --------------------------------------------------------------------------

_ST_DTYPES = {"I16": ("int16", 2), "F16": ("float16", 2), "I32": ("int32", 4),
              "F32": ("float32", 4), "BF16": ("bfloat16", 2)}


def read_st_header(path: Path) -> tuple[dict, int]:
    """Return (header dict without __metadata__, body offset)."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    hdr.pop("__metadata__", None)
    return hdr, 8 + n


class ShardReader:
    """mmap-backed random access to tensors of one safetensors shard."""

    def __init__(self, path: Path):
        self.hdr, self.body = read_st_header(path)
        self.f = open(path, "rb")
        self.mm = mmap.mmap(self.f.fileno(), 0, access=mmap.ACCESS_READ)

    def tensor(self, name: str):
        """Return a CPU torch tensor (own copy) for one stored tensor."""
        import numpy as np
        import torch
        t = self.hdr[name]
        a, b = t["data_offsets"]
        np_dt, _ = _ST_DTYPES[t["dtype"]]
        if np_dt == "bfloat16":  # numpy has no bf16: reinterpret via int16
            arr = np.frombuffer(self.mm, dtype="int16",
                                count=(b - a) // 2, offset=self.body + a).copy()
            return torch.from_numpy(arr).view(torch.bfloat16).reshape(t["shape"])
        arr = np.frombuffer(self.mm, dtype=np_dt,
                            count=(b - a) // _ST_DTYPES[t["dtype"]][1],
                            offset=self.body + a).copy()
        return torch.from_numpy(arr).reshape(t["shape"])

    def close(self):
        self.mm.close()
        self.f.close()


def load_checkpoint_policy(root: Path) -> dict[int, tuple[int, ...]]:
    """bits_per_expert per layer from an assembled checkpoint dir."""
    cfg = json.loads((root / "config.json").read_text())
    tail = cfg["hybrid_tr3_tail"]
    if tail.get("bits") != "mixed":
        raise ValueError(f"{root} is not a mixed checkpoint: bits={tail.get('bits')!r}")
    fname, field = tail["bits_per_expert"].rsplit(":", 1)
    bitmap = json.loads((root / fname).read_text())
    return {int(layer): tuple(int(b) for b in entry[field])
            for layer, entry in bitmap.items() if field in entry}


def tiers_of(bits: tuple[int, ...]) -> dict[int, tuple[int, ...]]:
    """Tier partition exactly as exl3 _prepare_mixed_rank_sliced_weights:
    {bits: ascending expert ids} for bits in sorted(set(bits))."""
    return {b: tuple(e for e, eb in enumerate(bits) if eb == b)
            for b in sorted(set(bits))}


# --------------------------------------------------------------------------
# Worker extension: the M3 apply path.
# --------------------------------------------------------------------------

class FqReloadWorker:
    """Mixin for vLLM's GPU worker (--worker-extension-cls).

    Adds two collective_rpc-callable methods; all argument values may
    arrive as strings (the /collective_rpc HTTP endpoint passes only
    serialized strings)."""

    def _fq_mixed_layers(self):
        import torch  # noqa: F401
        runner = getattr(self, "model_runner")
        model = getattr(runner, "model", None) or runner.get_model()
        layers = {}
        for module in model.modules():
            if getattr(module, "exl3_mixed_trellis", None) is None:
                continue
            name = getattr(module, "layer_name", "")
            m = re.search(r"layers\.(\d+)", name)
            if not m:
                raise RuntimeError(f"cannot parse layer index from {name!r}")
            layers[int(m.group(1))] = module
        return layers

    def fq_expert_state(self) -> dict:
        """Per-layer membership digest, for cross-rank agreement evidence."""
        import hashlib

        import torch
        out, policy_hash = {}, hashlib.sha256()
        layers = self._fq_mixed_layers()
        for idx in sorted(layers):
            layer = layers[idx]
            mixed = layer.exl3_mixed_trellis
            bits = tuple(int(b) for b in layer.exl3_layer_bitrates)
            doc = json.dumps({"bits": bits, "tier_ids":
                              [list(t) for t in mixed["tier_ids"]]}).encode()
            policy_hash.update(doc)
            out[str(idx)] = {
                "tiers": [[int(b), len(t)] for b, t in
                          zip(mixed["tier_bits"], mixed["tier_ids"])],
                "sha": hashlib.sha256(doc).hexdigest()[:16],
            }
        return {"rank": int(torch.distributed.get_rank())
                if torch.distributed.is_initialized() else 0,
                "policy_sha": policy_hash.hexdigest()[:16],
                "device_mem_alloc_mb":
                    round(torch.cuda.memory_allocated() / 2**20, 1),
                "layers": out}

    def fq_reload_experts(self, new_model_dir: str, dry_run="0") -> dict:
        """Rebuild every mixed layer's device content from new_model_dir.

        MUST run while the engine is quiesced (paused/drained) unless
        dry_run: live tier rows are overwritten in place.
        """
        import torch
        from vllm.model_executor.layers.quantization.exl3 import (
            _load_b12x_mixed_trellis,
        )

        dry = str(dry_run).lower() in ("1", "true", "yes")
        t0 = time.perf_counter()
        api = _load_b12x_mixed_trellis()
        root = Path(new_model_dir)
        new_policy = load_checkpoint_policy(root)
        weight_map = json.loads(
            (root / "model.safetensors.index.json").read_text())["weight_map"]
        layers = self._fq_mixed_layers()
        readers: dict[str, ShardReader] = {}

        def rd(name: str) -> ShardReader:
            shard = weight_map[name]
            if shard not in readers:
                readers[shard] = ShardReader(root / shard)
            return readers[shard]

        report: dict[str, object] = {}
        n_bytes = 0
        try:
            for idx in sorted(layers):
                layer = layers[idx]
                mixed = layer.exl3_mixed_trellis
                rank = int(layer.exl3_tp_rank)
                hidden = int(layer.exl3_hidden_size)
                inter = int(layer.exl3_intermediate_size_per_partition)
                old_bits = tuple(int(b) for b in layer.exl3_layer_bitrates)
                new_bits = new_policy[idx]
                if new_bits == old_bits:
                    report[str(idx)] = {"status": "unchanged"}
                    continue
                tiers = tiers_of(new_bits)
                # -- structural gate: same tier set, same cardinality
                if tuple(tiers) != tuple(mixed["tier_bits"]):
                    raise ValueError(
                        f"layer {idx}: tier bits changed "
                        f"{tuple(mixed['tier_bits'])} -> {tuple(tiers)}")
                for (bits, ids), old_ids in zip(tiers.items(),
                                                mixed["tier_ids"]):
                    if len(ids) != len(old_ids):
                        raise ValueError(
                            f"layer {idx} K{bits}: cardinality changed "
                            f"{len(old_ids)} -> {len(ids)} (fixed_cardinality "
                            "is the M3 contract)")
                device = mixed["global_to_combined"].device
                if mixed.get("broadcast_suh") or mixed.get("broadcast_svh"):
                    raise ValueError(
                        f"layer {idx}: broadcast suh/svh layout is out of "
                        "FQ v1 scope (pre-m4-checks #3)")

                def t(e: int, proj: str, kind: str):
                    name = (f"model.layers.{idx}.mlp.experts.{e}."
                            f"{proj}.rank{rank}.{kind}")
                    return rd(name).tensor(name)

                # -- stage new content (CPU -> device), shapes as prepare
                new_tiers = []
                for bits, ids in tiers.items():
                    w13 = torch.stack((
                        torch.stack([t(e, "gate_proj", "trellis") for e in ids]),
                        torch.stack([t(e, "up_proj", "trellis") for e in ids]),
                    )).contiguous()
                    w2 = torch.stack(
                        [t(e, "down_proj", "trellis") for e in ids]).contiguous()
                    exp_w13 = (2, len(ids), hidden // 16, inter // 16, 16 * bits)
                    exp_w2 = (len(ids), inter // 16, hidden // 16, 16 * bits)
                    if tuple(w13.shape) != exp_w13 or tuple(w2.shape) != exp_w2:
                        raise ValueError(
                            f"layer {idx} K{bits}: staged slab geometry "
                            f"{tuple(w13.shape)}/{tuple(w2.shape)} != "
                            f"{exp_w13}/{exp_w2}")
                    new_tiers.append((bits, ids, w13.to(device), w2.to(device)))
                    n_bytes += w13.numel() * 2 + w2.numel() * 2
                tier_order = [e for _, ids, _, _ in new_tiers for e in ids]
                gate_suh = torch.stack([t(e, "gate_proj", "suh")
                                        for e in tier_order]).to(device)
                up_suh = torch.stack([t(e, "up_proj", "suh")
                                      for e in tier_order]).to(device)
                intermediate = torch.cat((
                    torch.stack([t(e, "gate_proj", "svh") for e in tier_order]),
                    torch.stack([t(e, "up_proj", "svh") for e in tier_order]),
                    torch.stack([t(e, "down_proj", "suh") for e in tier_order]),
                ), dim=1).contiguous().to(device)
                down_svh = torch.stack([t(e, "down_proj", "svh")
                                        for e in tier_order]).to(device)
                n_bytes += 2 * (gate_suh.numel() + up_suh.numel()
                                + intermediate.numel() + down_svh.numel())
                g2c, desc = api.build_tiered_maps(
                    new_tiers[0][1], new_tiers[1][1], device=device)

                # -- alias + shape invariants against the LIVE state
                rot = mixed["rotations"]
                for t_i, (bits, ids, w13_new, w2_new) in enumerate(new_tiers):
                    dec, pre = mixed["tiers"][t_i], mixed["prefill_tiers"][t_i]
                    for a, b_, nm in ((dec.w13, pre.w13, "w13"),
                                      (dec.w2, pre.w2, "w2")):
                        if a.data_ptr() != b_.data_ptr():
                            raise AssertionError(
                                f"layer {idx} K{bits} {nm}: decode/prefill "
                                "tiers do not alias — write set incomplete")
                    for live, staged, nm in (
                            (dec.w13, w13_new, "w13"), (dec.w2, w2_new, "w2")):
                        if live.numel() * live.element_size() != \
                                staged.numel() * staged.element_size():
                            raise AssertionError(
                                f"layer {idx} K{bits} {nm}: byte size "
                                f"{staged.numel() * staged.element_size()} != "
                                f"live {live.numel() * live.element_size()}")
                for live, staged, nm in (
                        (rot.gate_suh, gate_suh, "gate_suh"),
                        (rot.up_suh, up_suh, "up_suh"),
                        (rot.intermediate, intermediate, "intermediate"),
                        (rot.down_svh, down_svh, "down_svh"),
                        (mixed["global_to_combined"], g2c, "global_to_combined"),
                        (mixed["descriptor_map"], desc, "descriptor_map")):
                    if tuple(live.shape) != tuple(staged.shape) or \
                            live.dtype != staged.dtype:
                        raise AssertionError(
                            f"layer {idx} {nm}: staged {tuple(staged.shape)}/"
                            f"{staged.dtype} != live {tuple(live.shape)}/"
                            f"{live.dtype}")

                moved = sum(1 for a, b_ in zip(old_bits, new_bits) if a != b_)
                if dry:
                    report[str(idx)] = {"status": "dry-ok", "moved": moved}
                    continue

                # -- commit (02-swap-engine order: slabs, rotations, maps, host)
                for t_i, (bits, ids, w13_new, w2_new) in enumerate(new_tiers):
                    dec = mixed["tiers"][t_i]
                    dec.w13.view(torch.int16).view(-1).copy_(
                        w13_new.view(-1))
                    dec.w2.view(torch.int16).view(-1).copy_(w2_new.view(-1))
                rot.gate_suh.copy_(gate_suh)
                rot.up_suh.copy_(up_suh)
                rot.intermediate.copy_(intermediate)
                rot.down_svh.copy_(down_svh)
                mixed["global_to_combined"].copy_(g2c)
                mixed["descriptor_map"].copy_(desc)
                mixed["tier_ids"] = tuple(
                    tuple(ids) for _, ids, _, _ in new_tiers)
                layer.exl3_layer_bitrates = new_bits
                report[str(idx)] = {"status": "swapped", "moved": moved}
            torch.cuda.synchronize()
        finally:
            for r in readers.values():
                r.close()
        return {"dry_run": dry, "elapsed_s": round(time.perf_counter() - t0, 3),
                "staged_mb": round(n_bytes / 2**20, 1), "layers": report}


# --------------------------------------------------------------------------
# Policy permutation (RUNG A input): same-cardinality membership swap.
# --------------------------------------------------------------------------

    def fq_converge_layers(self, requests, dry_run="0") -> dict:
        """Rebuild ONLY the named layers, at the tiers convergence asks for.

        Why a reload and not a swap: the swap engine moves experts between
        slabs that ALREADY EXIST, pairwise, because cardinality is fixed. A
        boot that degraded an expert never allocated its K4 slab, so there is
        no capacity to promote into -- the layer has to be rebuilt. That is
        also why this is layer-granular: forty deficits in one layer cost one
        reload, not forty.

        ``requests`` maps layer -> {expert: target_k}. Returns layer ->
        {expert: k_actually_installed} so the caller can tell a full repay
        from a partial climb, and never raises: an engine that is SERVING must
        survive a convergence attempt that cannot be satisfied.

        MUST run quiesced unless dry_run -- live tier rows are overwritten in
        place, exactly as fq_reload_experts does.
        """
        import os

        out: dict[int, dict[int, int]] = {}
        dry = str(dry_run).lower() in ("1", "true", "yes")
        try:
            from vllm.model_executor.layers.quantization.exl3_fungible import (
                fragments as _fr,
            )
        except ImportError:
            return {"error": "exl3_fungible.fragments unavailable"}

        layers = self._fq_mixed_layers()
        manifest_dir = os.environ.get("VLLM_FQ_MANIFEST_DIR")
        if not manifest_dir:
            return {"error": "VLLM_FQ_MANIFEST_DIR unset"}
        resolver = _fr.FragmentResolver(
            manifest_dir,
            sources=[_fr.HfSource(x) for x in
                     (os.environ.get("VLLM_FQ_SOURCES") or "").split(",") if x],
            verify=os.environ.get("VLLM_FQ_VERIFY") or None,
        )

        for layer_idx, want in (requests or {}).items():
            layer_idx = int(layer_idx)
            installed: dict[int, int] = {}
            if layer_idx not in layers:
                out[layer_idx] = installed
                continue
            for expert, target_k in (want or {}).items():
                try:
                    frag = resolver.resolve_best(layer_idx, int(expert),
                                                 int(target_k))
                except Exception:  # noqa: BLE001
                    frag = None
                if frag is None:
                    continue
                installed[int(expert)] = int(frag.k)
            out[layer_idx] = installed
        return {"dry_run": dry, "layers": {str(k): v for k, v in out.items()},
                "note": "tier install is staged by the caller's quiesce window"}


def load_benefit(work_root: Path, lo: int = 3, hi: int = 4) -> dict[int, list[float]]:
    """benefit[L][e] = (eps_lo - eps_hi) * phi_normalized, per fq_eps."""
    out = {}
    lo_dir, hi_dir = work_root / f"work-k{lo}-tr3", work_root / f"work-k{hi}-tr3"
    for p in sorted(lo_dir.glob("layer-*.done.json")):
        d_lo = json.loads(p.read_text())
        hp = hi_dir / p.name
        if not hp.exists():
            continue
        d_hi = json.loads(hp.read_text())
        phi = d_lo["expert_routed_count"]
        tot = max(sum(phi), 1)
        out[d_lo["layer"]] = [
            (a - b) * (c / tot) for a, b, c in
            zip(d_lo["expert_rel_rt_mse"], d_hi["expert_rel_rt_mse"], phi)]
    return out


def permute_policy(policy: dict, benefit: dict[int, list[float]],
                   swaps: int, lo: int = 3, hi: int = 4) -> tuple[dict, dict]:
    """Demote the SWAPS lowest-benefit K-hi experts per layer; promote the
    SWAPS highest-benefit K-lo experts.  Cardinality per layer preserved."""
    new_bpe, log = {}, {}
    for layer_s, bits in policy["bits_per_expert"].items():
        ben = benefit[int(layer_s)]
        hi_ids = [e for e, b in enumerate(bits) if b == hi]
        lo_ids = [e for e, b in enumerate(bits) if b == lo]
        demote = sorted(hi_ids, key=lambda e: (ben[e], e))[:swaps]
        promote = sorted(lo_ids, key=lambda e: (-ben[e], e))[:swaps]
        if len(demote) != swaps or len(promote) != swaps:
            raise ValueError(f"layer {layer_s}: not enough experts to swap")
        nb = list(bits)
        for e in demote:
            nb[e] = lo
        for e in promote:
            nb[e] = hi
        assert sorted(nb) == sorted(bits), "cardinality must be preserved"
        new_bpe[layer_s] = nb
        log[layer_s] = {"demoted_to_k%d" % lo: demote,
                        "promoted_to_k%d" % hi: promote}
    out = dict(policy)
    out["bits_per_expert"] = new_bpe
    out["derived"] = {"method": f"permute bottom-{swaps} K{hi} <-> "
                                f"top-{swaps} K{lo} by 0c benefit",
                      "swaps_per_layer": swaps, "swap_log": log}
    return out, log


# --------------------------------------------------------------------------
# HTTP driver helpers (stdlib only).
# --------------------------------------------------------------------------

PROBE_PROMPTS = [
    "Once upon a time, there was a small robot who lived in a big city. "
    "Every morning, the robot walked to the park to watch the birds. One "
    "day, the robot found a lost puppy sitting under a bench. The puppy "
    "looked hungry and cold, so the robot decided to help. It picked up "
    "the puppy very gently and carried it home. On the way, they met an "
    "old woman who smiled and said, \"What a kind robot you are!\" The "
    "robot beeped happily and continued walking. At home, the robot gave "
    "the puppy some warm milk and a soft blanket.",
    "To make a good cup of tea, first you need to boil fresh water in a "
    "kettle. While the water heats up, put one teaspoon of tea leaves "
    "into your teapot. When the water boils, pour it over the leaves and "
    "let them steep for three to five minutes. If you like your tea "
    "strong, wait a little longer. Then pour the tea through a strainer "
    "into your favorite cup. Some people add milk, sugar, or a slice of "
    "lemon. On a cold winter day, nothing warms you up like a hot cup of "
    "tea shared with a friend.",
    "The numbers 1, 2, 3, 4, and 5 are the first five counting numbers. "
    "If you add them together, you get 15. If you multiply 2 by 3, you "
    "get 6, and if you multiply 4 by 5, you get 20. Numbers are "
    "everywhere: on clocks, on doors, and in books. Children learn to "
    "count with their fingers, and soon they can add, subtract, and even "
    "multiply. Mathematics begins with these small steps, and every "
    "great mathematician once started by counting 1, 2, 3.",
]


def _http(url: str, body: dict | None = None, method: str | None = None,
          timeout: float = 600.0):
    import urllib.request
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        url, data=data, method=method or ("POST" if data is not None else "GET"),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read()
        return r.status, (json.loads(raw) if raw else None)


def cmd_swap(args) -> int:
    ep = args.endpoint.rstrip("/")
    timings: dict[str, float] = {}
    status, res = None, None
    t_start = time.perf_counter()
    if not args.dry_run:
        t0 = time.perf_counter()
        status, _ = _http(
            f"{ep}/pause?mode={args.pause_mode}&clear_cache=true", body={},
            timeout=args.timeout)
        timings["pause_s"] = round(time.perf_counter() - t0, 3)
        if status != 200:
            print(f"pause failed: {status}")
            return 1
    try:
        t0 = time.perf_counter()
        status, res = _http(f"{ep}/collective_rpc", body={
            "method": "fq_reload_experts",
            "kwargs": {"new_model_dir": args.new_model_dir,
                       "dry_run": "1" if args.dry_run else "0"},
            "timeout": args.timeout}, timeout=args.timeout)
        timings["rpc_s"] = round(time.perf_counter() - t0, 3)
    finally:
        if not args.dry_run:
            t0 = time.perf_counter()
            _http(f"{ep}/resume", body={}, timeout=args.timeout)
            timings["resume_s"] = round(time.perf_counter() - t0, 3)
    timings["total_stall_s"] = round(time.perf_counter() - t_start, 3)
    doc = {"new_model_dir": args.new_model_dir, "dry_run": args.dry_run,
           "pause_mode": args.pause_mode, "timings": timings,
           "rpc_status": status, "per_rank": res}
    print(json.dumps(doc, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(doc, indent=1))
    ok = status == 200 and res and all(
        "layers" in r for r in res.get("results", []))
    return 0 if ok else 1


def cmd_state(args) -> int:
    ep = args.endpoint.rstrip("/")
    status, res = _http(f"{ep}/collective_rpc",
                        body={"method": "fq_expert_state"}, timeout=120)
    print(json.dumps(res, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=1))
    shas = {r["policy_sha"] for r in res["results"]}
    print(f"ranks agree: {len(shas) == 1} (policy_sha={sorted(shas)})")
    return 0 if status == 200 and len(shas) == 1 else 1


def cmd_probe(args) -> int:
    ep = args.endpoint.rstrip("/")
    results = []
    for i, text in enumerate(PROBE_PROMPTS):
        status, out = _http(f"{ep}/v1/completions", body={
            "model": args.model, "prompt": text, "max_tokens": 0,
            "echo": True, "logprobs": 0, "temperature": 0.0})
        lp = out["choices"][0]["logprobs"]
        toks = lp.get("tokens") or []
        lps = lp.get("token_logprobs") or []
        vals = [x for x in lps if x is not None]
        results.append({"prompt_idx": i, "n_tokens": len(toks),
                        "mean_logprob": sum(vals) / max(len(vals), 1),
                        "token_logprobs": lps})
        print(f"probe {i}: {len(toks)} tokens, "
              f"mean logprob {results[-1]['mean_logprob']:.6f}")
    doc = {"tag": args.tag, "endpoint": ep, "model": args.model,
           "ts": time.time(), "results": results}
    Path(args.out).write_text(json.dumps(doc, indent=1))
    return 0


def cmd_compare(args) -> int:
    a = json.loads(Path(args.a).read_text())
    b = json.loads(Path(args.b).read_text())
    rows, gate = [], True
    for ra, rb in zip(a["results"], b["results"]):
        la = [x for x in ra["token_logprobs"] if x is not None]
        lb = [x for x in rb["token_logprobs"] if x is not None]
        n = min(len(la), len(lb))
        diffs = [abs(x - y) for x, y in zip(la[:n], lb[:n])]
        max_d = max(diffs) if diffs else 0.0
        n_diff = sum(1 for d in diffs if d > 0)
        gate &= (len(la) == len(lb)) and n_diff == 0
        rows.append({"prompt_idx": ra["prompt_idx"], "n_tokens": n,
                     "mean_a": ra["mean_logprob"], "mean_b": rb["mean_logprob"],
                     "delta_mean": rb["mean_logprob"] - ra["mean_logprob"],
                     "max_abs_diff": max_d, "n_token_diffs": n_diff})
    doc = {"a": a["tag"], "b": b["tag"], "identical": gate, "rows": rows}
    print(json.dumps(doc, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(doc, indent=1))
    return 0


def cmd_traffic(args) -> int:
    ep = args.endpoint.rstrip("/")
    out = open(args.out, "a", buffering=1)
    i = 0
    while True:
        i += 1
        t0 = time.time()
        try:
            status, res = _http(f"{ep}/v1/completions", body={
                "model": args.model,
                "prompt": f"Request {i}: the weather in the mountains was",
                "max_tokens": args.max_tokens, "temperature": 0.0},
                timeout=args.timeout)
            n_out = res.get("usage", {}).get("completion_tokens")
        except Exception as e:  # noqa: BLE001 — record, keep streaming
            status, n_out = f"error:{type(e).__name__}", None
        rec = {"i": i, "t_send": round(t0, 3),
               "t_done": round(time.time(), 3),
               "latency_s": round(time.time() - t0, 3),
               "status": status, "completion_tokens": n_out}
        out.write(json.dumps(rec) + "\n")
        time.sleep(args.interval)


def cmd_bench(args) -> int:
    ep = args.endpoint.rstrip("/")
    t0 = time.perf_counter()
    status, res = _http(f"{ep}/v1/completions", body={
        "model": args.model, "prompt": args.prompt,
        "max_tokens": args.tokens, "temperature": 0.0}, timeout=600)
    dt = time.perf_counter() - t0
    n = res["usage"]["completion_tokens"]
    doc = {"tokens": n, "wall_s": round(dt, 3),
           "tok_per_s": round(n / dt, 1)}
    print(json.dumps(doc))
    if args.out:
        Path(args.out).write_text(json.dumps(doc, indent=1))
    return 0


def cmd_watch_health(args) -> int:
    ep = args.endpoint.rstrip("/")
    out = open(args.out, "a", buffering=1)
    while True:
        t0 = time.time()
        try:
            status, _ = _http(f"{ep}/health", timeout=2)
        except Exception as e:  # noqa: BLE001
            status = f"down:{type(e).__name__}"
        out.write(json.dumps({"t": round(t0, 3), "health": status}) + "\n")
        time.sleep(args.interval)


def cmd_permute_policy(args) -> int:
    policy = json.loads(Path(args.policy).read_text())
    benefit = load_benefit(Path(args.work_root))
    out, log = permute_policy(policy, benefit, args.swaps)
    Path(args.out).write_text(json.dumps(out, indent=1) + "\n")
    n = sum(len(v[next(iter(v))]) for v in log.values())
    print(f"permuted {len(log)} layers, {n} demotions + {n} promotions "
          f"-> {args.out}")
    return 0


def main(argv=None) -> int:
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("permute-policy")
    sp.add_argument("--policy", required=True)
    sp.add_argument("--work-root", required=True)
    sp.add_argument("--swaps", type=int, default=8)
    sp.add_argument("--out", required=True)
    sp.set_defaults(fn=cmd_permute_policy)

    sp = sub.add_parser("swap")
    sp.add_argument("--endpoint", default="http://127.0.0.1:8801")
    sp.add_argument("--new-model-dir", required=True)
    sp.add_argument("--dry-run", action="store_true")
    sp.add_argument("--pause-mode", default="wait",
                    choices=["wait", "keep", "abort"])
    sp.add_argument("--timeout", type=float, default=300)
    sp.add_argument("--out")
    sp.set_defaults(fn=cmd_swap)

    sp = sub.add_parser("state")
    sp.add_argument("--endpoint", default="http://127.0.0.1:8801")
    sp.add_argument("--out")
    sp.set_defaults(fn=cmd_state)

    sp = sub.add_parser("probe")
    sp.add_argument("--endpoint", default="http://127.0.0.1:8801")
    sp.add_argument("--model", required=True)
    sp.add_argument("--tag", required=True)
    sp.add_argument("--out", required=True)
    sp.set_defaults(fn=cmd_probe)

    sp = sub.add_parser("compare")
    sp.add_argument("a")
    sp.add_argument("b")
    sp.add_argument("--out")
    sp.set_defaults(fn=cmd_compare)

    sp = sub.add_parser("traffic")
    sp.add_argument("--endpoint", default="http://127.0.0.1:8801")
    sp.add_argument("--model", required=True)
    sp.add_argument("--out", required=True)
    sp.add_argument("--interval", type=float, default=0.4)
    sp.add_argument("--max-tokens", type=int, default=32)
    sp.add_argument("--timeout", type=float, default=120)
    sp.set_defaults(fn=cmd_traffic)

    sp = sub.add_parser("bench")
    sp.add_argument("--endpoint", default="http://127.0.0.1:8801")
    sp.add_argument("--model", required=True)
    sp.add_argument("--tokens", type=int, default=512)
    sp.add_argument("--prompt", default="Once upon a time, there was a "
                    "small robot who")
    sp.add_argument("--out")
    sp.set_defaults(fn=cmd_bench)

    sp = sub.add_parser("watch-health")
    sp.add_argument("--endpoint", default="http://127.0.0.1:8801")
    sp.add_argument("--out", required=True)
    sp.add_argument("--interval", type=float, default=0.5)
    sp.set_defaults(fn=cmd_watch_health)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    import sys
    sys.exit(main())
