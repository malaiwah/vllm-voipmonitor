#!/usr/bin/env python3
"""fruit_encode_driver — run the sha-pinned GLM-5.2 TR3 encoder on SIQ-Fruit.

Imports encode_tr3_v31.py (the calibration_encoder bundle, sha e9a85a47…)
as a library and overrides its owner-pinned constants for the Fruit proxy,
following the exact pattern encode_b300.py used for the production build:
constants from config.json, SLICE recomputed, BF16 direct loader, prebuilt
SM120 exllamav3_ext (PYTHONPATH=/opt/exllamav3 equivalent inside gg-env),
self-spawning single-GPU workers (v31's own orchestrate_encode would
re-exec the raw script and lose overrides — never call it).

Modes: --smoke | --oracle | --encode | --assemble (+ internal --worker-rank).
K selection: --bits {3,4}; run --encode once per K on the SAME capture.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

BUNDLE = Path(
    "/home/mbelleau/.cache/huggingface/hub/"
    "models--brandonmusic--GLM-5.2-EXL3-TR3-3.0bpw/snapshots/"
    "9297b9f1d53af5c67cffa01e30cc071a1ff7144b/calibration_encoder")
ENCODER = BUNDLE / "encode_tr3_v31.py"
ENCODER_SHA = "e9a85a47e165c8d8644354cef611efbb81dfd9ba88544ca59f0c80ee6bc75032"
DEFAULT_SRC = (
    "/home/mbelleau/.cache/huggingface/hub/"
    "models--malaiwah--GLM-5.2-SIQ-Fruit-Instruct-bf16/snapshots/"
    "678954f65e056a0f508e21eeb9251c655bb9463f")


def load_base(bits: int, src_dir: str):
    raw = ENCODER.read_bytes()
    got = hashlib.sha256(raw).hexdigest()
    if got != ENCODER_SHA:
        raise SystemExit(f"encoder sha mismatch: {got} != pinned {ENCODER_SHA}")
    spec = importlib.util.spec_from_file_location(f"fruit_tr3_k{bits}", ENCODER)
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)

    cfg = json.load(open(Path(src_dir) / "config.json"))
    m.NUM_LAYERS = cfg["num_hidden_layers"]            # 13
    m.HIDDEN = cfg["hidden_size"]                      # 1024
    m.MOE_INTER = cfg["moe_intermediate_size"]         # 512
    m.SLICE = m.MOE_INTER // m.TP                      # 128 — import-time value is stale
    m.FIRST_MOE_LAYER = cfg["first_k_dense_replace"]   # 3
    m.NUM_EXPERTS = cfg["n_routed_experts"]            # 256
    m.BITS = bits
    m.KEEP_NVFP4 = 0                                   # all experts trellis-carried

    def load_expert_bf16_fruit(src, layer, expert, proj, device, logfile):
        torch, _ = m._lazy_torch()
        key = m.expert_key(layer, expert, proj, "weight")
        reader = src.reader_for(key, logfile)
        dtype, shape, _, _ = reader.tensors[key]
        expected = ((m.MOE_INTER, m.HIDDEN) if proj != "down_proj"
                    else (m.HIDDEN, m.MOE_INTER))
        if dtype != "BF16" or tuple(shape) != expected:
            raise RuntimeError(
                f"{key}: expected BF16 {expected}, got {dtype} {shape}")
        raw_t = reader.read_bytes(key)
        if len(raw_t) != m.MOE_INTER * m.HIDDEN * 2:
            raise RuntimeError(f"{key}: BF16 payload length mismatch")
        t = torch.frombuffer(bytearray(raw_t), dtype=torch.bfloat16).view(*shape)
        return t.to(device)

    m.load_expert_bf16 = load_expert_bf16_fruit
    return m


def run_workers(args) -> None:
    layers = args.layers
    procs = []
    for rank in range(args.workers):
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu_list[rank % len(args.gpu_list)])
        cmd = [sys.executable, str(Path(__file__).resolve()),
               "--worker-rank", str(rank),
               "--bits", str(args.bits),
               "--src", args.src, "--work", args.work,
               "--capture-dir", args.capture_dir,
               "--layers", layers,
               "--workers", str(args.workers),
               "--gpus", args.gpus,
               "--min-routed", str(args.min_routed),
               "--lockstep", str(args.lockstep)]
        p = subprocess.Popen(cmd, env=env)
        procs.append((rank, p))
        print(f"spawned worker {rank} pid {p.pid} "
              f"gpu {env['CUDA_VISIBLE_DEVICES']}", flush=True)
    fail = 0
    for rank, p in procs:
        rc = p.wait()
        print(f"worker {rank} exit {rc}", flush=True)
        fail += rc != 0
    if fail:
        raise SystemExit(f"{fail} workers failed")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bits", type=int, required=True, choices=(3, 4))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--oracle", action="store_true")
    ap.add_argument("--encode", action="store_true")
    ap.add_argument("--assemble", action="store_true")
    ap.add_argument("--worker-rank", type=int, default=None)
    ap.add_argument("--src", default=DEFAULT_SRC)
    ap.add_argument("--work", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--capture-dir", default="/home/mbelleau/fq-0c/capture")
    ap.add_argument("--layers", default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--gpus", default="0,1,2,3,4,5,6,7",
                    help="comma list of physical GPU ids for workers")
    ap.add_argument("--min-routed", type=int, default=1024)
    ap.add_argument("--lockstep", default="auto")
    ap.add_argument("--io-workers", type=int, default=12)
    ap.add_argument("--oracle-experts", default="0,1")
    ap.add_argument("--oracle-log", default="/home/mbelleau/fq-0c/oracle.log")
    args = ap.parse_args()
    args.gpu_list = [int(g) for g in str(args.gpus).split(",")]

    m = load_base(args.bits, args.src)
    if args.layers is None:
        args.layers = f"{m.FIRST_MOE_LAYER}-{m.NUM_LAYERS - 1}"
    if args.work is None:
        args.work = f"/home/mbelleau/fq-0c/work-k{args.bits}-tr3"
    Path(args.work).mkdir(parents=True, exist_ok=True)

    if args.worker_rank is not None:
        src = m.SourceModel(args.src)
        logfile = open(Path(args.work) / f"worker-{args.worker_rank}.log", "a")
        lockstep = m.resolve_lockstep(args.lockstep)
        todo = m.parse_layers(args.layers)[args.worker_rank::args.workers]
        for L in todo:
            if m.layer_done(args.work, L):
                print(f"layer {L}: done, skip", flush=True)
                continue
            t0 = time.time()
            m.process_layer(src, args.work, L, None, args.capture_dir,
                            args.min_routed, logfile, lockstep)
            print(f"layer {L}: encoded K{args.bits} in {time.time()-t0:.0f}s",
                  flush=True)
        return

    if args.smoke:
        m.smoke(argparse.Namespace())
        return
    if args.oracle:
        ns = argparse.Namespace(
            src=args.src, capture_dir=args.capture_dir,
            oracle_experts=args.oracle_experts, oracle_log=args.oracle_log,
            layers=args.layers, min_routed=args.min_routed,
            lockstep=args.lockstep)
        m.oracle(ns)
        return
    if args.encode:
        m.check_capture(args.capture_dir, m.parse_layers(args.layers))
        run_workers(args)
        missing = [L for L in m.parse_layers(args.layers)
                   if not m.layer_done(args.work, L)]
        if missing:
            raise SystemExit(f"missing layers after encode: {missing}")
        print(f"encode K{args.bits} complete: {args.layers}", flush=True)
        return
    if args.assemble:
        if args.out is None:
            args.out = f"/home/mbelleau/fq-0c/fruit-k{args.bits}"
        ns = argparse.Namespace(src=args.src, work=args.work, out=args.out,
                                io_workers=args.io_workers)
        m.assemble(ns)
        print(f"assembled K{args.bits} -> {args.out}", flush=True)
        return
    raise SystemExit("pick a mode: --smoke/--oracle/--encode/--assemble")


if __name__ == "__main__":
    main()
