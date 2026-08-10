#!/usr/bin/env python3
"""0f(ii): benchmark quantize_exl3() K3 vs K4 on one real GLM-5.2 expert.

Times the lazy-encode unit of work (07-lazy-encode.md): one expert =
gate_proj + up_proj + down_proj, real BF16 weights from zai-org/GLM-5.2,
synthetic Hessian (X~N(0,1), count>0 → real LDL+trellis path; timing is
representative, quality is not measured here — 0f(ii) is documentation
only per 01 §6). Reports per-tensor and per-expert seconds, split into
H-finalize (amortizable) vs quantize-proper, and sizes
VLLM_FQ_ENCODE_BUDGET_PCT.

Run inside gg-env with CUDA_VISIBLE_DEVICES=<gpu>.
"""
import json
import struct
import sys
import time
from pathlib import Path

import torch

SNAP = Path(sys.argv[1]) if len(sys.argv) > 1 else None
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(".")
LAYER, EXPERT = 30, 137
KS = (3, 4)
H_SAMPLES = 4096
REPEATS = 3


def load_tensor(snap: Path, name: str) -> torch.Tensor:
    idx = json.load(open(snap / "model.safetensors.index.json"))
    shard = snap / idx["weight_map"][name]
    with open(shard, "rb") as f:
        hlen = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(hlen))
        t = hdr[name]
        a, b = t["data_offsets"]
        f.seek(8 + hlen + a)
        raw = f.read(b - a)
    w = torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).reshape(t["shape"])
    return w


def fresh_H(k: int, device) -> dict:
    x = torch.randn(H_SAMPLES, k, device=device, dtype=torch.float32)
    H = x.T @ x
    return {"H": H, "count": H_SAMPLES, "finalized": False, "L": None,
            "device": device, "diag": None, "su": None, "q_fallback": False}


def main():
    from exllamav3.modules.quant.exl3_lib.quantize import quantize_exl3

    device = torch.device("cuda:0")
    torch.cuda.init()
    results = {"config": {"layer": LAYER, "expert": EXPERT, "h_samples": H_SAMPLES,
                          "repeats": REPEATS, "gpu": torch.cuda.get_device_name(0)},
               "tensors": {}}

    projs = {
        "gate_proj": None,  # (in=6144, out=2048) after transpose
        "up_proj": None,
        "down_proj": None,  # (in=2048, out=6144)
    }
    for p in projs:
        name = f"model.layers.{LAYER}.mlp.experts.{EXPERT}.{p}.weight"
        w = load_tensor(SNAP, name)          # stored (out, in)
        projs[p] = w.T.contiguous().float()  # row major (in, out)
        print(f"{p}: {tuple(projs[p].shape)}", flush=True)

    for p, w in projs.items():
        k_in = w.shape[0]
        results["tensors"][p] = {"shape": list(w.shape), "runs": {}}
        for K in KS:
            qa = {"K": K, "seed": 0, "sigma_reg": 0.025, "devices": ["cuda:0"],
                  "device_ratios": [1.0], "apply_out_scales": None, "mcg": True}
            # warmup (JIT, hadamard caches) — not timed
            hd = fresh_H(k_in, device)
            quantize_exl3(w.to(device), hd, qa, return_weight_q=False)
            torch.cuda.synchronize()

            finalize_s, quant_s = [], []
            for _ in range(REPEATS):
                hd = fresh_H(k_in, device)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                out = quantize_exl3(w.to(device), hd, qa, return_weight_q=False)
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                # second call reuses finalized H (L cached) → quantize-proper cost
                out2 = quantize_exl3(w.to(device), hd, qa, return_weight_q=False)
                torch.cuda.synchronize()
                t2 = time.perf_counter()
                finalize_s.append(t1 - t0)
                quant_s.append(t2 - t1)
            proxy_err = float(out[1]) if isinstance(out, tuple) else None
            results["tensors"][p]["runs"][f"K{K}"] = {
                "cold_s": sorted(finalize_s)[len(finalize_s) // 2],
                "warmH_s": sorted(quant_s)[len(quant_s) // 2],
                "proxy_err": proxy_err,
            }
            print(f"{p} K{K}: cold {results['tensors'][p]['runs'][f'K{K}']['cold_s']:.2f}s "
                  f"warm-H {results['tensors'][p]['runs'][f'K{K}']['warmH_s']:.2f}s "
                  f"proxy_err {proxy_err}", flush=True)

    # Extrapolation: expert = gate+up+down cold (worst case: fresh H both shapes)
    for K in KS:
        expert_cold = sum(results["tensors"][p]["runs"][f"K{K}"]["cold_s"] for p in projs)
        expert_warm = sum(results["tensors"][p]["runs"][f"K{K}"]["warmH_s"] for p in projs)
        layer_cold = expert_cold * 256
        results[f"extrapolation_K{K}"] = {
            "expert_cold_s": round(expert_cold, 2),
            "expert_warmH_s": round(expert_warm, 2),
            "layer_256experts_cold_h": round(layer_cold / 3600, 2),
            "full_model_75layers_cold_h": round(layer_cold * 75 / 3600, 1),
            "experts_per_hour_at_5pct_budget": round(3600 * 0.05 / expert_cold, 1),
        }
        print(f"K{K} extrapolation: {results[f'extrapolation_K{K}']}", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(results, indent=1))
    print("done", flush=True)


if __name__ == "__main__":
    main()
