#!/usr/bin/env python3
"""PoC: Additive Residual Encoding for EXL3 Trellis — ACCURATE.

Uses real EXL3 Viterbi trellis quantizer (ext.quantize_tiles).
Works in regularized space. Tests multiple residual paths:

  Scalar residuals:
    K3 via 2+1   (K2 base + 1-bit scalar residual)
    K4 via 3+1   (K3 base + 1-bit scalar residual)
    K4 via 2+1+1 (K2 base + two chained 1-bit scalar residuals)

  Trellis residuals (the real question):
    K3 via 2+1t  (K2 base + 1-bit trellis residual)
    K4 via 2+2t  (K2 base + 2-bit trellis residual — DIRECT, no compounding)
    K4 via 2+1t+1t (K2 base + two chained 1-bit trellis residuals)

The trellis residual uses the SAME ext.quantize_tiles kernel on the residual
tiles — exploiting inter-weight correlations in the error, not just sign.

Also tests shared-H: all residuals use the K2 base's regularization (same
suh/svh/Hadamard/g_scale). No separate suh/svh per K level.
"""

from __future__ import annotations
import argparse, json, math, os, sys, types, importlib.util, time
from pathlib import Path
import torch, torch.nn.functional as F

EXL3_PKG = "/opt/fruit-pip/exllamav3"
HAD_K, HAD_N = 128, 128

def _bootstrap():
    pkg = types.ModuleType("exllamav3"); pkg.__path__ = [EXL3_PKG]; sys.modules["exllamav3"] = pkg
    for sub in ["util", "modules", "modules.quant", "modules.quant.exl3_lib"]:
        full = f"exllamav3.{sub}"; m = types.ModuleType(full)
        m.__path__ = [f"{EXL3_PKG}/{sub.replace('.', '/')}"]; sys.modules[full] = m
    class _DPB:
        def __init__(self, *a, **kw): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def update(self, *a): pass
        def new_task(self, *a, **kw): pass
    _s = types.ModuleType("exllamav3.util.progress"); _s.ProgressBar = _DPB; sys.modules["exllamav3.util.progress"] = _s
    _s = types.ModuleType("exllamav3.util.memory"); _s.free_mem = lambda: None; _s.list_gpu_tensors = lambda: []; sys.modules["exllamav3.util.memory"] = _s
    _s = types.ModuleType("exllamav3.util"); _s.__path__ = [f"{EXL3_PKG}/util"]; _s.cuda_sync_active = lambda *a, **kw: torch.cuda.synchronize(); sys.modules["exllamav3.util"] = _s
    _s = types.ModuleType("exllamav3.util.tensor"); _s.save_tensor_image = lambda *a, **kw: None; sys.modules["exllamav3.util.tensor"] = _s
    print("Loading ext (JIT)...", flush=True)
    spec = importlib.util.spec_from_file_location("exllamav3.ext", f"{EXL3_PKG}/ext.py")
    m = importlib.util.module_from_spec(spec); sys.modules["exllamav3.ext"] = m; spec.loader.exec_module(m)
    ext = m.exllamav3_ext; print("  ext OK", flush=True)
    print("Loading hadamard...", flush=True)
    spec = importlib.util.spec_from_file_location("exllamav3.util.hadamard", f"{EXL3_PKG}/util/hadamard.py")
    m = importlib.util.module_from_spec(spec); sys.modules["exllamav3.util.hadamard"] = m; spec.loader.exec_module(m)
    ghd = m.get_hadamard_dt; print("  hadamard OK", flush=True)
    print("Loading quantize...", flush=True)
    spec = importlib.util.spec_from_file_location("exllamav3.modules.quant.exl3_lib.quantize", f"{EXL3_PKG}/modules/quant/exl3_lib/quantize.py")
    m = importlib.util.module_from_spec(spec); sys.modules["exllamav3.modules.quant.exl3_lib.quantize"] = m; spec.loader.exec_module(m)
    tcp = m.tensor_core_perm; tcpi = m.tensor_core_perm_i; qtf = m.quantize_tiles
    cbs = m.codebook_scale; print("  quantize OK", flush=True)
    return ext, ghd, tcp, tcpi, qtf, cbs


def block_rms(x, dim, keepdim=False):
    return x.square().mean(dim=dim, keepdim=keepdim).sqrt()


def regularize(w, device, ghd, cbs):
    """EXL3 regularization: sign flips + per-channel scales + block Hadamard."""
    k, n = w.shape
    g = torch.Generator(device="cpu").manual_seed(0)
    su = (torch.randn(k, generator=g).sign() + 1e-5).sign().float().to(device)
    sv = (torch.randn(n, generator=g).sign() + 1e-5).sign().float().to(device).unsqueeze(0)
    # Output channel scales
    out_scales = block_rms(w, dim=0, keepdim=True)
    mean = out_scales.mean().item()
    if mean > 1e-30: out_scales = out_scales / mean
    sv = (sv * out_scales + 1e-10).float()
    w = (w / sv).contiguous()
    had_n = ghd(HAD_N, device, torch.float, 1.0 / math.sqrt(HAD_N))
    w = (w.view(k, n // HAD_N, HAD_N) @ had_n).view(k, n).contiguous()
    in_scales = block_rms(w, dim=1, keepdim=True).clamp(min=1e-30)
    su = (su.unsqueeze(1) * in_scales / (-cbs) + 1e-10).float()
    w = (w / su).contiguous()
    had_k = ghd(HAD_K, device, torch.float, 1.0 / math.sqrt(HAD_K))
    w = (had_k @ w.view(k // HAD_K, HAD_K, n)).view(k, n).contiguous()
    return w, su, sv


def quantize_trellis(w_reg, K, device, tcp, tcpi, qtf):
    """Quantize using real EXL3 Viterbi trellis. Returns quantized weight in regularized space."""
    k, n = w_reg.shape; tiles_n = n // 16; weight_q = torch.zeros_like(w_reg)
    qa = {"K": K, "mcg": True}; perm = tcp(device); perm_i = tcpi(device)
    for bi in range(0, k, 16):
        rows = w_reg[bi:bi+16]
        tiles = rows.reshape(16, tiles_n, 16).permute(1, 0, 2).reshape(tiles_n, 256)
        tiles = tiles[:, perm].contiguous()
        quant_w, _ = qtf(tiles, qa)
        quant_w = quant_w[:, perm_i].reshape(tiles_n, 16, 16).permute(1, 0, 2).reshape(16, n)
        weight_q[bi:bi+16] = quant_w
    return weight_q


def q1b_global(r):
    s = r.abs().mean().item()
    return (torch.zeros_like(r), 0.0) if s < 1e-12 else (torch.sign(r) * s, s)

def metrics(w_ref, w_ap):
    e = w_ref - w_ap
    return {"mse": e.pow(2).mean().item(),
            "cos": F.cosine_similarity(w_ref.flatten().unsqueeze(0), w_ap.flatten().unsqueeze(0), dim=1).item(),
            "rel_frob": (e.norm() / w_ref.norm()).item()}


def run_poc(w, name, device, ext, ghd, tcp, tcpi, qtf, cbs):
    w = w.to(device).float()
    w_reg, su, sv = regularize(w, device, ghd, cbs)

    # Standalone trellis quantization at K=2,3,4
    qk = {}
    for K in [2, 3, 4]:
        t0 = time.perf_counter()
        qk[K] = quantize_trellis(w_reg, K, device, tcp, tcpi, qtf)
        torch.cuda.synchronize(); t1 = time.perf_counter()
        mse = (w_reg - qk[K]).pow(2).mean().item()
        print(f"    K{K}: MSE={mse:.6e}  time={t1-t0:.2f}s", flush=True)

    r23 = w_reg - qk[2]   # K2→K3 residual
    r34 = w_reg - qk[3]   # K3→K4 residual
    r24 = w_reg - qk[2]   # K2→K4 residual (same as r23, but we'll quantize at 2-bit)

    # --- Scalar residuals ---
    r23_s1 = q1b_global(r23)[0]       # 1-bit scalar
    r34_s1 = q1b_global(r34)[0]       # 1-bit scalar
    rch = w_reg - (qk[2] + r23_s1)    # chained residual from approximate K3
    rch_s1 = q1b_global(rch)[0]       # 1-bit scalar (second stage)

    # --- Trellis residuals (use ext.quantize_tiles on the residual) ---
    print("    Computing trellis residuals...", flush=True)
    r23_t1 = quantize_trellis(r23, 1, device, tcp, tcpi, qtf)   # 1-bit trellis
    r24_t2 = quantize_trellis(r24, 2, device, tcp, tcpi, qtf)   # 2-bit trellis (DIRECT K2→K4)
    r34_t1 = quantize_trellis(r34, 1, device, tcp, tcpi, qtf)   # 1-bit trellis (K3→K4)
    rch_t1 = quantize_trellis(rch, 1, device, tcp, tcpi, qtf)   # 1-bit trellis (chained second stage)
    torch.cuda.synchronize()

    # --- Reconstruct all paths ---
    paths = {}
    # Standalone
    for K in [2, 3, 4]: paths[f"K{K}_standalone"] = qk[K]
    # Scalar residuals
    paths["K3_2+1s"] = qk[2] + r23_s1
    paths["K4_3+1s"] = qk[3] + r34_s1
    paths["K4_2+1s+1s"] = qk[2] + r23_s1 + rch_s1
    # Trellis residuals
    paths["K3_2+1t"] = qk[2] + r23_t1
    paths["K4_3+1t"] = qk[3] + r34_t1
    paths["K4_2+2t"] = qk[2] + r24_t2           # DIRECT: 2-bit trellis residual, no compounding
    paths["K4_2+1t+1t"] = qk[2] + r23_t1 + rch_t1  # CUMULATIVE: two 1-bit trellis residuals

    # --- Measure ---
    res = {label: metrics(w_reg, w_ap) for label, w_ap in paths.items()}
    e2, e3, e4 = res["K2_standalone"]["mse"], res["K3_standalone"]["mse"], res["K4_standalone"]["mse"]
    g23, g34, g24 = e2-e3, e3-e4, e2-e4
    gc = {}
    for l, r in res.items():
        if l in ("K2_standalone", "K3_standalone", "K4_standalone"): continue
        if l.startswith("K3_"): gc[l] = (e2 - r["mse"]) / g23 if g23 > 0 else 0
        elif l.startswith("K4_3+1"): gc[l] = (e3 - r["mse"]) / g34 if g34 > 0 else 0
        elif l.startswith("K4_2+"): gc[l] = (e2 - r["mse"]) / g24 if g24 > 0 else 0

    # Residual stats
    rs = {
        "r_23": {"std": r23.std().item(), "abs_mean": r23.abs().mean().item(),
                  "kurtosis": ((r23 - r23.mean()).pow(4).mean().item() / (r23.std().item()**4 + 1e-12) - 3.0)},
        "r_34": {"std": r34.std().item(), "abs_mean": r34.abs().mean().item(),
                  "kurtosis": ((r34 - r34.mean()).pow(4).mean().item() / (r34.std().item()**4 + 1e-12) - 3.0)},
    }
    return {"tensor": name, "shape": list(w.shape), "metrics": res, "gap_closed": gc,
            "residual_stats": rs, "eps": {"K2": e2, "K3": e3, "K4": e4},
            "improvement_ratios": {"K2_to_K3": e2/e3 if e3 > 0 else float("inf"),
                                    "K3_to_K4": e3/e4 if e4 > 0 else float("inf")}}


LABELS = ["K3_2+1s", "K3_2+1t", "K4_3+1s", "K4_3+1t", "K4_2+2t", "K4_2+1s+1s", "K4_2+1t+1t"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--data-dir", default="/tmp/poc_residual/data")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    dev = torch.device(args.device)
    print(f"Device: {dev}  GPU: {torch.cuda.get_device_name(0)}", flush=True)
    ext, ghd, tcp, tcpi, qtf, cbs = _bootstrap()

    dd = Path(args.data_dir); all_r = {}
    for proj in ["gate_proj", "up_proj", "down_proj"]:
        p = dd / f"{proj}.pt"
        if not p.exists(): print(f"SKIP {proj}", flush=True); continue
        w = torch.load(p, map_location="cpu")
        print(f"\n{'='*60}\nProcessing {proj}: shape={tuple(w.shape)}\n{'='*60}", flush=True)
        r = run_poc(w, proj, dev, ext, ghd, tcp, tcpi, qtf, cbs)
        all_r[proj] = r
        eps = r["eps"]; gc = r["gap_closed"]; ir = r["improvement_ratios"]
        print(f"\n  Standalone (MSE, regularized space):", flush=True)
        print(f"    K2: {eps['K2']:.6e}", flush=True)
        print(f"    K3: {eps['K3']:.6e}  ({ir['K2_to_K3']:.2f}x)", flush=True)
        print(f"    K4: {eps['K4']:.6e}  ({ir['K3_to_K4']:.2f}x)", flush=True)
        print(f"\n  Residual paths:", flush=True)
        for lb in LABELS:
            m = r["metrics"][lb]; g = gc.get(lb, 0)
            print(f"    {lb:16s}: MSE={m['mse']:.6e}  cos={m['cos']:.6f}  gap={g:.1%}", flush=True)
        for rn, rsv in r["residual_stats"].items():
            print(f"    {rn}: std={rsv['std']:.2e}  kurt={rsv['kurtosis']:.2f}", flush=True)

    # Aggregate
    print(f"\n{'='*60}\nAGGREGATE (mean across 3 projections)\n{'='*60}", flush=True)
    if all_r:
        agg, agg_gc, eps_a = {}, {}, {}
        for pr, r in all_r.items():
            for lb, m in r["metrics"].items():
                agg.setdefault(lb, {"mse": [], "cos": []})
                agg[lb]["mse"].append(m["mse"]); agg[lb]["cos"].append(m["cos"])
            for lb, g in r["gap_closed"].items(): agg_gc.setdefault(lb, []).append(g)
            for K, e in r["eps"].items(): eps_a.setdefault(K, []).append(e)
        print(f"\n  Standalone:", flush=True)
        for K in ["K2", "K3", "K4"]:
            v = eps_a.get(K, [0]); print(f"    {K}: {sum(v)/len(v):.6e}", flush=True)
        print(f"\n  Residual paths:", flush=True)
        for lb in LABELS:
            if lb in agg:
                mse = sum(agg[lb]["mse"])/len(agg[lb]["mse"]); cos = sum(agg[lb]["cos"])/len(agg[lb]["cos"])
                g = sum(agg_gc.get(lb, [0]))/len(agg_gc.get(lb, [0]))
                print(f"    {lb:16s}: MSE={mse:.6e}  cos={cos:.6f}  gap={g:.1%}", flush=True)
        e2 = sum(eps_a["K2"])/len(eps_a["K2"]); e3 = sum(eps_a["K3"])/len(eps_a["K3"]); e4 = sum(eps_a["K4"])/len(eps_a["K4"])
        print(f"\n  Improvement: K2→K3: {e2/e3:.2f}x  K3→K4: {e3/e4:.2f}x", flush=True)

        # Summary table
        print(f"\n  ┌─────────────────────────────────────────────────────────┐", flush=True)
        print(f"  │ PATH COMPARISON (gap closed vs standalone at same K)   │", flush=True)
        print(f"  ├──────────────────┬──────────┬──────────┬───────────────┤", flush=True)
        print(f"  │ Path             │ Bits/w   │ Gap %    │ vs true K     │", flush=True)
        print(f"  ├──────────────────┼──────────┼──────────┼───────────────┤", flush=True)
        for lb in LABELS:
            if lb in agg:
                mse = sum(agg[lb]["mse"])/len(agg[lb]["mse"])
                g = sum(agg_gc.get(lb, [0]))/len(agg_gc.get(lb, [0]))
                if lb.startswith("K3_"): bits = "3 (2+1)"; ref_mse = e3; ref_name = "K3"
                elif lb.startswith("K4_3+1"): bits = "4 (3+1)"; ref_mse = e4; ref_name = "K4"
                elif lb.startswith("K4_2+2"): bits = "4 (2+2)"; ref_mse = e4; ref_name = "K4"
                elif lb.startswith("K4_2+1"): bits = "4 (2+1+1)"; ref_mse = e4; ref_name = "K4"
                else: bits = "?"; ref_mse = e4; ref_name = "?"
                ratio = mse / ref_mse if ref_mse > 0 else float("inf")
                print(f"  │ {lb:16s} │ {bits:8s} │ {g:7.1%}  │ {ratio:.2f}x {ref_name:2s}     │", flush=True)
        print(f"  └──────────────────┴──────────┴──────────┴───────────────┘", flush=True)

    out = {"poc": "additive_residual_encoding", "version": "accurate_trellis_v2",
           "quantizer": "real EXL3 Viterbi trellis (ext.quantize_tiles)",
           "model": "zai-org/GLM-5.2 layer 30 expert 137", "gpu": torch.cuda.get_device_name(0),
           "note": "All measurements in regularized space. s=scalar, t=trellis residual.",
           "per_tensor": all_r}
    op = args.out or "results.json"; Path(op).write_text(json.dumps(out, indent=2, default=str))
    print(f"\nResults saved to {op}", flush=True)

if __name__ == "__main__":
    main()
