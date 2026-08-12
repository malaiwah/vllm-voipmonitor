#!/usr/bin/env python3
"""PoC v4b: Deep dive on sparse residual + combinations.

Tests:
  - Finer sparsity levels (0.1% to 10%)
  - Sparse + Lloyd-Max combinations
  - Sparse on K3 residual (K3+sparse)
  - Proper bit-budget accounting with index storage
  - Sparse + 2-bit on K2 residual for K4 path
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
    k, n = w.shape
    g = torch.Generator(device="cpu").manual_seed(0)
    su = (torch.randn(k, generator=g).sign() + 1e-5).sign().float().to(device)
    sv = (torch.randn(n, generator=g).sign() + 1e-5).sign().float().to(device)
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

def q1b_scalar(r):
    s = r.abs().mean().item()
    return torch.zeros_like(r) if s < 1e-12 else torch.sign(r) * s

def q2b_lloyd(r):
    sigma = r.std().item()
    if sigma < 1e-12: return torch.zeros_like(r)
    levels = torch.tensor([-1.5104, -0.4528, 0.4528, 1.5104], device=r.device) * sigma
    flat = r.flatten().unsqueeze(1)
    d = torch.cdist(flat, levels.unsqueeze(1))
    return levels[d.argmin(dim=1)].reshape(r.shape)

def q_sparse_1bit(r, top_pct):
    """1-bit sign for all + fp16 override on top-k% largest |r|.
    Bit cost (per weight): 1 (sign) + top_pct/100 * 16 (fp16 values)
    + index overhead (see bit_budget_sparse)
    """
    flat = r.flatten()
    sigma = flat.std().item()
    if sigma < 1e-12: return torch.zeros_like(r)
    k = max(1, int(flat.numel() * top_pct / 100))
    topk_vals, topk_idx = flat.abs().topk(k)
    s = flat.abs().mean().item()
    result = torch.sign(flat) * s
    result[topk_idx] = flat[topk_idx].to(torch.float16).float()
    return result.reshape(r.shape)

def q_sparse_only(r, top_pct):
    """Sparse-only: fp16 on top-k%, zero elsewhere.
    Bit cost: index overhead + top_pct/100 * 16
    """
    flat = r.flatten()
    sigma = flat.std().item()
    if sigma < 1e-12: return torch.zeros_like(r)
    k = max(1, int(flat.numel() * top_pct / 100))
    topk_vals, topk_idx = flat.abs().topk(k)
    result = torch.zeros_like(flat, dtype=r.dtype)
    result[topk_idx] = flat[topk_idx]
    return result.reshape(r.shape)

def q_sparse_plus_lloyd(r, top_pct):
    """2-bit Lloyd-Max + fp16 override on top-k% largest residual-after-Lloyd.
    Bit cost: 2 (Lloyd-Max) + top_pct/100 * 16 + index overhead
    """
    lloyd = q2b_lloyd(r)
    residual = r - lloyd
    flat = residual.flatten()
    k = max(1, int(flat.numel() * top_pct / 100))
    topk_vals, topk_idx = flat.abs().topk(k)
    result = lloyd.flatten().clone()
    result[topk_idx] = flat[topk_idx] + lloyd.flatten()[topk_idx]  # exact value
    return result.reshape(r.shape)

def bit_budget_sparse(n_weights, top_pct, base_bits=0, sign_bit=False):
    """Compute actual bits/weight including index storage.
    Sparse indices stored as sorted (index, value) pairs.
    Index width = ceil(log2(n_weights)) bits.
    Value width = 16 bits (fp16).
    """
    n_sparse = int(n_weights * top_pct / 100)
    idx_bits = max(1, math.ceil(math.log2(n_weights))) if n_sparse > 0 else 0
    sparse_bits = n_sparse * (idx_bits + 16)  # index + value
    total_bits = base_bits * n_weights  # base quantization
    if sign_bit:
        total_bits += n_weights  # 1 bit/weight for sign
    total_bits += sparse_bits
    return total_bits / n_weights

def metrics(w_ref, w_ap):
    e = w_ref - w_ap
    return {"mse": e.pow(2).mean().item(),
            "cos": F.cosine_similarity(w_ref.flatten().unsqueeze(0), w_ap.flatten().unsqueeze(0), dim=1).item()}

def run_poc(w, name, device, ext, ghd, tcp, tcpi, qtf, cbs):
    w = w.to(device).float()
    w_reg, su, sv = regularize(w, device, ghd, cbs)
    n_weights = w_reg.numel()

    qk = {}
    for K in [2, 3, 4]:
        qk[K] = quantize_trellis(w_reg, K, device, tcp, tcpi, qtf)
        torch.cuda.synchronize()
        mse = (w_reg - qk[K]).pow(2).mean().item()
        print(f"    K{K}: MSE={mse:.6e}", flush=True)

    r23 = w_reg - qk[2]; r34 = w_reg - qk[3]; r24 = w_reg - qk[2]

    paths = {}
    bits = {}

    for K in [2, 3, 4]:
        paths[f"K{K}_standalone"] = qk[K]
        bits[f"K{K}_standalone"] = float(K)

    # Baselines
    paths["K3_2+1s"] = qk[2] + q1b_scalar(r23); bits["K3_2+1s"] = 3.0
    paths["K4_2+2lloyd"] = qk[2] + q2b_lloyd(r24); bits["K4_2+2lloyd"] = 4.0
    paths["K4_3+2lm"] = qk[3] + q2b_lloyd(r34); bits["K4_3+2lm"] = 5.0
    rch = w_reg - (qk[2] + q1b_scalar(r23))
    paths["K4_2+1s+1s"] = qk[2] + q1b_scalar(r23) + q1b_scalar(rch); bits["K4_2+1s+1s"] = 4.0

    # Sparse 1-bit: finer granularity
    print("  Sparse 1-bit variants...", flush=True)
    for pct in [0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0]:
        label = f"K3_2+sparse{pct}%+1s"
        paths[label] = qk[2] + q_sparse_1bit(r23, top_pct=pct)
        bits[label] = bit_budget_sparse(n_weights, pct, base_bits=2, sign_bit=True)

        label = f"K4_2+sparse{pct}%+1s"
        paths[label] = qk[2] + q_sparse_1bit(r24, top_pct=pct)
        bits[label] = bit_budget_sparse(n_weights, pct, base_bits=2, sign_bit=True)
    torch.cuda.synchronize()

    # Sparse-only (no sign bit): K2 + sparse correction directly
    print("  Sparse-only variants...", flush=True)
    for pct in [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]:
        label = f"K3_2+sparse_only{pct}%"
        paths[label] = qk[2] + q_sparse_only(r23, top_pct=pct)
        bits[label] = bit_budget_sparse(n_weights, pct, base_bits=2, sign_bit=False)

        label = f"K4_2+sparse_only{pct}%"
        paths[label] = qk[2] + q_sparse_only(r24, top_pct=pct)
        bits[label] = bit_budget_sparse(n_weights, pct, base_bits=2, sign_bit=False)
    torch.cuda.synchronize()

    # Sparse + Lloyd-Max
    print("  Sparse + Lloyd-Max variants...", flush=True)
    for pct in [0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0]:
        label = f"K4_2+2lm+sparse{pct}%"
        paths[label] = qk[2] + q_sparse_plus_lloyd(r24, top_pct=pct)
        bits[label] = bit_budget_sparse(n_weights, pct, base_bits=4, sign_bit=False)

        label = f"K5_3+2lm+sparse{pct}%"
        paths[label] = qk[3] + q_sparse_plus_lloyd(r34, top_pct=pct)
        bits[label] = bit_budget_sparse(n_weights, pct, base_bits=5, sign_bit=False)
    torch.cuda.synchronize()

    # K3 + sparse only (no 2-bit)
    print("  K3 + sparse only...", flush=True)
    for pct in [0.5, 1.0, 2.0, 5.0, 10.0]:
        label = f"K4_3+sparse_only{pct}%"
        paths[label] = qk[3] + q_sparse_only(r34, top_pct=pct)
        bits[label] = bit_budget_sparse(n_weights, pct, base_bits=3, sign_bit=False)
    torch.cuda.synchronize()

    # K2 + 1-bit + sparse override (3-tier)
    print("  3-tier: K2 + 1s + sparse override on residual...", flush=True)
    r_after_1s = r23 - q1b_scalar(r23)  # residual after 1-bit
    for pct in [0.5, 1.0, 2.0, 5.0]:
        label = f"K4_2+1s+sparse{pct}%"
        # Start from K2+1s, add sparse correction on the remaining error
        r_rem = w_reg - (qk[2] + q1b_scalar(r23))
        paths[label] = qk[2] + q1b_scalar(r23) + q_sparse_only(r_rem, top_pct=pct)
        bits[label] = bit_budget_sparse(n_weights, pct, base_bits=3, sign_bit=False)
    torch.cuda.synchronize()

    # Measure all
    res = {}
    e2 = (w_reg - qk[2]).pow(2).mean().item()
    e3 = (w_reg - qk[3]).pow(2).mean().item()
    e4 = (w_reg - qk[4]).pow(2).mean().item()
    for label, w_ap in paths.items():
        m = metrics(w_reg, w_ap)
        mse = m["mse"]
        # Gap closed relative to appropriate reference
        if label.startswith("K3_") or label.startswith("K4_2+sparse") or label.startswith("K4_2+1s"):
            gc = (e2 - mse) / (e2 - e4) if (e2 - e4) > 0 else 0  # gap toward K4
        elif label.startswith("K4_3+"):
            gc = (e3 - mse) / (e3 - e4) if (e3 - e4) > 0 else 0
        elif label.startswith("K5_"):
            gc = (e3 - mse) / (e3 - e4) if (e3 - e4) > 0 else 0
        else:
            gc = (e2 - mse) / (e2 - e4) if (e2 - e4) > 0 else 0
        m["gap_to_K4"] = gc
        m["bits"] = bits.get(label, 0)
        m["mse_per_bit"] = mse / max(bits.get(label, 0.001), 0.001)  # efficiency
        res[label] = m
    return res, e2, e3, e4

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--data-dir", default="/tmp/poc_residual/data")
    ap.add_argument("--out", default="/tmp/poc_residual/results_v4b.json")
    args = ap.parse_args()
    dev = torch.device(args.device)
    print(f"Device: {dev}  GPU: {torch.cuda.get_device_name(0)}", flush=True)
    ext, ghd, tcp, tcpi, qtf, cbs = _bootstrap()

    dd = Path(args.data_dir); all_res = {}; standalone = {}
    for proj in ["gate_proj", "up_proj", "down_proj"]:
        p = dd / f"{proj}.pt"
        if not p.exists(): print(f"SKIP {proj}", flush=True); continue
        w = torch.load(p, map_location="cpu")
        print(f"\n{'='*60}\nProcessing {proj}: shape={tuple(w.shape)}\n{'='*60}", flush=True)
        res, e2, e3, e4 = run_poc(w, proj, dev, ext, ghd, tcp, tcpi, qtf, cbs)
        all_res[proj] = res
        standalone[proj] = {"K2": e2, "K3": e3, "K4": e4}
        print(f"\n  Standalone: K2={e2:.4e} K3={e3:.4e} K4={e4:.4e}", flush=True)

        # Sort by bits, then by gap
        print(f"\n  All paths (sorted by gap_to_K4 desc):", flush=True)
        for lb in sorted(res.keys(), key=lambda x: res[x]["gap_to_K4"], reverse=True)[:25]:
            m = res[lb]
            print(f"    {lb:30s}: MSE={m['mse']:.6e}  bits={m['bits']:.3f}  "
                  f"gap_K4={m['gap_to_K4']:.1%}  cos={m['cos']:.6f}", flush=True)

    # Aggregate
    print(f"\n{'='*60}\nAGGREGATE\n{'='*60}", flush=True)
    agg = {}
    for pr, res in all_res.items():
        for lb, m in res.items():
            agg.setdefault(lb, {"mse": [], "cos": [], "gap_to_K4": [], "bits": []})
            agg[lb]["mse"].append(m["mse"])
            agg[lb]["cos"].append(m["cos"])
            agg[lb]["gap_to_K4"].append(m["gap_to_K4"])
            agg[lb]["bits"].append(m["bits"])

    # Average standalone
    e2 = sum(standalone[p]["K2"] for p in standalone) / len(standalone)
    e3 = sum(standalone[p]["K3"] for p in standalone) / len(standalone)
    e4 = sum(standalone[p]["K4"] for p in standalone) / len(standalone)
    print(f"\n  Standalone: K2={e2:.4e} K3={e3:.4e} K4={e4:.4e}", flush=True)

    print(f"\n  All paths (sorted by gap_to_K4 desc, top 30):", flush=True)
    for lb in sorted(agg.keys(), key=lambda x: sum(agg[x]["gap_to_K4"])/len(agg[x]["gap_to_K4"]), reverse=True)[:30]:
        mse = sum(agg[lb]["mse"])/len(agg[lb]["mse"])
        cos = sum(agg[lb]["cos"])/len(agg[lb]["cos"])
        gc = sum(agg[lb]["gap_to_K4"])/len(agg[lb]["gap_to_K4"])
        bt = sum(agg[lb]["bits"])/len(agg[lb]["bits"])
        print(f"    {lb:30s}: MSE={mse:.6e}  bits={bt:.3f}  gap_K4={gc:.1%}  cos={cos:.6f}", flush=True)

    # Pareto analysis: best MSE at each bit level
    print(f"\n  Pareto frontier (best MSE per ~0.5-bit bucket):", flush=True)
    bit_buckets = {}
    for lb in agg:
        bt = sum(agg[lb]["bits"])/len(agg[lb]["bits"])
        bucket = round(bt * 2) / 2  # 0.5-bit granularity
        mse = sum(agg[lb]["mse"])/len(agg[lb]["mse"])
        if bucket not in bit_buckets or mse < bit_buckets[bucket][1]:
            bit_buckets[bucket] = (lb, mse, bt)
    for bucket in sorted(bit_buckets.keys()):
        lb, mse, bt = bit_buckets[bucket]
        print(f"    ~{bucket:.1f} bits: {lb:30s}  MSE={mse:.6e}  (actual {bt:.3f} bits)", flush=True)

    out = {"poc": "additive_residual_v4b_sparse_deep_dive", "gpu": torch.cuda.get_device_name(0),
           "per_tensor": all_res, "standalone": {p: standalone[p] for p in standalone}}
    Path(args.out).write_text(json.dumps(out, indent=2, default=str))
    print(f"\nResults saved to {args.out}", flush=True)

if __name__ == "__main__":
    main()
