#!/usr/bin/env python3
"""PoC v3: Optimization opportunities for additive residual encoding.

Tests:
  1. Lloyd-Max 2-bit quantizer (optimal for Gaussian residuals)
  2. Low-rank cross-expert residual (does the layer have shared error structure?)
  3. Layer-level common-mode residual (mean residual across experts)
  4. K-means trained codebook on residuals
  5. Per-projection vs global scale (already known: negligible)
"""

from __future__ import annotations
import argparse, json, math, os, sys, types, importlib.util, time
from pathlib import Path
import torch, torch.nn.functional as F
import numpy as np

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
    sv = (torch.randn(n, generator=g).sign() + 1e-5).sign().float().to(device).unsqueeze(0)
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


# ---------------------------------------------------------------------------
# Residual quantizers
# ---------------------------------------------------------------------------

def q1b_scalar(r):
    """1-bit scalar: sign × mean(|r|). Optimal for zero-mean Gaussian."""
    s = r.abs().mean().item()
    return torch.zeros_like(r) if s < 1e-12 else torch.sign(r) * s


def q2b_uniform(r):
    """2-bit uniform: evenly spaced levels. Baseline."""
    max_abs = r.abs().max().item()
    if max_abs < 1e-12: return torch.zeros_like(r)
    # 4 levels: -3/2, -1/2, 1/2, 3/2 × step
    step = 2 * max_abs / 3
    q = torch.round(r / step) * step
    # Clamp to 4 levels
    q = torch.clamp(q, -1.5 * step, 1.5 * step)
    return q


def q2b_lloyd_max(r):
    """2-bit Lloyd-Max: optimal 4-level quantizer for Gaussian distribution.
    Levels at ±0.45σ and ±1.51σ (symmetric, zero-mean Gaussian)."""
    sigma = r.std().item()
    if sigma < 1e-12: return torch.zeros_like(r)
    # Optimal levels for unit Gaussian: ±0.4528, ±1.5104
    levels = torch.tensor([-1.5104, -0.4528, 0.4528, 1.5104], device=r.device) * sigma
    # Assign each element to nearest level
    dists = torch.cdist(r.flatten().unsqueeze(1), levels.unsqueeze(1))
    idx = dists.argmin(dim=1)
    q = levels[idx].reshape(r.shape)
    return q


def q2b_kmeans(r, n_iter=20):
    """2-bit k-means: train 4 clusters on actual residual distribution."""
    flat = r.flatten()
    if flat.abs().max() < 1e-12: return torch.zeros_like(r)
    # Initialize with Lloyd-Max levels
    sigma = flat.std().item()
    centroids = torch.tensor([-1.5104, -0.4528, 0.4528, 1.5104], device=r.device) * sigma
    for _ in range(n_iter):
        dists = torch.cdist(flat.unsqueeze(1), centroids.unsqueeze(1))
        idx = dists.argmin(dim=1)
        for c in range(4):
            mask = idx == c
            if mask.any():
                centroids[c] = flat[mask].mean()
    dists = torch.cdist(flat.unsqueeze(1), centroids.unsqueeze(1))
    idx = dists.argmin(dim=1)
    return centroids[idx].reshape(r.shape)


def q1b_lloyd_max(r):
    """1-bit Lloyd-Max: optimal 2-level for Gaussian = sign × σ√(2/π) = sign × mean(|r|).
    This is identical to q1b_scalar — confirms scalar is already optimal for 1-bit."""
    return q1b_scalar(r)  # same thing


def metrics(w_ref, w_ap):
    e = w_ref - w_ap
    return {"mse": e.pow(2).mean().item(),
            "cos": F.cosine_similarity(w_ref.flatten().unsqueeze(0), w_ap.flatten().unsqueeze(0), dim=1).item()}


def run_single_expert(w, device, ghd, tcp, tcpi, qtf, cbs):
    """Run all residual variants on a single expert tensor."""
    w = w.to(device).float()
    w_reg, su, sv = regularize(w, device, ghd, cbs)

    qk = {}
    for K in [2, 3, 4]:
        qk[K] = quantize_trellis(w_reg, K, device, tcp, tcpi, qtf)
        torch.cuda.synchronize()

    r23 = w_reg - qk[2]  # K2→K3 residual
    r34 = w_reg - qk[3]  # K3→K4 residual

    paths = {}
    for K in [2, 3, 4]: paths[f"K{K}_standalone"] = qk[K]

    # 1-bit scalar (baseline)
    paths["K3_2+1s"] = qk[2] + q1b_scalar(r23)
    paths["K4_3+1s"] = qk[3] + q1b_scalar(r34)

    # 2-bit approaches for K2→K4 direct
    r24 = w_reg - qk[2]
    paths["K4_2+2uniform"] = qk[2] + q2b_uniform(r24)
    paths["K4_2+2lloyd"] = qk[2] + q2b_lloyd_max(r24)
    paths["K4_2+2kmeans"] = qk[2] + q2b_kmeans(r24)

    # Chained 1+1 scalar
    rch = w_reg - (qk[2] + q1b_scalar(r23))
    paths["K4_2+1s+1s"] = qk[2] + q1b_scalar(r23) + q1b_scalar(rch)

    # Chained 1+1 with Lloyd-Max second stage
    rch_lm = w_reg - (qk[2] + q1b_scalar(r23))
    paths["K4_2+1s+1lm"] = qk[2] + q1b_scalar(r23) + q1b_lloyd_max(rch_lm)

    # 2-bit Lloyd-Max for K3→K4 (from true K3)
    paths["K4_3+2lm"] = qk[3] + q2b_lloyd_max(r34)

    res = {label: metrics(w_reg, w_ap) for label, w_ap in paths.items()}
    return w_reg, qk, r23, r34, res


def run_cross_expert_analysis(data_dir, device, ghd, tcp, tcpi, qtf, cbs):
    """Analyze cross-expert residual structure for low-rank opportunities."""
    print(f"\n{'='*60}", flush=True)
    print("CROSS-EXPERT ANALYSIS (layer-level sharing)", flush=True)
    print(f"{'='*60}", flush=True)

    # We only have 1 expert (137), so we can't do true cross-expert analysis.
    # But we CAN analyze the 3 projections (gate/up/down) as if they were
    # "3 experts" to see if residuals share structure.
    residuals = {}
    regs = {}
    qk2_all = {}
    for proj in ["gate_proj", "up_proj", "down_proj"]:
        p = data_dir / f"{proj}.pt"
        if not p.exists(): continue
        w = torch.load(p, map_location="cpu")
        w_reg, su, sv = regularize(w.to(device).float(), device, ghd, cbs)
        qk2 = quantize_trellis(w_reg, 2, device, tcp, tcpi, qtf)
        torch.cuda.synchronize()
        r = w_reg - qk2
        residuals[proj] = r.flatten()
        regs[proj] = w_reg
        qk2_all[proj] = qk2
        print(f"  {proj}: residual std={r.std():.4f} mean={r.mean():.6f}", flush=True)

    if len(residuals) < 2:
        print("  Need ≥2 projections for cross-expert analysis", flush=True)
        return None

    # Stack residuals: (n_projections, numel)
    R = torch.stack(list(residuals.values()))  # (3, d)
    n_experts, d = R.shape
    print(f"\n  Residual matrix: {R.shape}", flush=True)

    # SVD to find rank structure
    U, S, Vh = torch.linalg.svd(R, full_matrices=False)
    total_var = S.pow(2).sum().item()
    print(f"\n  Singular values: {['%.4f' % s for s in S.tolist()]}", flush=True)
    for r in [1, 2, 3]:
        var_captured = S[:r].pow(2).sum().item() / total_var
        print(f"  Rank-{r} captures {var_captured:.1%} of residual variance", flush=True)

    # Low-rank reconstruction: R ≈ U[:,:r] @ diag(S[:r]) @ Vh[:r,:]
    results = {}
    for r in [1, 2, 3]:
        R_lr = U[:, :r] @ torch.diag(S[:r]) @ Vh[:r, :]
        # Quantize the low-rank factors
        # U[:,:r] is (3, r) — tiny, store as fp16
        # Vh[:r,:] is (r, d) — the big part, 1-bit quantize each row
        V_rows = Vh[:r, :]  # (r, d)
        V_quant = torch.stack([q1b_scalar(v) for v in V_rows])
        R_lr_q = U[:, :r] @ torch.diag(S[:r]) @ V_quant
        mse_lr = (R - R_lr_q).pow(2).mean().item()
        mse_direct = (R - R).pow(2).mean().item()  # 0
        # Per-expert residual MSE with low-rank
        for i, proj in enumerate(list(residuals.keys())):
            r_orig = residuals[proj]
            r_approx = R_lr_q[i]
            w_reg = regs[proj]
            qk2 = qk2_all[proj]
            w_approx = qk2 + r_approx
            m = metrics(w_reg, w_approx)
            results[f"K3_lowrank{r}_{proj}"] = m
            print(f"  K3 rank-{r} {proj}: MSE={m['mse']:.6e} cos={m['cos']:.6f}", flush=True)

    # Common-mode (mean residual)
    r_mean = R.mean(dim=0)
    r_mean_q = q1b_scalar(r_mean)
    for i, proj in enumerate(list(residuals.keys())):
        w_reg = regs[proj]; qk2 = qk2_all[proj]
        w_approx = qk2 + r_mean_q
        m = metrics(w_reg, w_approx)
        results[f"K3_commonmode_{proj}"] = m
        print(f"  K3 common-mode {proj}: MSE={m['mse']:.6e} cos={m['cos']:.6f}", flush=True)

    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--data-dir", default="/tmp/poc_residual/data")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    dev = torch.device(args.device)
    print(f"Device: {dev}  GPU: {torch.cuda.get_device_name(0)}", flush=True)
    ext, ghd, tcp, tcpi, qtf, cbs = _bootstrap()

    dd = Path(args.data_dir)
    all_res = {}
    for proj in ["gate_proj", "up_proj", "down_proj"]:
        p = dd / f"{proj}.pt"
        if not p.exists(): print(f"SKIP {proj}", flush=True); continue
        w = torch.load(p, map_location="cpu")
        print(f"\n{'='*60}\nProcessing {proj}: shape={tuple(w.shape)}\n{'='*60}", flush=True)
        _, _, _, _, res = run_single_expert(w, dev, ghd, tcp, tcpi, qtf, cbs)
        all_res[proj] = res
        e2 = res["K2_standalone"]["mse"]; e3 = res["K3_standalone"]["mse"]; e4 = res["K4_standalone"]["mse"]
        g23 = e2 - e3; g34 = e3 - e4; g24 = e2 - e4
        print(f"\n  Standalone: K2={e2:.4e} K3={e3:.4e} ({e2/e3:.2f}x) K4={e4:.4e} ({e3/e4:.2f}x)", flush=True)
        print(f"\n  Residual paths:", flush=True)
        for lb in sorted(res.keys()):
            if lb in ("K2_standalone", "K3_standalone", "K4_standalone"): continue
            m = res[lb]; mse = m["mse"]
            if lb.startswith("K3_"): gc = (e2 - mse) / g23 if g23 > 0 else 0
            elif lb.startswith("K4_3+"): gc = (e3 - mse) / g34 if g34 > 0 else 0
            elif lb.startswith("K4_2+"): gc = (e2 - mse) / g24 if g24 > 0 else 0
            else: gc = 0
            print(f"    {lb:20s}: MSE={mse:.6e}  cos={m['cos']:.6f}  gap={gc:.1%}", flush=True)

    # Aggregate
    print(f"\n{'='*60}\nAGGREGATE\n{'='*60}", flush=True)
    agg = {}; eps_a = {}
    for pr, res in all_res.items():
        for lb, m in res.items():
            agg.setdefault(lb, {"mse": [], "cos": []})
            agg[lb]["mse"].append(m["mse"]); agg[lb]["cos"].append(m["cos"])
        for K in [2, 3, 4]:
            eps_a.setdefault(f"K{K}", []).append(res[f"K{K}_standalone"]["mse"])
    e2 = sum(eps_a["K2"])/len(eps_a["K2"]); e3 = sum(eps_a["K3"])/len(eps_a["K3"]); e4 = sum(eps_a["K4"])/len(eps_a["K4"])
    print(f"\n  Standalone: K2={e2:.4e} K3={e3:.4e} K4={e4:.4e}", flush=True)
    print(f"\n  All paths (mean MSE, gap closed):", flush=True)
    for lb in sorted(agg.keys()):
        if lb in ("K2_standalone", "K3_standalone", "K4_standalone"): continue
        mse = sum(agg[lb]["mse"])/len(agg[lb]["mse"])
        cos = sum(agg[lb]["cos"])/len(agg[lb]["cos"])
        if lb.startswith("K3_"): gc = (e2 - mse) / (e2 - e3) if e2 > e3 else 0
        elif lb.startswith("K4_3+"): gc = (e3 - mse) / (e3 - e4) if e3 > e4 else 0
        elif lb.startswith("K4_2+"): gc = (e2 - mse) / (e2 - e4) if e2 > e4 else 0
        else: gc = 0
        print(f"    {lb:20s}: MSE={mse:.6e}  cos={cos:.6f}  gap={gc:.1%}", flush=True)

    # Cross-expert analysis
    cross_res = run_cross_expert_analysis(dd, dev, ghd, tcp, tcpi, qtf, cbs)

    out = {"poc": "additive_residual_v3", "quantizer": "real EXL3 Viterbi trellis",
           "gpu": torch.cuda.get_device_name(0), "per_tensor": all_res,
           "cross_expert": cross_res}
    op = args.out or "results_v3.json"; Path(op).write_text(json.dumps(out, indent=2, default=str))
    print(f"\nResults saved to {op}", flush=True)

if __name__ == "__main__":
    main()
