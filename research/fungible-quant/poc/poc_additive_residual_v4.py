#!/usr/bin/env python3
"""PoC v4: Comprehensive optimization opportunities for additive residual encoding.

Tests all 6 ideas from the literature research:
  #6 Hessian-weighted residual scale
  #2 Adaptive lattice (grid-searched α₁, α₂ per group)
  #3 Low-rank residual subspace (SVD within expert)
  #5 Sparse residual (top-k% at 8-bit, rest binarized)
  #7 Multi-codebook additive (AQLM-style, 2 codebooks × 4 entries)
  #1 Matryoshka approximation (alternating optimization)

Uses real EXL3 Viterbi trellis on RTX 5090.
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

def make_hessian(k_in, device, n_samples=4096):
    x = torch.randn(n_samples, k_in, device=device, dtype=torch.float32)
    return x.T @ x  # (k, k)

# ---------------------------------------------------------------------------
# Baseline residual quantizers
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Idea #6: Hessian-weighted residual scale
# ---------------------------------------------------------------------------

def q1b_hessian_weighted(r, H, device):
    """Optimal 1-bit scale under Hessian-weighted MSE.
    s = trace(sign(r) @ H @ r^T) / trace(sign(r) @ H @ sign(r)^T)
    = sum(sign(r) * (H @ r)) / sum(sign(r) * (H @ sign(r)))
    """
    signs = torch.sign(r)
    s_vals = r.abs().mean().item()
    if s_vals < 1e-12: return torch.zeros_like(r)
    Hr = H @ r  # (k, k) @ (k, n) = (k, n)
    Hs = H @ signs  # (k, n)
    num = (signs * Hr).sum().item()
    den = (signs * Hs).sum().item()
    s = num / max(den, 1e-12)
    return signs * s

# ---------------------------------------------------------------------------
# Idea #2: Adaptive lattice (grid-searched α₁, α₂)
# ---------------------------------------------------------------------------

def q2b_adaptive_lattice(r, group_size=128):
    """2-bit with per-group optimal α₁, α₂ via grid search.
    r̂ = α₁ * sign(r) + α₂ * sign(r - α₁*sign(r))
    """
    k, n = r.shape
    result = torch.zeros_like(r)
    sigma = r.std().item()
    if sigma < 1e-12: return result
    # Search grid for α₁, α₂ (in units of σ)
    alphas = [0.3, 0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5]
    betas = [0.1, 0.2, 0.3, 0.5, 0.7]
    for gi in range(0, k, group_size):
        for gj in range(0, n, group_size):
            block = r[gi:gi+group_size, gj:gj+group_size]
            best_mse = float('inf')
            best_recon = torch.zeros_like(block)
            for a in alphas:
                a_val = a * sigma
                r1 = a_val * torch.sign(block)
                r_rem = block - r1
                for b in betas:
                    b_val = b * sigma
                    r2 = b_val * torch.sign(r_rem)
                    recon = r1 + r2
                    mse = (block - recon).pow(2).mean().item()
                    if mse < best_mse:
                        best_mse = mse
                        best_recon = recon
            result[gi:gi+group_size, gj:gj+group_size] = best_recon
    return result

# ---------------------------------------------------------------------------
# Idea #3: Low-rank residual subspace
# ---------------------------------------------------------------------------

def q_lowrank_plus_1bit(r, rank=4, device="cuda:0"):
    """Decompose residual as rank-r approximation (8-bit) + 1-bit scalar remainder.
    Memory: rank*(k+n)*2 bytes + 1 bit/weight. Overhead ≈ 0.
    """
    k, n = r.shape
    # SVD on the residual (k×n)
    U, S, Vh = torch.linalg.svd(r, full_matrices=False)
    # Rank-r approximation
    r_lowrank = U[:, :rank] @ torch.diag(S[:rank]) @ Vh[:rank, :]
    # 1-bit scalar on the remainder
    r_rem = r - r_lowrank
    r_rem_1bit = q1b_scalar(r_rem)
    return r_lowrank + r_rem_1bit, rank

# ---------------------------------------------------------------------------
# Idea #5: Sparse residual
# ---------------------------------------------------------------------------

def q_sparse_1bit(r, top_pct=2.0):
    """Top-k% at 8-bit (fp16), rest binarized.
    Memory: top_pct%*8 + (1-top_pct%)*1 = 1 + 7*top_pct%/100 bits/weight
    """
    flat = r.flatten()
    sigma = flat.std().item()
    if sigma < 1e-12: return torch.zeros_like(r)
    k = max(1, int(flat.numel() * top_pct / 100))
    # Find top-k by absolute value
    topk_vals, topk_idx = flat.abs().topk(k)
    # Create sparse correction
    result = torch.zeros_like(flat, dtype=r.dtype)
    # 1-bit for everything
    s = flat.abs().mean().item()
    result = torch.sign(flat) * s
    # Override top-k with exact values (fp16 precision)
    result[topk_idx] = flat[topk_idx].to(torch.float16).float()
    return result.reshape(r.shape)

# ---------------------------------------------------------------------------
# Idea #7: Multi-codebook additive (AQLM-style)
# ---------------------------------------------------------------------------

def q_multicodebook(r, M=2, codebook_bits=2, group_size=128, n_iter=10):
    """M additive codebooks, each with 2^codebook_bits entries.
    Each group of `group_size` weights is sum of M codebook lookups.
    """
    k, n = r.shape
    flat = r.flatten()
    n_groups = flat.numel() // group_size
    groups = flat[:n_groups * group_size].reshape(n_groups, group_size)
    codebook_size = 2 ** codebook_bits
    # Initialize codebooks with random centroids from the data
    codebooks = []
    for m in range(M):
        idx = torch.randint(0, n_groups, (codebook_size,))
        cb = groups[idx].clone() / M  # split the signal across codebooks
        codebooks.append(cb)
    # Assignments: for each group, pick the best codeword from each codebook
    assignments = torch.zeros(n_groups, M, dtype=torch.long, device=r.device)
    # Greedy alternating optimization
    for iteration in range(n_iter):
        for m in range(M):
            # Fix all codebooks except m, find best assignment for m
            residual = groups.clone()
            for other in range(M):
                if other != m:
                    residual -= codebooks[other][assignments[:, other]]
            # For each group, find nearest codeword in codebook m
            # distances: (n_groups, codebook_size)
            dists = torch.cdist(residual, codebooks[m])
            assignments[:, m] = dists.argmin(dim=1)
            # Update codebook m
            for c in range(codebook_size):
                mask = assignments[:, m] == c
                if mask.any():
                    codebooks[m][c] = groups[mask].mean(dim=0)
                    # Subtract other codebooks' contribution
                    for other in range(M):
                        if other != m:
                            codebooks[m][c] -= codebooks[other][assignments[mask, other]].mean(dim=0)
    # Reconstruct
    recon_flat = torch.zeros_like(flat)
    for m in range(M):
        recon_flat[:n_groups * group_size] += (
            codebooks[m][assignments[:, m]].flatten()
            .reshape(n_groups * group_size)[:len(flat)]
        )
    # Handle remainder if flat.numel() % group_size != 0
    remainder = flat.numel() - n_groups * group_size
    if remainder > 0:
        rem = flat[n_groups * group_size:]
        s = rem.abs().mean().item()
        recon_flat[n_groups * group_size:] = torch.sign(rem) * s if s > 1e-12 else 0
    return recon_flat.reshape(r.shape)

# ---------------------------------------------------------------------------
# Idea #1: Matryoshka approximation (alternating optimization)
# ---------------------------------------------------------------------------

def q_matryoshka_approx(w_reg, qk2, r23, device):
    """Approximate Matryoshka: adjust the K2 reconstruction to minimize
    combined (K2 + 1-bit residual) error, not just K2 error.

    Simple version: after K2 + 1-bit residual, compute the error.
    Then adjust the 1-bit residual to account for the K2 error structure.
    This is equivalent to a single step of alternating optimization.
    """
    # Standard 1-bit residual
    r23_1bit = q1b_scalar(r23)
    w_k3_standard = qk2 + r23_1bit

    # Matryoshka adjustment: instead of quantizing r23 = W - K2,
    # quantize r23' = W - K2 - (K2_error_correlation * r23_1bit_error)
    # This is a correction for the correlation between K2 error and residual
    error_after_1bit = w_reg - w_k3_standard  # remaining error after 2+1

    # Adjust: find a correction term that reduces the remaining error
    # Simple approach: project the remaining error onto the 1-bit residual direction
    # and adjust the scale
    r23_dir = torch.sign(r23)
    projection = (error_after_1bit * r23_dir).sum() / (r23_dir * r23_dir).sum()
    # The adjusted scale accounts for the correlation
    s_original = r23.abs().mean().item()
    s_adjusted = s_original + projection.item()
    r23_adjusted = r23_dir * s_adjusted

    return qk2 + r23_adjusted

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def hessian_weighted_mse(w_ref, w_ap, H):
    """Hessian-weighted MSE: trace(E @ H @ E^T) / trace(W @ H @ W^T)"""
    E = w_ref - w_ap
    num = (E * (H @ E)).sum().item()
    den = (w_ref * (H @ w_ref)).sum().item()
    return num / max(den, 1e-8)

def metrics(w_ref, w_ap, H=None):
    e = w_ref - w_ap
    m = {"mse": e.pow(2).mean().item(),
         "cos": F.cosine_similarity(w_ref.flatten().unsqueeze(0), w_ap.flatten().unsqueeze(0), dim=1).item()}
    if H is not None:
        m["hw_mse"] = hessian_weighted_mse(w_ref, w_ap, H)
    return m

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_poc(w, name, device, ext, ghd, tcp, tcpi, qtf, cbs):
    w = w.to(device).float()
    w_reg, su, sv = regularize(w, device, ghd, cbs)
    H = make_hessian(w.shape[0], device)

    # Standalone trellis
    qk = {}
    for K in [2, 3, 4]:
        qk[K] = quantize_trellis(w_reg, K, device, tcp, tcpi, qtf)
        torch.cuda.synchronize()
        mse = (w_reg - qk[K]).pow(2).mean().item()
        hw = hessian_weighted_mse(w_reg, qk[K], H)
        print(f"    K{K}: MSE={mse:.6e} HW_MSE={hw:.6e}", flush=True)

    r23 = w_reg - qk[2]; r34 = w_reg - qk[3]; r24 = w_reg - qk[2]

    paths = {}
    for K in [2, 3, 4]: paths[f"K{K}_standalone"] = qk[K]

    # Baselines (from v3)
    paths["K3_2+1s"] = qk[2] + q1b_scalar(r23)
    paths["K4_3+1s"] = qk[3] + q1b_scalar(r34)
    paths["K4_2+2lloyd"] = qk[2] + q2b_lloyd(r24)
    rch = w_reg - (qk[2] + q1b_scalar(r23))
    paths["K4_2+1s+1s"] = qk[2] + q1b_scalar(r23) + q1b_scalar(rch)
    paths["K4_3+2lm"] = qk[3] + q2b_lloyd(r34)

    # #6: Hessian-weighted scale
    print("  Computing #6 Hessian-weighted...", flush=True)
    paths["K3_2+1hw"] = qk[2] + q1b_hessian_weighted(r23, H, device)
    paths["K4_3+1hw"] = qk[3] + q1b_hessian_weighted(r34, H, device)
    rch_hw = w_reg - (qk[2] + q1b_hessian_weighted(r23, H, device))
    paths["K4_2+1hw+1hw"] = qk[2] + q1b_hessian_weighted(r23, H, device) + q1b_hessian_weighted(rch_hw, H, device)
    torch.cuda.synchronize()

    # #2: Adaptive lattice
    print("  Computing #2 Adaptive lattice...", flush=True)
    paths["K4_2+2adaptive"] = qk[2] + q2b_adaptive_lattice(r24, group_size=128)
    torch.cuda.synchronize()

    # #3: Low-rank + 1-bit
    print("  Computing #3 Low-rank subspace...", flush=True)
    for rank in [1, 2, 4, 8]:
        recon, _ = q_lowrank_plus_1bit(r23, rank=rank, device=device)
        paths[f"K3_2+lr{rank}+1s"] = qk[2] + recon
        recon4, _ = q_lowrank_plus_1bit(r24, rank=rank, device=device)
        paths[f"K4_2+lr{rank}+1s"] = qk[2] + recon4
    torch.cuda.synchronize()

    # #5: Sparse residual
    print("  Computing #5 Sparse residual...", flush=True)
    for pct in [1.0, 2.0, 5.0]:
        paths[f"K3_2+sparse{pct}%+1s"] = qk[2] + q_sparse_1bit(r23, top_pct=pct)
        paths[f"K4_2+sparse{pct}%+1s"] = qk[2] + q_sparse_1bit(r24, top_pct=pct)
    torch.cuda.synchronize()

    # #7: Multi-codebook additive
    print("  Computing #7 Multi-codebook...", flush=True)
    for M in [2, 4]:
        for cb_bits in [2, 4]:
            paths[f"K3_2+mc_M{M}B{cb_bits}"] = qk[2] + q_multicodebook(r23, M=M, codebook_bits=cb_bits, group_size=128, n_iter=5)
            paths[f"K4_2+mc_M{M}B{cb_bits}"] = qk[2] + q_multicodebook(r24, M=M, codebook_bits=cb_bits, group_size=128, n_iter=5)
    torch.cuda.synchronize()

    # #1: Matryoshka approximation
    print("  Computing #1 Matryoshka approx...", flush=True)
    paths["K3_2+1matryoshka"] = q_matryoshka_approx(w_reg, qk[2], r23, device)
    torch.cuda.synchronize()

    # Measure all
    res = {label: metrics(w_reg, w_ap, H) for label, w_ap in paths.items()}
    return res, w_reg, qk

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--data-dir", default="/tmp/poc_residual/data")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    dev = torch.device(args.device)
    print(f"Device: {dev}  GPU: {torch.cuda.get_device_name(0)}", flush=True)
    ext, ghd, tcp, tcpi, qtf, cbs = _bootstrap()

    dd = Path(args.data_dir); all_res = {}
    for proj in ["gate_proj", "up_proj", "down_proj"]:
        p = dd / f"{proj}.pt"
        if not p.exists(): print(f"SKIP {proj}", flush=True); continue
        w = torch.load(p, map_location="cpu")
        print(f"\n{'='*60}\nProcessing {proj}: shape={tuple(w.shape)}\n{'='*60}", flush=True)
        res, w_reg, qk = run_poc(w, proj, dev, ext, ghd, tcp, tcpi, qtf, cbs)
        all_res[proj] = res
        e2 = res["K2_standalone"]["mse"]; e3 = res["K3_standalone"]["mse"]; e4 = res["K4_standalone"]["mse"]
        g23 = e2 - e3; g34 = e3 - e4; g24 = e2 - e4
        print(f"\n  Standalone: K2={e2:.4e} K3={e3:.4e} K4={e4:.4e}", flush=True)
        print(f"\n  All paths:", flush=True)
        for lb in sorted(res.keys()):
            if lb in ("K2_standalone", "K3_standalone", "K4_standalone"): continue
            m = res[lb]; mse = m["mse"]
            if lb.startswith("K3_"): gc = (e2 - mse) / g23 if g23 > 0 else 0
            elif lb.startswith("K4_3+"): gc = (e3 - mse) / g34 if g34 > 0 else 0
            elif lb.startswith("K4_2+"): gc = (e2 - mse) / g24 if g24 > 0 else 0
            else: gc = 0
            hw = m.get("hw_mse", 0)
            print(f"    {lb:28s}: MSE={mse:.6e}  HW={hw:.6e}  cos={m['cos']:.6f}  gap={gc:.1%}", flush=True)

    # Aggregate
    print(f"\n{'='*60}\nAGGREGATE\n{'='*60}", flush=True)
    agg = {}; eps_a = {}
    for pr, res in all_res.items():
        for lb, m in res.items():
            agg.setdefault(lb, {"mse": [], "cos": [], "hw_mse": []})
            agg[lb]["mse"].append(m["mse"]); agg[lb]["cos"].append(m["cos"])
            if "hw_mse" in m: agg[lb]["hw_mse"].append(m["hw_mse"])
        for K in [2, 3, 4]:
            eps_a.setdefault(f"K{K}", []).append(res[f"K{K}_standalone"]["mse"])
    e2 = sum(eps_a["K2"])/len(eps_a["K2"]); e3 = sum(eps_a["K3"])/len(eps_a["K3"]); e4 = sum(eps_a["K4"])/len(eps_a["K4"])
    g23 = e2 - e3; g34 = e3 - e4; g24 = e2 - e4
    print(f"\n  Standalone: K2={e2:.4e} K3={e3:.4e} K4={e4:.4e}", flush=True)
    print(f"\n  All paths (sorted by gap closed):", flush=True)
    sorted_labels = sorted(agg.keys(), key=lambda lb: (
        (e2 - sum(agg[lb]["mse"])/len(agg[lb]["mse"])) /
        (g23 if lb.startswith("K3_") else (g34 if lb.startswith("K4_3+") else g24))
        if any(s in lb for s in ["K3_", "K4_"]) else 0
    ), reverse=True)
    for lb in sorted_labels:
        if lb in ("K2_standalone", "K3_standalone", "K4_standalone"): continue
        mse = sum(agg[lb]["mse"])/len(agg[lb]["mse"])
        cos = sum(agg[lb]["cos"])/len(agg[lb]["cos"])
        if lb.startswith("K3_"): gc = (e2 - mse) / g23 if g23 > 0 else 0
        elif lb.startswith("K4_3+"): gc = (e3 - mse) / g34 if g34 > 0 else 0
        elif lb.startswith("K4_2+"): gc = (e2 - mse) / g24 if g24 > 0 else 0
        else: gc = 0
        hw = sum(agg[lb].get("hw_mse", [0]))/max(len(agg[lb].get("hw_mse", [1])), 1)
        print(f"    {lb:28s}: MSE={mse:.6e}  HW={hw:.6e}  cos={cos:.6f}  gap={gc:.1%}", flush=True)

    out = {"poc": "additive_residual_v4_comprehensive", "quantizer": "real EXL3 Viterbi trellis",
           "gpu": torch.cuda.get_device_name(0), "per_tensor": all_res,
           "standalone": {"K2": e2, "K3": e3, "K4": e4}}
    op = args.out or "results_v4.json"; Path(op).write_text(json.dumps(out, indent=2, default=str))
    print(f"\nResults saved to {op}", flush=True)

if __name__ == "__main__":
    main()
