#!/usr/bin/env python3
"""PoC v45b: Entropy of MSRT trellis indices (fixed — use quantized_idx).

v45 timed out due to float-based entropy computation.
Fix: qtf returns (quantized_tiles, quantized_idx) where quantized_idx
contains the actual short-encoded trellis indices. Use these directly.
"""
from __future__ import annotations
import argparse, json, math, os, sys, types, importlib.util, time
from pathlib import Path
import torch

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
    spec = importlib.util.spec_from_file_location("exllamav3.ext", f"{EXL3_PKG}/ext.py")
    m = importlib.util.module_from_spec(spec); sys.modules["exllamav3.ext"] = m; spec.loader.exec_module(m)
    ext = m.exllamav3_ext
    spec = importlib.util.spec_from_file_location("exllamav3.util.hadamard", f"{EXL3_PKG}/util/hadamard.py")
    m = importlib.util.module_from_spec(spec); sys.modules["exllamav3.util.hadamard"] = m; spec.loader.exec_module(m)
    ghd = m.get_hadamard_dt
    spec = importlib.util.spec_from_file_location("exllamav3.modules.quant.exl3_lib.quantize", f"{EXL3_PKG}/modules/quant/exl3_lib/quantize.py")
    m = importlib.util.module_from_spec(spec); sys.modules["exllamav3.modules.quant.exl3_lib.quantize"] = m; spec.loader.exec_module(m)
    tcp = m.tensor_core_perm; tcpi = m.tensor_core_perm_i; qtf = m.quantize_tiles; cbs = m.codebook_scale
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

def quantize_trellis_with_idx(data, K, device, tcp, tcpi, qtf):
    """Quantize and return both dequantized values and indices."""
    k, n = data.shape; tiles_n = n // 16; weight_q = torch.zeros_like(data)
    qa = {"K": K, "mcg": True}; perm = tcp(device); perm_i = tcpi(device)
    all_indices = []
    for bi in range(0, k, 16):
        rows = data[bi:bi+16]
        tiles = rows.reshape(16, tiles_n, 16).permute(1, 0, 2).reshape(tiles_n, 256)
        tiles = tiles[:, perm].contiguous()
        quant_w, quant_idx = qtf(tiles, qa)
        quant_w = quant_w[:, perm_i].reshape(tiles_n, 16, 16).permute(1, 0, 2).reshape(16, n)
        weight_q[bi:bi+16] = quant_w
        all_indices.append(quant_idx.cpu())
    return weight_q, all_indices

def rescaled_trellis_with_idx(base_q, residual, K_res, device, tcp, tcpi, qtf, cbs):
    residual_rms = residual.square().mean().sqrt().item()
    if residual_rms < 1e-12: return base_q, []
    scale = abs(cbs) / residual_rms
    scaled = residual * scale
    quant, indices = quantize_trellis_with_idx(scaled, K_res, device, tcp, tcpi, qtf)
    return base_q + quant / scale, indices

def compute_idx_entropy(indices_list):
    """Compute entropy of trellis indices (short values)."""
    if not indices_list:
        return 0.0
    all_idx = torch.cat(indices_list, dim=0).flatten()
    # Use bincount on int values
    all_idx_int = all_idx.int()
    min_val = all_idx_int.min().item()
    max_val = all_idx_int.max().item()
    if min_val == max_val:
        return 0.0
    # Shift to non-negative for bincount
    shifted = all_idx_int - min_val
    counts = torch.bincount(shifted, minlength=max_val - min_val + 1).float()
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return -(probs * probs.log2()).sum().item()

def run_experiment(data_dir, device, ext, ghd, tcp, tcpi, qtf, cbs_scale):
    results = {}
    f = data_dir / f"layer10_all_gate_proj.pt"
    if not f.exists(): return results
    all_experts = torch.load(f, map_location="cpu")
    n_experts = min(5, all_experts.shape[0])
    k, n = all_experts.shape[1], all_experts.shape[2]
    print(f"  {n_experts} experts, {k}x{n}", flush=True)

    stage_entropies = {}  # stage_name -> list of entropies

    for ei in range(n_experts):
        print(f"  Expert {ei}...", flush=True)
        w = all_experts[ei].to(device)
        w_reg, _, _ = regularize(w, device, ghd, cbs_scale)
        del w

        # K2 base
        qk2, k2_idx = quantize_trellis_with_idx(w_reg, 2, device, tcp, tcpi, qtf)
        r2 = w_reg - qk2
        k2_ent = compute_idx_entropy(k2_idx)
        stage_entropies.setdefault("K2_base", []).append(k2_ent)
        print(f"    K2: H={k2_ent:.3f} (raw=2.0)", flush=True)
        del k2_idx

        # MSRT 6bpw: K2+K1+K3trsc
        s1, s1_idx = rescaled_trellis_with_idx(qk2, r2, 1, device, tcp, tcpi, qtf, cbs_scale)
        r_s1 = w_reg - s1
        s1_ent = compute_idx_entropy(s1_idx)
        stage_entropies.setdefault("MSRT_6_K1", []).append(s1_ent)
        del s1_idx

        s2, s2_idx = rescaled_trellis_with_idx(s1, r_s1, 3, device, tcp, tcpi, qtf, cbs_scale)
        s2_ent = compute_idx_entropy(s2_idx)
        stage_entropies.setdefault("MSRT_6_K3", []).append(s2_ent)
        del s2_idx, s1, r_s1, s2

        total_6 = k2_ent + s1_ent + s2_ent
        print(f"    MSRT 6bpw: K1 H={s1_ent:.3f} (raw=1), K3 H={s2_ent:.3f} (raw=3)", flush=True)
        print(f"    Total: raw=6.0, entropy={total_6:.3f}, savings={(1-total_6/6)*100:.1f}%", flush=True)

        # MSRT 8bpw: K2+K1+K2+K3trsc
        s_8a, s8a_idx = rescaled_trellis_with_idx(qk2, r2, 1, device, tcp, tcpi, qtf, cbs_scale)
        r_8a = w_reg - s_8a
        s8a_ent = compute_idx_entropy(s8a_idx)
        stage_entropies.setdefault("MSRT_8_K1", []).append(s8a_ent)
        del s8a_idx

        s_8b, s8b_idx = rescaled_trellis_with_idx(s_8a, r_8a, 2, device, tcp, tcpi, qtf, cbs_scale)
        r_8b = w_reg - s_8b
        s8b_ent = compute_idx_entropy(s8b_idx)
        stage_entropies.setdefault("MSRT_8_K2", []).append(s8b_ent)
        del s8b_idx

        s_8c, s8c_idx = rescaled_trellis_with_idx(s_8b, r_8b, 3, device, tcp, tcpi, qtf, cbs_scale)
        s8c_ent = compute_idx_entropy(s8c_idx)
        stage_entropies.setdefault("MSRT_8_K3", []).append(s8c_ent)
        del s8c_idx, s_8a, r_8a, s_8b, r_8b, s_8c

        total_8 = k2_ent + s8a_ent + s8b_ent + s8c_ent
        print(f"    MSRT 8bpw: K1 H={s8a_ent:.3f} (raw=1), K2 H={s8b_ent:.3f} (raw=2), K3 H={s8c_ent:.3f} (raw=3)", flush=True)
        print(f"    Total: raw=8.0, entropy={total_8:.3f}, savings={(1-total_8/8)*100:.1f}%", flush=True)

        stage_entropies.setdefault("MSRT_6_total", []).append(total_6)
        stage_entropies.setdefault("MSRT_8_total", []).append(total_8)

        del w_reg, qk2, r2
        torch.cuda.empty_cache()

    # Summary
    print(f"\n  Entropy summary (avg over {n_experts} experts):", flush=True)
    print(f"  {'Stage':<20} {'raw bits':>8} {'entropy':>8} {'savings':>8}", flush=True)
    print(f"  {'-'*48}", flush=True)
    for name in sorted(stage_entropies.keys()):
        avg_ent = sum(stage_entropies[name]) / len(stage_entropies[name])
        if name == "K2_base":
            raw = 2.0
        elif "MSRT_6_K1" in name:
            raw = 1.0
        elif "MSRT_6_K3" in name or "MSRT_8_K3" in name:
            raw = 3.0
        elif "MSRT_8_K2" in name:
            raw = 2.0
        elif "MSRT_8_K1" in name:
            raw = 1.0
        elif "6_total" in name:
            raw = 6.0
        elif "8_total" in name:
            raw = 8.0
        else:
            raw = 0.0
        savings = (1 - avg_ent / raw) * 100 if raw > 0 else 0
        print(f"  {name:<20} {raw:>8.1f} {avg_ent:>8.3f} {savings:>7.1f}%", flush=True)

    results["entropies"] = {name: {"avg": sum(v)/len(v), "raw_bits": 
        2.0 if "K2_base" in name else 1.0 if "K1" in name else 3.0 if "K3" in name else 
        2.0 if "8_K2" in name else 6.0 if "6_total" in name else 8.0 if "8_total" in name else 0}
        for name, v in stage_entropies.items()}
    return results

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--data-dir", default="/tmp/poc_residual/glm52_data")
    ap.add_argument("--out", default="/tmp/poc_residual/results_v45b.json")
    args = ap.parse_args()
    dev = torch.device(args.device)
    print(f"Device: {dev}  GPU: {torch.cuda.get_device_name(0)}", flush=True)
    ext, ghd, tcp, tcpi, qtf, cbs_scale = _bootstrap()
    results = run_experiment(Path(args.data_dir), dev, ext, ghd, tcp, tcpi, qtf, cbs_scale)
    Path(args.out).write_text(json.dumps(results, indent=2, default=str))
    print(f"\nResults saved to {args.out}", flush=True)

if __name__ == "__main__":
    main()
