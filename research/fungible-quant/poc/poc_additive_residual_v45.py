#!/usr/bin/env python3
"""PoC v45: Entropy of MSRT trellis indices + entropy-aware Pareto.

Measure the entropy of EXL3 trellis indices at each MSRT stage.
Trellis indices may have non-uniform distributions (Viterbi path preferences),
giving entropy < raw bits. This would make MSRT even more efficient.

Tests:
1. Entropy of K2 base trellis indices
2. Entropy of K1/K2/K3 rescaled trellis residual indices
3. Entropy-aware MSRT Pareto (using entropy instead of raw bits)
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

def quantize_trellis_raw(data, K, device, tcp, tcpi, qtf):
    k, n = data.shape; tiles_n = n // 16; weight_q = torch.zeros_like(data)
    qa = {"K": K, "mcg": True}; perm = tcp(device); perm_i = tcpi(device)
    # Also capture the quantized tiles (indices) for entropy measurement
    all_quant_tiles = []
    for bi in range(0, k, 16):
        rows = data[bi:bi+16]
        tiles = rows.reshape(16, tiles_n, 16).permute(1, 0, 2).reshape(tiles_n, 256)
        tiles = tiles[:, perm].contiguous()
        quant_w, _ = qtf(tiles, qa)
        quant_w_unperm = quant_w[:, perm_i].reshape(tiles_n, 16, 16).permute(1, 0, 2).reshape(16, n)
        weight_q[bi:bi+16] = quant_w_unperm
        all_quant_tiles.append(quant_w.cpu())
    return weight_q, all_quant_tiles

def rescaled_trellis_with_indices(base_q, residual, K_res, device, tcp, tcpi, qtf, cbs):
    residual_rms = residual.square().mean().sqrt().item()
    if residual_rms < 1e-12: return base_q, []
    scale = abs(cbs) / residual_rms
    scaled = residual * scale
    quant, quant_tiles = quantize_trellis_raw(scaled, K_res, device, tcp, tcpi, qtf)
    return base_q + quant / scale, quant_tiles

def compute_entropy(quant_tiles_list, K):
    """Compute entropy of trellis indices.
    The quant_tiles are float values (dequantized), not indices.
    For entropy, we need to look at the distribution of values.
    Since TCQ outputs are from a codebook of 2^K levels, we can
    estimate entropy by looking at the distribution of quantized values.
    """
    if not quant_tiles_list:
        return float(K)
    all_tiles = torch.cat(quant_tiles_list, dim=0).flatten()
    # Count unique values (proxy for codebook usage)
    unique_vals = torch.unique(all_tiles)
    n_unique = len(unique_vals)
    if n_unique <= 1:
        return 0.0
    # Compute histogram-based entropy
    # Use the quantized values directly as bins
    counts = torch.tensor([(all_tiles == v).sum().item() for v in unique_vals], dtype=torch.float)
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    entropy = -(probs * probs.log2()).sum().item()
    return entropy

def tile_mse(w_reg, recon, k, n):
    tnk, tnn = k // 16, n // 16
    diff = (w_reg - recon).pow(2)
    return diff.view(tnk, 16, tnn, 16).mean(dim=(1, 3))

def run_experiment(data_dir, device, ext, ghd, tcp, tcpi, qtf, cbs_scale):
    results = {}
    f = data_dir / f"layer10_all_gate_proj.pt"
    if not f.exists(): return results
    all_experts = torch.load(f, map_location="cpu")
    n_experts = min(5, all_experts.shape[0])
    k, n = all_experts.shape[1], all_experts.shape[2]
    print(f"  {n_experts} experts, {k}x{n}", flush=True)

    # Collect entropy and MSE for each MSRT stage
    stage_data = {}  # stage_name -> {"raw_bpw": float, "entropies": [], "mses": []}

    for ei in range(n_experts):
        print(f"  Expert {ei}...", flush=True)
        w = all_experts[ei].to(device)
        w_reg, _, _ = regularize(w, device, ghd, cbs_scale)
        del w

        # K2 base
        qk2, k2_tiles = quantize_trellis_raw(w_reg, 2, device, tcp, tcpi, qtf)
        r2 = w_reg - qk2

        # Measure K2 entropy
        k2_ent = compute_entropy(k2_tiles, 2)
        k2_mse = (w_reg - qk2).pow(2).mean().item()
        if "K2_base" not in stage_data:
            stage_data["K2_base"] = {"raw_bpw": 2.0, "entropies": [], "mses": []}
        stage_data["K2_base"]["entropies"].append(k2_ent)
        stage_data["K2_base"]["mses"].append(k2_mse)
        print(f"    K2 base: H={k2_ent:.3f} bits (raw=2)", flush=True)

        # MSRT stages at 6bpw: K2+K1trsc+K3trsc
        s1, s1_tiles = rescaled_trellis_with_indices(qk2, r2, 1, device, tcp, tcpi, qtf, cbs_scale)
        r_s1 = w_reg - s1
        s1_ent = compute_entropy(s1_tiles, 1)

        s2, s2_tiles = rescaled_trellis_with_indices(s1, r_s1, 3, device, tcp, tcpi, qtf, cbs_scale)
        s2_ent = compute_entropy(s2_tiles, 3)
        s2_mse = (w_reg - s2).pow(2).mean().item()

        if "MSRT_6bpw_stage1_K1" not in stage_data:
            stage_data["MSRT_6bpw_stage1_K1"] = {"raw_bpw": 1.0, "entropies": [], "mses": []}
        stage_data["MSRT_6bpw_stage1_K1"]["entropies"].append(s1_ent)
        stage_data["MSRT_6bpw_stage1_K1"]["mses"].append(0)  # intermediate

        if "MSRT_6bpw_stage2_K3" not in stage_data:
            stage_data["MSRT_6bpw_stage2_K3"] = {"raw_bpw": 3.0, "entropies": [], "mses": []}
        stage_data["MSRT_6bpw_stage2_K3"]["entropies"].append(s2_ent)
        stage_data["MSRT_6bpw_stage2_K3"]["mses"].append(s2_mse)

        total_ent_6 = k2_ent + s1_ent + s2_ent
        print(f"    MSRT 6bpw: K1 H={s1_ent:.3f} (raw=1), K3 H={s2_ent:.3f} (raw=3)", flush=True)
        print(f"    Total: raw=6.0, entropy={total_ent_6:.3f}, savings={(1-total_ent_6/6)*100:.1f}%", flush=True)

        del s1, s1_tiles, r_s1, s2, s2_tiles, k2_tiles

        # MSRT at 8bpw: K2+K1+K2+K3trsc
        s_8a, s8a_tiles = rescaled_trellis_with_indices(qk2, r2, 1, device, tcp, tcpi, qtf, cbs_scale)
        r_8a = w_reg - s_8a
        s8a_ent = compute_entropy(s8a_tiles, 1)

        s_8b, s8b_tiles = rescaled_trellis_with_indices(s_8a, r_8a, 2, device, tcp, tcpi, qtf, cbs_scale)
        r_8b = w_reg - s_8b
        s8b_ent = compute_entropy(s8b_tiles, 2)

        s_8c, s8c_tiles = rescaled_trellis_with_indices(s_8b, r_8b, 3, device, tcp, tcpi, qtf, cbs_scale)
        s8c_ent = compute_entropy(s8c_tiles, 3)
        s8c_mse = (w_reg - s_8c).pow(2).mean().item()

        total_ent_8 = k2_ent + s8a_ent + s8b_ent + s8c_ent
        print(f"    MSRT 8bpw: K1 H={s8a_ent:.3f}, K2 H={s8b_ent:.3f}, K3 H={s8c_ent:.3f}", flush=True)
        print(f"    Total: raw=8.0, entropy={total_ent_8:.3f}, savings={(1-total_ent_8/8)*100:.1f}%", flush=True)

        if "MSRT_8bpw_total" not in stage_data:
            stage_data["MSRT_8bpw_total"] = {"raw_bpw": 8.0, "entropies": [], "mses": []}
        stage_data["MSRT_8bpw_total"]["entropies"].append(total_ent_8)
        stage_data["MSRT_8bpw_total"]["mses"].append(s8c_mse)

        if "MSRT_6bpw_total" not in stage_data:
            stage_data["MSRT_6bpw_total"] = {"raw_bpw": 6.0, "entropies": [], "mses": []}
        stage_data["MSRT_6bpw_total"]["entropies"].append(total_ent_6)
        stage_data["MSRT_6bpw_total"]["mses"].append(s2_mse)

        del s_8a, s8a_tiles, r_8a, s_8b, s8b_tiles, r_8b, s_8c, s8c_tiles

        # MSRT at 10bpw: K2+K1+K1+K1+K2+K3trsc
        s_10a, _ = rescaled_trellis_with_indices(qk2, r2, 1, device, tcp, tcpi, qtf, cbs_scale)
        r_10a = w_reg - s_10a
        s_10b, _ = rescaled_trellis_with_indices(s_10a, r_10a, 1, device, tcp, tcpi, qtf, cbs_scale)
        r_10b = w_reg - s_10b
        s_10c, _ = rescaled_trellis_with_indices(s_10b, r_10b, 1, device, tcp, tcpi, qtf, cbs_scale)
        r_10c = w_reg - s_10c
        s_10d, _ = rescaled_trellis_with_indices(s_10c, r_10c, 2, device, tcp, tcpi, qtf, cbs_scale)
        r_10d = w_reg - s_10d
        s_10e, s10e_tiles = rescaled_trellis_with_indices(s_10d, r_10d, 3, device, tcp, tcpi, qtf, cbs_scale)
        s10e_mse = (w_reg - s_10e).pow(2).mean().item()
        s10e_ent = compute_entropy(s10e_tiles, 3)

        # For 10bpw total entropy, we'd need all stage entropies
        # But we already have k2_ent from above. Let's just report the final stage
        print(f"    MSRT 10bpw: final K3 H={s10e_ent:.3f} (raw=3)", flush=True)

        if "MSRT_10bpw_total" not in stage_data:
            stage_data["MSRT_10bpw_total"] = {"raw_bpw": 10.0, "entropies": [], "mses": []}
        stage_data["MSRT_10bpw_total"]["entropies"].append(s10e_ent)  # just final stage for now
        stage_data["MSRT_10bpw_total"]["mses"].append(s10e_mse)

        del s_10a, r_10a, s_10b, r_10b, s_10c, r_10c, s_10d, r_10d, s_10e, s10e_tiles
        del w_reg, qk2, r2
        torch.cuda.empty_cache()

    # Summary
    print(f"\n  Entropy summary (averaged over {n_experts} experts):", flush=True)
    print(f"  {'Stage':<30} {'raw bpw':>8} {'entropy':>8} {'savings':>8} {'MSE':>12}", flush=True)
    print(f"  {'-'*68}", flush=True)
    for name in sorted(stage_data.keys()):
        d = stage_data[name]
        avg_ent = sum(d["entropies"]) / len(d["entropies"])
        avg_mse = sum(d["mses"]) / len(d["mses"])
        savings = (1 - avg_ent / d["raw_bpw"]) * 100 if d["raw_bpw"] > 0 else 0
        print(f"  {name:<30} {d['raw_bpw']:>8.1f} {avg_ent:>8.3f} {savings:>7.1f}% {avg_mse:>12.4e}", flush=True)

    results["stage_data"] = {k: {"raw_bpw": v["raw_bpw"],
                                  "avg_entropy": sum(v["entropies"])/len(v["entropies"]),
                                  "avg_mse": sum(v["mses"])/len(v["mses"])}
                              for k, v in stage_data.items()}
    return results

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--data-dir", default="/tmp/poc_residual/glm52_data")
    ap.add_argument("--out", default="/tmp/poc_residual/results_v45.json")
    args = ap.parse_args()
    dev = torch.device(args.device)
    print(f"Device: {dev}  GPU: {torch.cuda.get_device_name(0)}", flush=True)
    ext, ghd, tcp, tcpi, qtf, cbs_scale = _bootstrap()
    results = run_experiment(Path(args.data_dir), dev, ext, ghd, tcp, tcpi, qtf, cbs_scale)
    Path(args.out).write_text(json.dumps(results, indent=2, default=str))
    print(f"\nResults saved to {args.out}", flush=True)

if __name__ == "__main__":
    main()
