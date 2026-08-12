#!/usr/bin/env python3
"""PoC v40: Three-stage rescaled trellis + optimal stage allocation.

v39 showed 2-stage trellis is 17% better than single. Test:
1. Three-stage: K2+K2trsc+K2trsc+K2trsc (8 bpw) vs K2+K6trsc (8 bpw)
2. Three-stage: K2+K1trsc+K2trsc+K3trsc (8 bpw) — different allocation
3. Optimal 2-stage allocation at 7 bpw: K2+K1trsc+K4trsc vs K2+K2trsc+K3trsc
4. Does 2-stage help at 5-6 bpw too?
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
    for bi in range(0, k, 16):
        rows = data[bi:bi+16]
        tiles = rows.reshape(16, tiles_n, 16).permute(1, 0, 2).reshape(tiles_n, 256)
        tiles = tiles[:, perm].contiguous()
        quant_w, _ = qtf(tiles, qa)
        quant_w = quant_w[:, perm_i].reshape(tiles_n, 16, 16).permute(1, 0, 2).reshape(16, n)
        weight_q[bi:bi+16] = quant_w
    return weight_q

def rescaled_trellis(base_q, residual, K_res, device, tcp, tcpi, qtf, cbs):
    residual_rms = residual.square().mean().sqrt().item()
    if residual_rms < 1e-12: return base_q
    scale = abs(cbs) / residual_rms
    scaled = residual * scale
    quant = quantize_trellis_raw(scaled, K_res, device, tcp, tcpi, qtf)
    return base_q + quant / scale

def run_experiment(data_dir, device, ext, ghd, tcp, tcpi, qtf, cbs_scale):
    results = {}
    f = data_dir / f"layer10_all_gate_proj.pt"
    if not f.exists(): return results
    all_experts = torch.load(f, map_location="cpu")
    n_experts = min(5, all_experts.shape[0])
    k, n = all_experts.shape[1], all_experts.shape[2]
    print(f"  {n_experts} experts, {k}x{n}", flush=True)

    all_methods = {}

    for ei in range(n_experts):
        print(f"  Expert {ei}...", flush=True)
        w = all_experts[ei].to(device)
        w_reg, _, _ = regularize(w, device, ghd, cbs_scale)
        del w

        qk2 = quantize_trellis_raw(w_reg, 2, device, tcp, tcpi, qtf)
        r2 = w_reg - qk2

        methods = {}

        # Single-stage baselines
        for Kr in [3, 4, 5, 6]:
            methods[f"K2+K{Kr}trsc ({2+Kr}bpw)"] = (rescaled_trellis(qk2, r2, Kr, device, tcp, tcpi, qtf, cbs_scale), 2.0 + Kr)

        # 2-stage: test all allocations at 7 bpw (total residual = 5 bits)
        for s1, s2 in [(1,4), (2,3), (3,2), (4,1)]:
            recon_s1 = rescaled_trellis(qk2, r2, s1, device, tcp, tcpi, qtf, cbs_scale)
            r_s1 = w_reg - recon_s1
            recon_s2 = rescaled_trellis(recon_s1, r_s1, s2, device, tcp, tcpi, qtf, cbs_scale)
            methods[f"K2+K{s1}trsc+K{s2}trsc (2-stage, {2+s1+s2}bpw)"] = (recon_s2, 2.0 + s1 + s2)

        # 2-stage at 6 bpw (total residual = 4 bits)
        for s1, s2 in [(1,3), (2,2), (3,1)]:
            recon_s1 = rescaled_trellis(qk2, r2, s1, device, tcp, tcpi, qtf, cbs_scale)
            r_s1 = w_reg - recon_s1
            recon_s2 = rescaled_trellis(recon_s1, r_s1, s2, device, tcp, tcpi, qtf, cbs_scale)
            methods[f"K2+K{s1}trsc+K{s2}trsc (2-stage, {2+s1+s2}bpw)"] = (recon_s2, 2.0 + s1 + s2)

        # 2-stage at 8 bpw (total residual = 6 bits)
        for s1, s2 in [(2,4), (3,3), (4,2), (1,5), (5,1)]:
            recon_s1 = rescaled_trellis(qk2, r2, s1, device, tcp, tcpi, qtf, cbs_scale)
            r_s1 = w_reg - recon_s1
            recon_s2 = rescaled_trellis(recon_s1, r_s1, s2, device, tcp, tcpi, qtf, cbs_scale)
            methods[f"K2+K{s1}trsc+K{s2}trsc (2-stage, {2+s1+s2}bpw)"] = (recon_s2, 2.0 + s1 + s2)

        # 3-stage at 8 bpw (total residual = 6 bits)
        for s1, s2, s3 in [(2,2,2), (1,2,3), (1,3,2), (2,1,3), (3,1,2), (1,1,4)]:
            recon_s1 = rescaled_trellis(qk2, r2, s1, device, tcp, tcpi, qtf, cbs_scale)
            r_s1 = w_reg - recon_s1
            recon_s2 = rescaled_trellis(recon_s1, r_s1, s2, device, tcp, tcpi, qtf, cbs_scale)
            r_s2 = w_reg - recon_s2
            recon_s3 = rescaled_trellis(recon_s2, r_s2, s3, device, tcp, tcpi, qtf, cbs_scale)
            methods[f"K2+K{s1}+K{s2}+K{s3}trsc (3-stage, {2+s1+s2+s3}bpw)"] = (recon_s3, 2.0 + s1 + s2 + s3)

        # Compute MSEs
        for name, (recon, bpw) in methods.items():
            mse = (w_reg - recon).pow(2).mean().item()
            if name not in all_methods:
                all_methods[name] = {"mses": [], "bpw": bpw}
            all_methods[name]["mses"].append(mse)
            del recon

        del w_reg, qk2, r2
        torch.cuda.empty_cache()

    # Print sorted by bpw, then by MSE
    print(f"\n  {'Method':<55} {'bpw':>5} {'avg MSE':>12}", flush=True)
    print(f"  {'-'*75}", flush=True)
    for name in sorted(all_methods.keys(), key=lambda x: (all_methods[x]["bpw"], sum(all_methods[x]["mses"])/len(all_methods[x]["mses"]))):
        r = all_methods[name]
        avg = sum(r["mses"]) / len(r["mses"])
        print(f"  {name:<55} {r['bpw']:>5.0f} {avg:>12.4e}", flush=True)

    # Best at each bpw
    print(f"\n  Best at each bpw:", flush=True)
    bpw_best = {}
    for name, d in all_methods.items():
        bpw = int(d["bpw"])
        avg = sum(d["mses"]) / len(d["mses"])
        if bpw not in bpw_best or avg < bpw_best[bpw][1]:
            bpw_best[bpw] = (name, avg)
    for bpw in sorted(bpw_best.keys()):
        name, mse = bpw_best[bpw]
        print(f"    {bpw:>2} bpw: {name:<55} MSE={mse:.4e}", flush=True)

    results["methods"] = {k: {"bpw": v["bpw"], "avg_mse": sum(v["mses"])/len(v["mses"])}
                          for k, v in all_methods.items()}
    results["best_at_bpw"] = {str(bpw): {"name": name, "mse": mse} for bpw, (name, mse) in bpw_best.items()}
    return results

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--data-dir", default="/tmp/poc_residual/glm52_data")
    ap.add_argument("--out", default="/tmp/poc_residual/results_v40.json")
    args = ap.parse_args()
    dev = torch.device(args.device)
    print(f"Device: {dev}  GPU: {torch.cuda.get_device_name(0)}", flush=True)
    ext, ghd, tcp, tcpi, qtf, cbs_scale = _bootstrap()
    results = run_experiment(Path(args.data_dir), dev, ext, ghd, tcp, tcpi, qtf, cbs_scale)
    Path(args.out).write_text(json.dumps(results, indent=2, default=str))
    print(f"\nResults saved to {args.out}", flush=True)

if __name__ == "__main__":
    main()
