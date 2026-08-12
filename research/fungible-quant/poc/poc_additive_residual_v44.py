#!/usr/bin/env python3
"""PoC v44: MSRT vs RRQ-style RTN + per-row rescaling + cross-layer.

1. RRQ comparison: 2-bit RTN (round-to-nearest) on residual vs rescaled trellis.
   RRQ uses simple RTN, we use rescaled TCQ. How much does TCQ improve over RTN?

2. Per-row rescaling: Instead of global RMS, scale each row of the residual
   independently. Tests finer-grained rescaling.

3. Cross-layer: Test MSRT on layer 40 to verify it works across layers.
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

def per_row_rescaled_trellis(base_q, residual, K_res, device, tcp, tcpi, qtf, cbs):
    """Per-row rescaling: scale each row of residual to |cbs|."""
    k, n = residual.shape
    row_rms = block_rms(residual, dim=1, keepdim=True).clamp(min=1e-12)
    scale = abs(cbs) / row_rms  # (k, 1)
    scaled = residual * scale
    quant = quantize_trellis_raw(scaled, K_res, device, tcp, tcpi, qtf)
    return base_q + quant / scale

def rtn_quantize(residual, n_bits, device):
    """Round-to-nearest quantization (RRQ-style). N-bit uniform."""
    n_levels = 2 ** n_bits
    max_val = residual.abs().max().item()
    if max_val < 1e-12: return torch.zeros_like(residual)
    step = 2 * max_val / (n_levels - 1)
    quantized = torch.round(residual / step) * step
    return quantized

def rescaled_rtn(base_q, residual, n_bits, device):
    """RRQ-style: rescale residual, RTN quantize, scale back."""
    residual_rms = residual.square().mean().sqrt().item()
    if residual_rms < 1e-12: return base_q
    scale = 1.0 / residual_rms  # normalize to unit RMS
    scaled = residual * scale
    quant = rtn_quantize(scaled, n_bits, device)
    return base_q + quant / scale

def run_experiment(data_dir, device, ext, ghd, tcp, tcpi, qtf, cbs_scale):
    results = {}
    all_methods = {}

    for layer in [10, 40]:
        f = data_dir / f"layer{layer}_all_gate_proj.pt"
        if not f.exists(): continue
        all_experts = torch.load(f, map_location="cpu")
        n_experts = min(5, all_experts.shape[0])
        k, n = all_experts.shape[1], all_experts.shape[2]
        print(f"\n  Layer {layer}: {n_experts} experts, {k}x{n}", flush=True)

        layer_methods = {}

        for ei in range(n_experts):
            print(f"    Expert {ei}...", flush=True)
            w = all_experts[ei].to(device)
            w_reg, _, _ = regularize(w, device, ghd, cbs_scale)
            del w

            qk2 = quantize_trellis_raw(w_reg, 2, device, tcp, tcpi, qtf)
            r2 = w_reg - qk2

            methods = {}

            # MSRT best (from v41)
            # 6bpw: K2+K1trsc+K3trsc
            s1 = rescaled_trellis(qk2, r2, 1, device, tcp, tcpi, qtf, cbs_scale)
            r_s1 = w_reg - s1
            methods["MSRT_6bpw"] = (rescaled_trellis(s1, r_s1, 3, device, tcp, tcpi, qtf, cbs_scale), 6.0)

            # 8bpw: K2+K1+K2+K3trsc
            s_8a = rescaled_trellis(qk2, r2, 1, device, tcp, tcpi, qtf, cbs_scale)
            r_8a = w_reg - s_8a
            s_8b = rescaled_trellis(s_8a, r_8a, 2, device, tcp, tcpi, qtf, cbs_scale)
            r_8b = w_reg - s_8b
            methods["MSRT_8bpw"] = (rescaled_trellis(s_8b, r_8b, 3, device, tcp, tcpi, qtf, cbs_scale), 8.0)

            # === RRQ-style: RTN on residual ===
            # 6bpw: K2 + 2-bit RTN + 2-bit RTN
            rrq_s1 = rescaled_rtn(qk2, r2, 2, device)
            r_rrq1 = w_reg - rrq_s1
            methods["RRQ_6bpw"] = (rescaled_rtn(rrq_s1, r_rrq1, 2, device), 6.0)

            # 8bpw: K2 + 2-bit RTN + 2-bit RTN + 2-bit RTN
            rrq_s2 = rescaled_rtn(rrq_s1, r_rrq1, 2, device)
            r_rrq2 = w_reg - rrq_s2
            methods["RRQ_8bpw"] = (rescaled_rtn(rrq_s2, r_rrq2, 2, device), 8.0)

            # === Per-row rescaled MSRT ===
            # 6bpw: K2 + K1 per-row + K3 per-row
            pr_s1 = per_row_rescaled_trellis(qk2, r2, 1, device, tcp, tcpi, qtf, cbs_scale)
            r_pr1 = w_reg - pr_s1
            methods["MSRT_perrow_6bpw"] = (per_row_rescaled_trellis(pr_s1, r_pr1, 3, device, tcp, tcpi, qtf, cbs_scale), 6.0)

            # 8bpw: K2 + K1 per-row + K2 per-row + K3 per-row
            pr_8a = per_row_rescaled_trellis(qk2, r2, 1, device, tcp, tcpi, qtf, cbs_scale)
            r_pr8a = w_reg - pr_8a
            pr_8b = per_row_rescaled_trellis(pr_8a, r_pr8a, 2, device, tcp, tcpi, qtf, cbs_scale)
            r_pr8b = w_reg - pr_8b
            methods["MSRT_perrow_8bpw"] = (per_row_rescaled_trellis(pr_8b, r_pr8b, 3, device, tcp, tcpi, qtf, cbs_scale), 8.0)

            for name, (recon, bpw) in methods.items():
                mse = (w_reg - recon).pow(2).mean().item()
                key = f"L{layer}_{name}"
                if key not in all_methods:
                    all_methods[key] = {"mses": [], "bpw": bpw}
                all_methods[key]["mses"].append(mse)
                layer_methods.setdefault(name, {"mses": [], "bpw": bpw})["mses"].append(mse)
                del recon

            del w_reg, qk2, r2
            torch.cuda.empty_cache()

        # Print layer results
        print(f"\n  Layer {layer} results:", flush=True)
        print(f"  {'Method':<25} {'bpw':>5} {'avg MSE':>12}", flush=True)
        for name in sorted(layer_methods.keys(), key=lambda x: (layer_methods[x]["bpw"], x)):
            r = layer_methods[name]
            avg = sum(r["mses"]) / len(r["mses"])
            print(f"  {name:<25} {r['bpw']:>5.0f} {avg:>12.4e}", flush=True)

    # Cross-layer comparison
    print(f"\n  Cross-layer comparison (MSRT 8bpw):", flush=True)
    for method in ["MSRT_8bpw", "RRQ_8bpw", "MSRT_perrow_8bpw"]:
        l10_key = f"L10_{method}"
        l40_key = f"L40_{method}"
        if l10_key in all_methods and l40_key in all_methods:
            l10_mse = sum(all_methods[l10_key]["mses"]) / len(all_methods[l10_key]["mses"])
            l40_mse = sum(all_methods[l40_key]["mses"]) / len(all_methods[l40_key]["mses"])
            ratio = l40_mse / l10_mse
            print(f"  {method:<25} L10={l10_mse:.4e}  L40={l40_mse:.4e}  ratio={ratio:.4f}", flush=True)

    results["methods"] = {k: {"bpw": v["bpw"], "avg_mse": sum(v["mses"])/len(v["mses"])}
                          for k, v in all_methods.items()}
    return results

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--data-dir", default="/tmp/poc_residual/glm52_data")
    ap.add_argument("--out", default="/tmp/poc_residual/results_v44.json")
    args = ap.parse_args()
    dev = torch.device(args.device)
    print(f"Device: {dev}  GPU: {torch.cuda.get_device_name(0)}", flush=True)
    ext, ghd, tcp, tcpi, qtf, cbs_scale = _bootstrap()
    results = run_experiment(Path(args.data_dir), dev, ext, ghd, tcp, tcpi, qtf, cbs_scale)
    Path(args.out).write_text(json.dumps(results, indent=2, default=str))
    print(f"\nResults saved to {args.out}", flush=True)

if __name__ == "__main__":
    main()
