#!/usr/bin/env python3
"""PoC v51: Compare MSRT cartridge vs native K4 on real GLM-5.2 weights.

Question: Can a K3 base + K1trsc cartridge (3.422 bpw, same as willfalco)
match native K4 quality on selected experts?

Also: Can K2 base + K2trsc cartridge (4bpw) match native K4?

Measurements:
  - MSE against BF16 original (in regularized space = original space, Hadamard is orthogonal)
  - Cosine similarity against BF16 original
  - Relative error (||W - Q||_F / ||W||_F)
  - Memory estimates for each configuration

Compared approaches:
  1. brandonmusic-style: K3 only (3.0bpw, all experts)
  2. willfalco-style: K3 for 148 experts, K4 for 108 experts (3.422bpw)
  3. MSRT Option A: K2 base (all) + K2trsc cartridge (108 experts) = 2.844bpw
  4. MSRT Option B: K3 base (all) + K1trsc cartridge (108 experts) = 3.422bpw (same bpw as willfalco)
  5. MSRT Option C: K3 base (all) + K1trsc cartridge (all 256) = 4.0bpw
  6. Native K4 only (4.0bpw, all experts) — reference
"""
from __future__ import annotations
import argparse, json, math, os, sys, types, importlib.util, gc
from pathlib import Path
import torch

EXL3_PKG = "/opt/fruit-pip/exllamav3"

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
    return ext, ghd, m.tensor_core_perm, m.tensor_core_perm_i, m.quantize_tiles, m.codebook_scale

def block_rms(x, dim, keepdim=False):
    return x.square().mean(dim=dim, keepdim=keepdim).sqrt()

def regularize(w, device, ghd, cbs, had_k=128, had_n=128, seed=0):
    k, n = w.shape
    g = torch.Generator(device="cpu").manual_seed(seed)
    su = (torch.randn(k, generator=g).sign() + 1e-5).sign().float().to(device)
    sv = (torch.randn(n, generator=g).sign() + 1e-5).sign().float().to(device)
    out_scales = block_rms(w, dim=0, keepdim=True)
    mean = out_scales.mean().item()
    if mean > 1e-30: out_scales = out_scales / mean
    sv = (sv * out_scales + 1e-10).float()
    w = (w / sv).contiguous()
    had_n_mat = ghd(had_n, device, torch.float, 1.0 / math.sqrt(had_n))
    w = (w.view(k, n // had_n, had_n) @ had_n_mat).view(k, n).contiguous()
    in_scales = block_rms(w, dim=1, keepdim=True).clamp(min=1e-30)
    su = (su.unsqueeze(1) * in_scales / (-cbs) + 1e-10).float()
    w = (w / su).contiguous()
    had_k_mat = ghd(had_k, device, torch.float, 1.0 / math.sqrt(had_k))
    w = (had_k_mat @ w.view(k // had_k, had_k, n)).view(k, n).contiguous()
    return w

def quantize_trellis_raw(data, K, device, tcp, tcpi, qtf):
    k, n = data.shape; tiles_n = n // 16; weight_q = torch.zeros_like(data)
    qa = {"K": K, "mcg": True}
    perm = tcp(device); perm_i = tcpi(device)
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

def measure(w_reg, w_quant, device):
    """Measure MSE, cosine similarity, and relative error in regularized space.
    Hadamard is orthogonal so these equal the original-space metrics."""
    diff = w_reg - w_quant
    mse = diff.pow(2).mean().item()
    # Cosine similarity (flatten)
    a = w_reg.flatten()
    b = w_quant.flatten()
    cos_sim = torch.dot(a, b).item() / (a.norm().item() * b.norm().item() + 1e-30)
    # Relative Frobenius error
    rel_err = diff.norm().item() / (w_reg.norm().item() + 1e-30)
    # Max absolute error
    max_err = diff.abs().max().item()
    return {
        "mse": mse,
        "cosine": cos_sim,
        "rel_frob": rel_err,
        "max_abs_err": max_err,
    }

def run_experiment(data_dir, device, ghd, tcp, tcpi, qtf, cbs):
    results = {}
    # Use layers 10 and 40 for validation
    for layer in [10, 40]:
        f = data_dir / f"layer{layer}_all_gate_proj.pt"
        if not f.exists():
            continue
        experts = torch.load(f, map_location="cpu")
        n = min(10, experts.shape[0])

        # willfalco uses 148 K3 + 108 K4. We'll use first 6 as "K4 experts" and rest as "K3 experts"
        # (proportional: 108/256 ≈ 42%, so 4-6 of 10 experts)
        n_k4_experts = 4  # 40% ≈ willfalco's 42%
        k4_expert_ids = list(range(n_k4_experts))
        k3_expert_ids = list(range(n_k4_experts, n))

        layer_results = {}

        configs = [
            # (name, method, applies_to)
            ("K3_all", "K3", list(range(n))),
            ("K4_all", "K4", list(range(n))),
            ("K2_all", "K2", list(range(n))),
            ("willfalco_mixed", "mixed_k3k4", None),  # K3 for k3_expert_ids, K4 for k4_expert_ids
            ("MSRT_K2base_K2trsc_cart108", "msrt_k2_k2trsc", k4_expert_ids),
            ("MSRT_K3base_K1trsc_cart108", "msrt_k3_k1trsc", k4_expert_ids),
            ("MSRT_K3base_K1trsc_all", "msrt_k3_k1trsc", list(range(n))),
        ]

        print(f"\n=== Layer {layer} ({n} experts, {n_k4_experts} as 'K4/cartridge') ===", flush=True)
        print(f"{'Config':<35} {'MSE':>12} {'cosine':>8} {'rel_Frob':>10} {'max_abs':>10}", flush=True)
        print("-" * 80, flush=True)

        for name, method, applies_to in configs:
            mses = []; cosines = []; rel_forbs = []; max_errs = []

            for ei in range(n):
                w = experts[ei].to(device)
                w_reg = regularize(w, device, ghd, cbs)
                del w

                if method == "K2":
                    q = quantize_trellis_raw(w_reg, 2, device, tcp, tcpi, qtf)
                elif method == "K3":
                    q = quantize_trellis_raw(w_reg, 3, device, tcp, tcpi, qtf)
                elif method == "K4":
                    q = quantize_trellis_raw(w_reg, 4, device, tcp, tcpi, qtf)
                elif method == "mixed_k3k4":
                    if ei in k4_expert_ids:
                        q = quantize_trellis_raw(w_reg, 4, device, tcp, tcpi, qtf)
                    else:
                        q = quantize_trellis_raw(w_reg, 3, device, tcp, tcpi, qtf)
                elif method == "msrt_k2_k2trsc":
                    if applies_to and ei in applies_to:
                        # K2 base + K2trsc cartridge = 4bpw
                        q = quantize_trellis_raw(w_reg, 2, device, tcp, tcpi, qtf)
                        r = w_reg - q
                        q = rescaled_trellis(q, r, 2, device, tcp, tcpi, qtf, cbs)
                    else:
                        # K2 base only
                        q = quantize_trellis_raw(w_reg, 2, device, tcp, tcpi, qtf)
                elif method == "msrt_k3_k1trsc":
                    if applies_to and ei in applies_to:
                        # K3 base + K1trsc cartridge = 4bpw
                        q = quantize_trellis_raw(w_reg, 3, device, tcp, tcpi, qtf)
                        r = w_reg - q
                        q = rescaled_trellis(q, r, 1, device, tcp, tcpi, qtf, cbs)
                    else:
                        # K3 base only
                        q = quantize_trellis_raw(w_reg, 3, device, tcp, tcpi, qtf)
                else:
                    continue

                m = measure(w_reg, q, device)
                mses.append(m["mse"]); cosines.append(m["cosine"])
                rel_forbs.append(m["rel_frob"]); max_errs.append(m["max_abs_err"])
                del w_reg, q
                torch.cuda.empty_cache()

            avg_mse = sum(mses) / len(mses)
            avg_cos = sum(cosines) / len(cosines)
            avg_rf = sum(rel_forbs) / len(rel_forbs)
            avg_max = sum(max_errs) / len(max_errs)

            layer_results[name] = {
                "avg_mse": avg_mse, "avg_cosine": avg_cos,
                "avg_rel_frob": avg_rf, "avg_max_abs": avg_max,
                "n_experts": n,
                # Also per-expert-group breakdown
                "k4_group_mse": sum(mses[i] for i in range(n) if i in k4_expert_ids) / n_k4_experts,
                "k3_group_mse": sum(mses[i] for i in range(n) if i not in k4_expert_ids) / (n - n_k4_experts) if n > n_k4_experts else 0,
            }
            print(f"{name:<35} {avg_mse:>12.4e} {avg_cos:>8.6f} {avg_rf:>10.6f} {avg_max:>10.4f}", flush=True)

        # Per-group breakdown
        print(f"\n  Per-group breakdown (cartridge/K4 experts vs non-cartridge/K3):", flush=True)
        print(f"  {'Config':<35} {'K4-grp MSE':>12} {'K3-grp MSE':>12}", flush=True)
        print("  " + "-" * 60, flush=True)
        for name, r in layer_results.items():
            print(f"  {name:<35} {r['k4_group_mse']:>12.4e} {r['k3_group_mse']:>12.4e}", flush=True)

        results[f"layer{layer}"] = layer_results

    # Memory estimation
    print("\n=== MEMORY ESTIMATION (TP4, per rank) ===", flush=True)
    print()
    GiB_per_bpw = 64.97 / 3.0  # from brandonmusic measured
    n_experts_total = 256
    n_k4 = 108  # willfalco's K4 count
    n_k3 = 148  # willfalco's K3 count

    mem_configs = [
        ("brandonmusic 3.0bpw (all K3)", 3.0),
        ("willfalco 3.42bpw (148K3+108K4)", (n_k3*3 + n_k4*4) / n_experts_total),
        ("MSRT K2base + K2trsc cart108", (n_experts_total*2 + n_k4*2) / n_experts_total),
        ("MSRT K3base + K1trsc cart108", (n_experts_total*3 + n_k4*1) / n_experts_total),
        ("MSRT K3base + K1trsc all256", (n_experts_total*3 + n_experts_total*1) / n_experts_total),
        ("Native K4 (all)", 4.0),
    ]

    print(f"{'Config':<40} {'eff_bpw':>7} {'GiB/rank':>10} {'Base':>8} {'Cart':>8} {'Total':>8}")
    print("-" * 80)
    for name, bpw in mem_configs:
        gib = GiB_per_bpw * bpw
        # Split into base and cartridge
        if "K2base" in name:
            base_gib = GiB_per_bpw * 2.0
            cart_gib = gib - base_gib
        elif "K3base" in name and "cart" in name:
            base_gib = GiB_per_bpw * 3.0
            cart_gib = gib - base_gib
        else:
            base_gib = gib
            cart_gib = 0
        print(f"{name:<40} {bpw:>7.3f} {gib:>10.1f} {base_gib:>8.1f} {cart_gib:>8.1f} {gib:>8.1f}")

    # Quality comparison summary
    print("\n=== QUALITY SUMMARY (layer 10, cartridge experts only) ===", flush=True)
    if "layer10" in results:
        r = results["layer10"]
        k4_mse = r.get("K4_all", {}).get("k4_group_mse", 0)
        print(f"\n  Reference K4 MSE: {k4_mse:.4e}", flush=True)
        print(f"  {'Config':<40} {'K4-grp MSE':>12} {'vs K4':>8}")
        print("  " + "-" * 65)
        for name in ["K3_all", "K4_all", "willfalco_mixed",
                      "MSRT_K2base_K2trsc_cart108", "MSRT_K3base_K1trsc_cart108",
                      "MSRT_K3base_K1trsc_all"]:
            if name in r:
                mse = r[name]["k4_group_mse"]
                ratio = mse / k4_mse if k4_mse > 0 else 0
                better = "BETTER" if ratio < 1 else f"{ratio:.3f}x"
                print(f"  {name:<40} {mse:>12.4e} {better:>8}")

    return results

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--data-dir", default="/tmp/poc_residual/glm52_data")
    ap.add_argument("--out", default="/tmp/poc_residual/results_v51.json")
    args = ap.parse_args()
    dev = torch.device(args.device)
    print(f"Device: {dev}  GPU: {torch.cuda.get_device_name(0)}", flush=True)
    ext, ghd, tcp, tcpi, qtf, cbs = _bootstrap()
    print(f"codebook_scale = {cbs}", flush=True)
    results = run_experiment(Path(args.data_dir), dev, ghd, tcp, tcpi, qtf, cbs)
    Path(args.out).write_text(json.dumps(results, indent=2, default=str))
    print(f"\nResults saved to {args.out}", flush=True)

if __name__ == "__main__":
    main()
