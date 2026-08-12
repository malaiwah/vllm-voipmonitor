#!/usr/bin/env python3
"""PoC v52: Dual-cartridge MSRT — K2 base + tiered cartridges.

Test the user's proposal: K2 base (all experts) with a DUAL cartridge system:
  - Cartridge A (K1trsc): applied to MOST experts → brings them to K3-equivalent
  - Cartridge B (K2trsc): applied to SELECTED experts → brings them to K4-equivalent
  - Optional Cartridge C (K3trsc): applied to FEW experts → brings them to K5-equivalent

This creates a 3-tier effective quantization from a single K2 base:
  Tier 0: K2 only (2.0 bpw)     — cold/unused experts
  Tier 1: K2+K1trsc (3.0 bpw)   — standard experts (K3-equivalent)
  Tier 2: K2+K2trsc (4.0 bpw)   — important experts (K4-equivalent)
  Tier 3: K2+K3trsc (5.0 bpw)   — hot experts (K5-equivalent)

Compare against:
  - brandonmusic 3.0bpw (all K3)
  - willfalco 3.42bpw (148 K3 + 108 K4)
  - MSRT single-cartridge variants from v51

The key advantage: ALL tiers share the same K2 base. Swapping cartridges
changes the effective quantization without touching the base. The base is
always K2 (43.3 GiB/rank), cartridges are additive and independently
loadable via LoRA hot-swap.
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
    diff = w_reg - w_quant
    mse = diff.pow(2).mean().item()
    a = w_reg.flatten(); b = w_quant.flatten()
    cos_sim = torch.dot(a, b).item() / (a.norm().item() * b.norm().item() + 1e-30)
    rel_err = diff.norm().item() / (w_reg.norm().item() + 1e-30)
    max_err = diff.abs().max().item()
    return {"mse": mse, "cosine": cos_sim, "rel_frob": rel_err, "max_abs_err": max_err}

def run_experiment(data_dir, device, ghd, tcp, tcpi, qtf, cbs):
    results = {}

    for layer in [10, 40]:
        f = data_dir / f"layer{layer}_all_gate_proj.pt"
        if not f.exists(): continue
        experts = torch.load(f, map_location="cpu")
        n = min(10, experts.shape[0])

        # Tier assignment (proportional to willfalco 148/108 split, but with 3 tiers):
        # 10 experts: 4 cold (K2 only), 4 standard (K2+K1trsc), 2 hot (K2+K1trsc+K2trsc)
        # This gives: 4*2 + 4*3 + 2*4 = 28 bits / 10 = 2.8 bpw
        # For willfalco comparison: 6 standard (K2+K1trsc=3bpw) + 4 hot (K2+K1trsc+K2trsc=4bpw)
        # = 6*3 + 4*4 = 34 / 10 = 3.4 bpw ≈ willfalco's 3.42

        n_hot = 2     # K2+K1trsc+K2trsc = 4bpw (K4-equivalent)
        n_std = 4     # K2+K1trsc = 3bpw (K3-equivalent)
        n_cold = n - n_hot - n_std  # K2 only = 2bpw

        hot_ids = list(range(n_hot))
        std_ids = list(range(n_hot, n_hot + n_std))
        cold_ids = list(range(n_hot + n_std, n))
        # Also a willfalco-matching split: all get K1trsc, hot get +K2trsc
        std_all_ids = list(range(n - n_hot))  # all non-hot get K1trsc
        hot_all_ids = list(range(n - n_hot, n))

        # Also a 3-tier with K3trsc for hot
        n_hot3 = 2
        n_std3 = 4
        n_cold3 = n - n_hot3 - n_std3

        layer_results = {}

        configs = [
            # References
            ("K3_all", "single", {"k": 3}, list(range(n))),
            ("K4_all", "single", {"k": 4}, list(range(n))),
            ("K2_all", "single", {"k": 2}, list(range(n))),

            # willfalco-style: K3 for most, K4 for selected
            ("willfalco_K3K4", "single_mixed", {"k3_ids": std_all_ids, "k4_ids": hot_all_ids}, None),

            # Single-cartridge references (from v51)
            ("MSRT_K3base_K1trsc_4hot", "msrt", {"base_k": 3, "stages": [1]}, hot_all_ids),

            # === DUAL CARTRIDGE CONFIGS ===

            # Dual-A: K2 base + K1trsc (std) + K2trsc (hot) = 3-tier (2/3/4 bpw)
            # 4 cold at K2, 4 std at K2+K1trsc, 2 hot at K2+K1trsc+K2trsc
            ("DualK2_K1std_K2hot", "msrt_tiered", {
                "base_k": 2,
                "tiers": [
                    {"ids": cold_ids, "stages": []},           # K2 only (2bpw)
                    {"ids": std_ids, "stages": [1]},           # K2+K1trsc (3bpw)
                    {"ids": hot_ids, "stages": [1, 2]},        # K2+K1trsc+K2trsc (4bpw)
                ]
            }, None),

            # Dual-B: K2 base + K1trsc (ALL non-hot) + K2trsc (hot) = matches willfalco bpw
            # 8 std at K2+K1trsc (3bpw), 2 hot at K2+K1trsc+K2trsc (4bpw)
            ("DualK2_K1all_K2hot", "msrt_tiered", {
                "base_k": 2,
                "tiers": [
                    {"ids": std_all_ids, "stages": [1]},       # K2+K1trsc (3bpw)
                    {"ids": hot_all_ids, "stages": [1, 2]},    # K2+K1trsc+K2trsc (4bpw)
                ]
            }, None),

            # Dual-C: K2 base + K1trsc (ALL) + K2trsc (hot) + K3trsc (very hot)
            # 3-tier with K5-equivalent on hottest
            ("DualK2_K1all_K2std_K3hot", "msrt_tiered", {
                "base_k": 2,
                "tiers": [
                    {"ids": cold_ids, "stages": [1]},           # K2+K1trsc (3bpw)
                    {"ids": std_ids, "stages": [1, 2]},         # K2+K1trsc+K2trsc (4bpw)
                    {"ids": hot_ids, "stages": [1, 2, 3]},      # K2+K1trsc+K2trsc+K3trsc (5bpw)
                ]
            }, None),

            # Dual-D: Same as Dual-B but with K3trsc instead of K2trsc for hot
            # 8 at K2+K1trsc (3bpw), 2 at K2+K1trsc+K3trsc (5bpw)
            ("DualK2_K1all_K3hot", "msrt_tiered", {
                "base_k": 2,
                "tiers": [
                    {"ids": std_all_ids, "stages": [1]},       # K2+K1trsc (3bpw)
                    {"ids": hot_all_ids, "stages": [1, 3]},    # K2+K1trsc+K3trsc (5bpw)
                ]
            }, None),
        ]

        print(f"\n=== Layer {layer} ({n} experts: {n_hot} hot, {n_std} std, {n_cold} cold) ===", flush=True)
        print(f"{'Config':<35} {'avg MSE':>12} {'cosine':>8} {'rel_Frob':>10} {'eff_bpw':>8}", flush=True)
        print("-" * 80, flush=True)

        for name, method, params, applies_to in configs:
            mses = []; cosines = []; rel_forbs = []
            total_bits = 0

            for ei in range(n):
                w = experts[ei].to(device)
                w_reg = regularize(w, device, ghd, cbs)
                del w

                if method == "single":
                    k = params["k"]
                    q = quantize_trellis_raw(w_reg, k, device, tcp, tcpi, qtf)
                    total_bits += k

                elif method == "single_mixed":
                    if ei in params.get("k4_ids", []):
                        q = quantize_trellis_raw(w_reg, 4, device, tcp, tcpi, qtf)
                        total_bits += 4
                    else:
                        q = quantize_trellis_raw(w_reg, 3, device, tcp, tcpi, qtf)
                        total_bits += 3

                elif method == "msrt":
                    base_k = params["base_k"]
                    stages = params["stages"]
                    if applies_to and ei in applies_to:
                        q = quantize_trellis_raw(w_reg, base_k, device, tcp, tcpi, qtf)
                        for sk in stages:
                            r = w_reg - q
                            q = rescaled_trellis(q, r, sk, device, tcp, tcpi, qtf, cbs)
                        total_bits += base_k + sum(stages)
                    else:
                        q = quantize_trellis_raw(w_reg, base_k, device, tcp, tcpi, qtf)
                        total_bits += base_k

                elif method == "msrt_tiered":
                    base_k = params["base_k"]
                    tiers = params["tiers"]
                    # Find which tier this expert belongs to
                    expert_stages = []
                    for tier in tiers:
                        if ei in tier["ids"]:
                            expert_stages = tier["stages"]
                            break

                    q = quantize_trellis_raw(w_reg, base_k, device, tcp, tcpi, qtf)
                    total_bits += base_k
                    for sk in expert_stages:
                        r = w_reg - q
                        q = rescaled_trellis(q, r, sk, device, tcp, tcpi, qtf, cbs)
                        total_bits += sk

                m = measure(w_reg, q, device)
                mses.append(m["mse"]); cosines.append(m["cosine"]); rel_forbs.append(m["rel_frob"])
                del w_reg, q
                torch.cuda.empty_cache()

            avg_mse = sum(mses) / len(mses)
            avg_cos = sum(cosines) / len(cosines)
            avg_rf = sum(rel_forbs) / len(rel_forbs)
            eff_bpw = total_bits / n

            layer_results[name] = {
                "avg_mse": avg_mse, "avg_cosine": avg_cos,
                "avg_rel_frob": avg_rf, "eff_bpw": eff_bpw,
                "n_experts": n,
                "per_expert_mses": mses,
            }
            print(f"{name:<35} {avg_mse:>12.4e} {avg_cos:>8.6f} {avg_rf:>10.6f} {eff_bpw:>8.3f}", flush=True)

        # Per-tier breakdown for dual-cartridge configs
        print(f"\n  Per-tier breakdown:", flush=True)
        print(f"  {'Config':<35} {'hot MSE':>12} {'std MSE':>12} {'cold MSE':>12}", flush=True)
        print("  " + "-" * 75, flush=True)
        for name, r in layer_results.items():
            per_expert = r["per_expert_mses"]
            if len(per_expert) >= n:
                hot_mse = sum(per_expert[i] for i in hot_ids) / len(hot_ids) if hot_ids else 0
                std_mse = sum(per_expert[i] for i in std_ids) / len(std_ids) if std_ids else 0
                cold_mse = sum(per_expert[i] for i in cold_ids) / len(cold_ids) if cold_ids else 0
                print(f"  {name:<35} {hot_mse:>12.4e} {std_mse:>12.4e} {cold_mse:>12.4e}", flush=True)

        results[f"layer{layer}"] = layer_results

    # Memory estimation
    print("\n=== MEMORY ESTIMATION (TP4, per rank, 256 experts) ===", flush=True)
    GiB_per_bpw = 64.97 / 3.0
    N = 256

    # willfalco: 148 K3 + 108 K4 = 3.422 bpw, 74.1 GiB/rank
    # DualK2_K1all_K2hot: 148 K2+K1trsc (3bpw) + 108 K2+K1trsc+K2trsc (4bpw)
    #   = (148*3 + 108*4) / 256 = 3.422 bpw → SAME as willfalco!
    # But split: base = 256*2 = 512 bits, cart_A(K1) = 256*1 = 256 bits, cart_B(K2) = 108*2 = 216 bits
    # Total = 512 + 256 + 216 = 984 bits / 256 = 3.844 bpw — NO, that's wrong
    # Correct: K2 base (all 256) = 2 bpw, K1trsc (all 256) = 1 bpw, K2trsc (108) = 2*108/256 = 0.844 bpw
    # Total = 2 + 1 + 0.844 = 3.844 bpw — that's because ALL get K1trsc, not just 148

    # For willfalco-matching: 148 get K2+K1trsc (3bpw), 108 get K2+K1trsc+K2trsc (4bpw)
    # = (148*3 + 108*4) / 256 = (444 + 432) / 256 = 3.422 bpw
    # Base: 256*2 = 2.0 bpw
    # Cart A (K1, 256 experts): 1.0 bpw
    # Cart B (K2, 108 experts): 0.844 bpw
    # Total: 3.844 bpw — but that counts K1 on ALL 256, including the 108 that also get K2

    # Actually: the base is K2 for ALL. Then:
    # 148 experts get +K1trsc = 3bpw
    # 108 experts get +K1trsc +K2trsc = 4bpw
    # Base: 256 * 2 = 512 bits → 2.0 bpw
    # Cart A: 256 * 1 = 256 bits → 1.0 bpw (ALL get K1)
    # Cart B: 108 * 2 = 216 bits → 0.844 bpw (only hot get K2)
    # Total: (512 + 256 + 216) / 256 = 984/256 = 3.844 bpw
    # Wait, that's 3.844 not 3.422. The issue: ALL 256 get K1 (not just 148).
    # If only 148 get K1: (512 + 148 + 216) / 256 = 876/256 = 3.422 bpw ✓

    # So the configs are:
    mem_configs = [
        # (name, base_bpw, cart_A_bpw, cart_A_experts, cart_B_bpw, cart_B_experts, cart_C_bpw, cart_C_experts)
        ("brandonmusic K3 all", 3.0, 0, 0, 0, 0, 0, 0),
        ("willfalco 148K3+108K4", 0, 0, 0, 0, 0, 0, 0),  # special: not MSRT
        ("DualK2: 148×(K2+K1), 108×(K2+K1+K2)", 2.0, 1.0, 256, 2.0, 108, 0, 0),  # all get K1, 108 get K2
        ("DualK2: 148×(K2+K1), 108×(K2+K1+K2) v2", 2.0, 1.0, 148, 2.0, 108, 0, 0),  # only 148 get K1
        ("DualK2: 4×K2, 4×(K2+K1), 2×(K2+K1+K2)", 2.0, 1.0, int(N*0.4), 2.0, int(N*0.2), 0, 0),
        ("DualK2: 4×(K2+K1), 4×(K2+K1+K2), 2×(K2+K1+K2+K3)", 2.0, 1.0, N, 2.0, int(N*0.4), 3.0, int(N*0.2)),
        ("DualK2: 8×(K2+K1), 2×(K2+K1+K3)", 2.0, 1.0, 256, 3.0, int(N*0.2), 0, 0),
    ]

    print(f"\n{'Config':<55} {'eff_bpw':>7} {'GiB/rk':>8} {'Base':>6} {'CartA':>6} {'CartB':>6} {'CartC':>6}")
    print("-" * 95)

    for name, base_bpw, cA_bpw, cA_n, cB_bpw, cB_n, cC_bpw, cC_n in mem_configs:
        if "willfalco" in name:
            eff = (148*3 + 108*4) / N
            gib = GiB_per_bpw * eff
            print(f"{name:<55} {eff:>7.3f} {gib:>8.1f} {gib:>6.1f} {'—':>6} {'—':>6} {'—':>6}")
            continue

        base_bits = N * base_bpw
        cA_bits = cA_n * cA_bpw
        cB_bits = cB_n * cB_bpw
        cC_bits = cC_n * cC_bpw
        total_bits = base_bits + cA_bits + cB_bits + cC_bits
        eff_bpw = total_bits / N
        gib = GiB_per_bpw * eff_bpw
        base_gib = GiB_per_bpw * base_bpw
        cA_gib = GiB_per_bpw * (cA_bits / N)
        cB_gib = GiB_per_bpw * (cB_bits / N)
        cC_gib = GiB_per_bpw * (cC_bits / N) if cC_bits > 0 else 0
        print(f"{name:<55} {eff_bpw:>7.3f} {gib:>8.1f} {base_gib:>6.1f} {cA_gib:>6.1f} {cB_gib:>6.1f} {cC_gib:>6.1f}")

    # Quality comparison summary
    print("\n=== QUALITY SUMMARY (layer 10) ===", flush=True)
    if "layer10" in results:
        r = results["layer10"]
        k4_ref = r.get("K4_all", {}).get("avg_mse", 0)
        k3_ref = r.get("K3_all", {}).get("avg_mse", 0)
        print(f"\n  Reference: K3 MSE={k3_ref:.4e}, K4 MSE={k4_ref:.4e}", flush=True)
        print(f"  {'Config':<40} {'MSE':>12} {'bpw':>6} {'vs K3':>8} {'vs K4':>8}")
        print("  " + "-" * 78)
        for name in sorted(r.keys(), key=lambda x: r[x]["eff_bpw"]):
            d = r[name]
            vs_k3 = f"{d['avg_mse']/k3_ref:.4f}x" if k3_ref > 0 else "~"
            vs_k4 = f"{d['avg_mse']/k4_ref:.4f}x" if k4_ref > 0 else "~"
            print(f"  {name:<40} {d['avg_mse']:>12.4e} {d['eff_bpw']:>6.3f} {vs_k3:>8} {vs_k4:>8}")

    return results

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--data-dir", default="/tmp/poc_residual/glm52_data")
    ap.add_argument("--out", default="/tmp/poc_residual/results_v52.json")
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
