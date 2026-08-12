#!/usr/bin/env python3
"""PoC v23b: Per-row-group Lloyd-Max (fast) + vectorized per-tile.

v23 timed out due to Python for-loop over 49152 tiles. v23b vectorizes
per-tile LM using batched GPU operations (reshape → (n_tiles, 256),
batched distance computation).
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

def lloyd_max_global(r, n_bits, n_iters=20):
    n_levels = 2 ** n_bits
    sigma = r.std().item()
    if sigma < 1e-12: return torch.zeros_like(r)
    levels = torch.linspace(-3 * sigma, 3 * sigma, n_levels, device=r.device)
    flat = r.flatten()
    n_elem = flat.numel()
    chunk = max(1, min(n_elem, (1024 * 1024 * 1024) // (n_levels * 4)))
    for _ in range(n_iters):
        assign = torch.empty(n_elem, dtype=torch.long, device=r.device)
        for s in range(0, n_elem, chunk):
            e = min(s + chunk, n_elem)
            d = (flat[s:e].unsqueeze(1) - levels.unsqueeze(0)).abs()
            assign[s:e] = d.argmin(dim=1)
        new_levels = levels.clone()
        for i in range(n_levels):
            mask = assign == i
            if mask.sum() > 0: new_levels[i] = flat[mask].mean()
        if (new_levels - levels).abs().max() < 1e-10 * sigma: break
        levels = new_levels
    result = torch.empty_like(flat)
    for s in range(0, n_elem, chunk):
        e = min(s + chunk, n_elem)
        d = (flat[s:e].unsqueeze(1) - levels.unsqueeze(0)).abs()
        result[s:e] = levels[d.argmin(dim=1)]
    return result.reshape(r.shape)

def lloyd_max_per_row_group(r, n_bits, k, n, group_size=128, n_iters=15):
    n_levels = 2 ** n_bits
    result = torch.zeros_like(r)
    for rs in range(0, k, group_size):
        re = min(rs + group_size, k)
        chunk = r[rs:re]
        sigma = chunk.std().item()
        if sigma < 1e-12: continue
        levels = torch.linspace(-3 * sigma, 3 * sigma, n_levels, device=r.device)
        flat = chunk.flatten()
        for _ in range(n_iters):
            d = (flat.unsqueeze(1) - levels.unsqueeze(0)).abs()
            assign = d.argmin(dim=1)
            new_levels = levels.clone()
            for i in range(n_levels):
                mask = assign == i
                if mask.sum() > 0: new_levels[i] = flat[mask].mean()
            if (new_levels - levels).abs().max() < 1e-10 * sigma: break
            levels = new_levels
        d = (flat.unsqueeze(1) - levels.unsqueeze(0)).abs()
        result[rs:re] = levels[d.argmin(dim=1)].reshape(chunk.shape)
    return result

def lloyd_max_per_tile_batched(r, n_bits, k, n, n_iters=8):
    """Vectorized per-tile Lloyd-Max using batched GPU ops."""
    n_levels = 2 ** n_bits
    tnk, tnn = k // 16, n // 16
    n_tiles = tnk * tnn

    # Reshape into tiles: (n_tiles, 256)
    tiles = r.view(tnk, 16, tnn, 16).permute(0, 2, 1, 3).reshape(n_tiles, 256)

    # Per-tile sigma
    sigma = tiles.std(dim=1, keepdim=True).clamp(min=1e-12)  # (n_tiles, 1)

    # Initialize levels: (n_tiles, n_levels)
    base = torch.linspace(-3, 3, n_levels, device=r.device)
    levels = base.unsqueeze(0) * sigma  # (n_tiles, n_levels)

    for _ in range(n_iters):
        # Process in chunks of tiles to limit memory
        # (chunk, 256, n_levels) at 4 bytes = chunk * 256 * n_levels * 4
        # Limit to ~1GB
        chunk_sz = max(1, min(n_tiles, (256 * 1024 * 1024) // (256 * n_levels)))
        assign = torch.empty(n_tiles, 256, dtype=torch.long, device=r.device)
        for s in range(0, n_tiles, chunk_sz):
            e = min(s + chunk_sz, n_tiles)
            d = (tiles[s:e].unsqueeze(2) - levels[s:e].unsqueeze(1)).abs()  # (chunk, 256, n_levels)
            assign[s:e] = d.argmin(dim=2)
            del d

        # Update levels: for each level i, mean of tiles assigned to it
        new_levels = levels.clone()
        for i in range(n_levels):
            mask = (assign == i).float()  # (n_tiles, 256)
            count = mask.sum(dim=1)  # (n_tiles,)
            total = (tiles * mask).sum(dim=1)  # (n_tiles,)
            valid = count > 0
            new_levels[valid, i] = total[valid] / count[valid]

        diff = (new_levels - levels).abs().max().item()
        levels = new_levels
        if diff < 1e-10:
            break

    # Final assignment
    chunk_sz = max(1, min(n_tiles, (256 * 1024 * 1024) // (256 * n_levels)))
    result_tiles = torch.empty(n_tiles, 256, device=r.device)
    for s in range(0, n_tiles, chunk_sz):
        e = min(s + chunk_sz, n_tiles)
        d = (tiles[s:e].unsqueeze(2) - levels[s:e].unsqueeze(1)).abs()
        idx = d.argmin(dim=2)
        result_tiles[s:e] = levels[s:e].gather(1, idx)
        del d

    return result_tiles.reshape(tnk, tnn, 16, 16).permute(0, 2, 1, 3).reshape(k, n)

def run_experiment(data_dir, device, ext, ghd, tcp, tcpi, qtf, cbs):
    results = {}
    layer_idx = 10
    gate_file = data_dir / f"layer{layer_idx}_all_gate_proj.pt"
    if not gate_file.exists(): return results

    all_experts = torch.load(gate_file, map_location="cpu")
    n_experts = min(2, all_experts.shape[0])
    k, n = all_experts.shape[1], all_experts.shape[2]
    print(f"  {n_experts} experts, {k}x{n}", flush=True)

    all_methods = {}

    for ei in range(n_experts):
        print(f"  Expert {ei}...", flush=True)
        w = all_experts[ei].to(device)
        w_reg, _, _ = regularize(w, device, ghd, cbs)
        del w

        qk3 = quantize_trellis(w_reg, 3, device, tcp, tcpi, qtf)
        qk4 = quantize_trellis(w_reg, 4, device, tcp, tcpi, qtf)
        r3 = w_reg - qk3
        r4 = w_reg - qk4

        methods = {}

        # Global LM baseline
        for nbits in [2, 4]:
            lm = lloyd_max_global(r4, nbits)
            methods[f"K4+{nbits}LM_global"] = (qk4 + lm, 4 + nbits)
            del lm

        # Per-row-group LM (fast, 16 groups)
        for nbits in [2, 4]:
            n_groups = k // 128
            overhead = n_groups * (2 ** nbits) * 4 / (k * n)
            t0 = time.time()
            lm = lloyd_max_per_row_group(r4, nbits, k, n, group_size=128)
            t1 = time.time()
            methods[f"K4+{nbits}LM_row128(oh={overhead:.5f})"] = (qk4 + lm, 4 + nbits + overhead)
            print(f"    row128 {nbits}bit: {t1-t0:.1f}s", flush=True)
            del lm

        # Per-tile LM (vectorized batched)
        for nbits in [2, 4]:
            overhead = (2 ** nbits) * 4 / 256
            t0 = time.time()
            lm = lloyd_max_per_tile_batched(r4, nbits, k, n)
            t1 = time.time()
            methods[f"K4+{nbits}LM_tile(oh={overhead:.3f})"] = (qk4 + lm, 4 + nbits + overhead)
            print(f"    tile {nbits}bit: {t1-t0:.1f}s", flush=True)
            del lm

        # K3 + per-tile LM
        for nbits in [2, 4]:
            overhead = (2 ** nbits) * 4 / 256
            t0 = time.time()
            lm = lloyd_max_per_tile_batched(r3, nbits, k, n)
            t1 = time.time()
            methods[f"K3+{nbits}LM_tile(oh={overhead:.3f})"] = (qk3 + lm, 3 + nbits + overhead)
            print(f"    K3 tile {nbits}bit: {t1-t0:.1f}s", flush=True)
            del lm

        for name, (recon, bpw) in methods.items():
            mse = (w_reg - recon).pow(2).mean().item()
            if name not in all_methods:
                all_methods[name] = {"mses": [], "bpw": bpw}
            all_methods[name]["mses"].append(mse)
            del recon

        del w_reg, qk3, qk4, r3, r4
        torch.cuda.empty_cache()

    # Print
    print(f"\n  {'Method':<40} {'bpw':>7} {'MSE':>12}", flush=True)
    print(f"  {'-'*62}", flush=True)
    for name in sorted(all_methods.keys(), key=lambda x: all_methods[x]["bpw"]):
        r = all_methods[name]
        avg_mse = sum(r["mses"]) / len(r["mses"])
        print(f"  {name:<40} {r['bpw']:>7.3f} {avg_mse:>12.4e}", flush=True)

    results["methods"] = {k: {"bpw": v["bpw"], "avg_mse": sum(v["mses"])/len(v["mses"])}
                          for k, v in all_methods.items()}
    return results

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--data-dir", default="/tmp/poc_residual/glm52_data")
    ap.add_argument("--out", default="/tmp/poc_residual/results_v23b.json")
    args = ap.parse_args()
    dev = torch.device(args.device)
    print(f"Device: {dev}  GPU: {torch.cuda.get_device_name(0)}", flush=True)
    ext, ghd, tcp, tcpi, qtf, cbs = _bootstrap()
    results = run_experiment(Path(args.data_dir), dev, ext, ghd, tcp, tcpi, qtf, cbs)
    Path(args.out).write_text(json.dumps(results, indent=2, default=str))
    print(f"\nResults saved to {args.out}", flush=True)

if __name__ == "__main__":
    main()
