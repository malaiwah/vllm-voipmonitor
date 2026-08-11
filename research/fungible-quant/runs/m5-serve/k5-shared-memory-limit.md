# K5 as a mixed tier does not fit SM120 shared memory — 2026-08-11

A hard hardware/kernel constraint found by booting, not by reading. It bounds
what the fungible-quant ladder can offer on Blackwell RTX PRO 6000, and it is
directly relevant to GG PR #280 (native mixed K3/K4/K5 runtime).

## What happened

The mixed K3/K5 checkpoint loaded its weights fine (77.83 GiB/rank in 81.8 s)
and then all four TP workers died during kernel construction:

```
ValueError: W4A16 shared-memory footprint exceeds device opt-in limit:
            109568 > 101376 bytes (layout=trellis3_t256)
  b12x/moe/_shared/kernels/w4a16/mixed_trellis.py  ->  make_kernel(tier1_num_experts, tier1_bits)
  b12x/moe/_shared/kernels/w4a16/kernel.py:1132    ->  W4A16GemmKernel
```

`tier1` is the K5 tier. The pure-K3 build of the same checkpoint family boots
and serves normally, so this is specific to carrying a higher tier.

## Why — measured, not inferred

`_shared_memory_footprint` from the installed b12x, against the device's
`shared_memory_per_block_optin` = **101376 bytes**:

| cta_m | tile | K3 | K4 | K5 |
|---|---|---|---|---|
| 1 | 128x128 | 49408 | 57600 | 65792 |
| 1 | 256x128 | 82176 | 98560 | **114944** |
| 2 | 128x128 | 66048 | 74240 | 82432 |
| 2 | 256x128 | 98816 | **115200** | **131584** |
| 4 | 128x128 | 99328 | **107520** | **115712** |

Bold exceeds the limit. Two things fall out:

1. **The footprint grows ~8192 bytes per bit of tier width** at a fixed tile
   (K3→K4→K5 steps of 8192 at cta_m=1/128x128). The observed failure was
   109568 for K5, so K4 at that same configuration is
   `109568 - 8192 = 101376` — **exactly the opt-in limit, to the byte.**
   K3+K4 mixed therefore fits with zero headroom, and K5 is the first tier
   that cannot fit at all.
2. **K3 alone at cta_m=4 sits at 99328 — 2% under the limit.** The tile that
   works for a uniform-K3 model has no room left for a promoted tier.

## The mechanism

`compile_mixed_trellis` is called with `force_tile_config=mixed["tile_config"]`
(`exl3.py:1895`), and `mixed_trellis.py:814`'s `make_kernel(num_experts, bits)`
instantiates **every tier with that same forced tile**, varying only
`trellis_bits`. Nothing re-checks `_candidate_tile_fits` for the widest tier.

So the tile is sized for one tier and applied to all of them. The fitting
machinery exists (`_candidate_tile_fits`, `_select_tile_config` in
`kernel.py`) but the mixed path bypasses it.

**A one-line-shaped fix upstream**: select/validate the tile against
`max(tier_bits)` rather than the base tier. From the table, dropping to
`cta_m=2, 128x128` makes K5 fit at 82432 with 19% headroom — the capability
exists, it is just not being chosen.

## Consequences for this project

- **K3+K4 is the viable mixed ladder on SM120 today.** That happens to be
  exactly what the reference `3.42bpw` Coder quant uses, and what the
  convergence demo (scenario 1) calls for, so the headline experiment is
  unaffected.
- **Scenario 2's K2→K5 ladder is blocked on this** for the K5 rung. K2/K3/K4
  are unaffected. Either the upstream tile selection is fixed, or scenario 2
  runs as a K2/K3/K4 ladder with the K5 rung documented as hardware-blocked.
- Our K5 *segments* remain valid artifacts — this is a runtime kernel limit,
  not a problem with the encoded weights. They will work on a device with a
  larger shared-memory budget, or once the tile is chosen correctly.
- **The convergence measurement does not need K4/K5 weights at all** — it
  compares which experts the routing *would* promote against the reference
  bitmap, so it runs on a plain K3 serve. That is the path taken.

## Reproduce

```bash
gg-run.sh python - <<'EOF'
import sys; sys.path.insert(0, "/home/mbelleau/src/b12x")
from b12x.moe._shared.kernels.w4a16 import kernel as K
for bits in (3, 4, 5):
    print(bits, K._shared_memory_footprint(
        cta_m_blocks=4, tile_n=128, tile_k=128, scale_format="e4m3_k16",
        weight_layout=f"trellis{bits}_t256", weight_bits=bits))
EOF
```
