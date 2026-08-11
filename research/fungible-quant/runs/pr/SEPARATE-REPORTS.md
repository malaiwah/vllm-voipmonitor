# Two findings that belong in their own issues, not in the PR

Both were found while auditing the fungible-quant work, and neither is caused
by it. Both affect EXL3 users who have never heard of fungible quant, and
report (b) blocks a feature #280 is trying to ship. Burying them in a
21,820-line PR would lose them.

**File both as issues in `local-inference-lab/vllm` before opening the PR**,
then replace the `SEPARATE-REPORTS.md` references in `PR-BODY.md` with the
resulting issue numbers. Cross-reference (b) from #280.

The bodies below are ready to paste. Line numbers were verified against
`dev/gilded-gnosis` @ `fa033bd4e1b16d9d729ad94be2d87da5a13210ce`
(2026-08-11); re-check them if the branch has moved.

Neither report proposes a patch in the PR. `exl3.py` is not our file, both
fixes want a GPU test we could not run, and both are cheaper for whoever owns
that code to make than for us to argue for.

---
---

# Report (a) — ready to file

## Title

`EXL3 retains loader tensors until process_weights_after_loading, so any borrowed-buffer load format silently corrupts the checkpoint`

## Labels

`bug`, `quantization`, `exl3`, `loader`

## Body

### Summary

The EXL3 quantization methods hold every tensor a loader yields them, from the
moment of the yield until `process_weights_after_loading` runs — the whole
load plus the whole post-processing pass. The two operations they use to
"copy" it are identity functions in exactly the conditions a device-side
loader produces. So **any load format that lends a recycled buffer instead of
giving away owned storage will build the trellis slabs out of memory that has
already been overwritten.**

This has nothing to do with any particular feature; it affects every EXL3
checkpoint. It became reachable on `dev/gilded-gnosis` when #281
(`5d2079094`) merged, because `INSTANTTENSOR_COPY=0` is now a supported mode.

There is no crash. The model loads, serves, and emits plausible-looking
garbage.

### The retention

```python
# vllm/model_executor/layers/quantization/exl3.py:792
#   Exl3Parameter.load_exl3_weight (class at :776, def at :787)
self.exl3_tensors[shard_id] = loaded_weight.contiguous()

# exl3.py:1216
#   Exl3MoEParameter.load_exl3_weight (class at :1177, def at :1207),
#   the `if not self.exl3_preallocate:` branch at :1215
self.exl3_tensors[key] = loaded_weight.contiguous()

# exl3.py:1403
#   Exl3MoEMethod.process_weights_after_loading (class at :1276, def at :1379)
param.exl3_tensors[key] = tensor.to(device=device, non_blocking=True).contiguous()
```

The retention window closes only at `exl3.py:1709`, where
`_prepare_mixed_rank_sliced_weights` (def at `:1572`) calls
`param.exl3_tensors.clear()` once the tier objects own compact copies. A ring
buffer recycled per yield is long gone by then.

### Why the two "copies" are not copies

Both are identity operations on a tensor that is already contiguous and
already on the target device:

```python
t.contiguous() is t                                # True
t.to(device=t.device, non_blocking=True) is t      # True
```

`instanttensor` yields on the CUDA device
(`device = current_platform.current_device()`), which removes the one
accidental rescue: with a host-side loader, the `.to(device=cuda)` in
`process_weights_after_loading` *is* a real copy and would have saved this
by luck.

### Which load formats are affected

| load format | owning or borrowed | verdict |
|---|---|---|
| `auto` / `hf` / `safetensors` (all strategies) | owning — the returned tensor keeps the file mapping alive by refcount | safe |
| `runai_streamer` | owning — `runai_safetensors_weights_iterator` yields `tensor.clone()` | safe |
| `instanttensor`, `INSTANTTENSOR_COPY=1` (default) | owning — "yielded tensors are clones that own their memory and outlive the context" (instanttensor 0.1.9 `_impl.py`) | safe |
| **`instanttensor`, `INSTANTTENSOR_COPY=0`** | **borrowed** — views into an internal ring buffer reused during iteration and freed on `__exit__` | **corrupts** |
| `fastsafetensors`, world size 1 | owning — `parallel_loader.py:197` sets `need_clone = pg.size() == 1`, so the library clones | safe |
| **`fastsafetensors`, world size > 1** | **borrowed** — `need_clone` is False and `_consume_single_batch` calls `batch.fb.close()` in its `finally`; `file_buffer.py:159`: *"The returned tensor must not be used after `close()` unless the caller cloned"* | **corrupts** |

The `fastsafetensors` row is the one to worry about operationally: world size
> 1 is every tensor-parallel serve.

### Mixed-bitrate checkpoints take the worst branch

`Exl3MoEMethod.create_weights` (def at `exl3.py:1283`, the relevant expression
at `:1369`):

```python
preallocate = rank_sliced and suffix in (
    {"suh", "svh"} if getattr(layer, "exl3_mixed_bitrate", False)
                   else {"suh", "svh", "trellis"})
```

- **uniform rank-sliced**: `trellis` is preallocated and genuinely copied
  (`target.copy_(loaded_weight, non_blocking=True)`), so only `mcg` / `mul1`
  retain. Small — but they are the codebook constants, so corrupting them is
  silent and total.
- **mixed bitrate**: `trellis` drops out of the preallocate set (rows differ
  in width per K), so the bulk of the weights takes the retaining branch.

### Suggested fix

#281 already labels its tensors, at
`vllm/model_executor/model_loader/weight_utils.py:1257`:

```python
if not copy_tensors:
    # Parameter loaders consume borrowed tensors synchronously.
    # Loaders retaining a tensor past this yield must materialize
    # owned storage before InstantTensor advances its ring buffer.
    tensor._vllm_instanttensor_borrowed = True
```

So the minimal fix is one helper used in the two retaining branches
(`exl3.py:792` and `exl3.py:1216`):

```python
def _own(t: torch.Tensor) -> torch.Tensor:
    """Materialize owned storage for loaders that lend their buffers."""
    if getattr(t, "_vllm_instanttensor_borrowed", False):
        return t.clone()
    return t.contiguous()
```

That covers `instanttensor` exactly. It does **not** cover `fastsafetensors`
at world size > 1, which labels nothing. Two options there:

1. always `.clone()` in the retaining branches — one extra device copy per
   non-preallocated tensor, freed as soon as the tier objects are built; or
2. extend the marker to `fastsafetensors_weights_iterator` when
   `pg.size() > 1`.

(2) is the generalisable fix. Better still, and worth proposing separately:
`_vllm_instanttensor_borrowed` is a per-library ad-hoc attribute. A neutral
`tensor._vllm_borrowed = True` — or a capability flag on `LoadConfig` — set by
*every* borrowing iterator would let consumers do the right thing once instead
of once per loader.

### Reproducing

Not executed on GPU: the verdicts above are source reading plus an identified
mechanism, plus CPU tests that model the hazard. A 5-minute GPU check would
upgrade this to demonstrated — boot any EXL3 checkpoint with
`INSTANTTENSOR_COPY=0` and expect garbage output or a codebook assertion.

What *was* executed, on CPU:

- `test_contiguous_and_same_device_to_are_not_copies` — pins the torch
  identity the whole report rests on.
- `test_retaining_consumer_is_corrupted_by_borrowed_payloads` — reproduces the
  EXL3 retain pattern against a fake resolver that serves every fragment out
  of one scratch buffer and poisons it on the next fragment; every retained
  tensor comes back poisoned. This is a control test: it proves the fake
  models the real hazard.

Both live in `tests/exl3_fungible/test_loader_compat_cpu.py` in the fungible-quant
PR (`malaiwah/vllm-voipmonitor:fq/m1-stats-collector`), alongside 10 more
loader-lifetime tests. They can be lifted into an EXL3-owned test file
unchanged — they do not import anything from the fungible-quant package.

Full matrix, with a verdict and a mechanism for every `--load-format` in the
tree including `sharded_state`, `tensorizer` and `dummy` (all three
incompatible with EXL3 for an unrelated reason — EXL3 parameters are
zero-sized and their payload never appears in `model.state_dict()`):
`research/fungible-quant/runs/m5-serve/loader-compatibility.md` in
`malaiwah/vllm-voipmonitor`.

---
---

# Report (b) — ready to file

## Title

`Mixed-Trellis tile selection ignores max(tier_bits): K5 tiers exceed SM120 shared memory and all TP workers die at kernel construction`

## Labels

`bug`, `quantization`, `exl3`, `b12x`, `blackwell`

## Body

### Summary

The mixed-Trellis path picks one tile configuration from the layer's
dimensions alone, then instantiates **every tier** with that same tile,
varying only `trellis_bits`. Nothing re-checks that the widest tier fits in
shared memory. On SM120 (RTX PRO 6000 Blackwell) that makes **K5 unusable as
a mixed tier**, and it fails at kernel construction after the weights have
already loaded.

The tile-fitting machinery exists and is bypassed on this path. A tile that
makes K5 fit with 19% headroom is available — it is simply not being chosen.

This blocks the K5 half of **#280** (`[GG] EXL3 R7 native mixed K3/K4/K5
runtime`) on SM120.

### Symptom

A mixed K3/K5 checkpoint loads its weights fine (77.83 GiB/rank in 81.8 s),
then all four TP workers die:

```
ValueError: W4A16 shared-memory footprint exceeds device opt-in limit:
            109568 > 101376 bytes (layout=trellis3_t256)
  b12x/moe/_shared/kernels/w4a16/mixed_trellis.py  ->  make_kernel(tier1_num_experts, tier1_bits)
  b12x/moe/_shared/kernels/w4a16/kernel.py:1132    ->  W4A16GemmKernel
```

`tier1` is the K5 tier. The pure-K3 build of the same checkpoint family boots
and serves normally, so this is specific to carrying a higher tier.

### Measurement

`_shared_memory_footprint` (`b12x/.../w4a16/kernel.py:326`) evaluated against
this device's `shared_memory_per_block_optin` = **101376 bytes**:

| cta_m | tile | K3 | K4 | K5 |
|---|---|---|---|---|
| 1 | 128x128 | 49408 | 57600 | 65792 |
| 1 | 256x128 | 82176 | 98560 | **114944** |
| 2 | 128x128 | 66048 | 74240 | 82432 |
| 2 | 256x128 | 98816 | **115200** | **131584** |
| 4 | 128x128 | 99328 | **107520** | **115712** |

Bold exceeds the limit. Two things fall out:

1. The footprint grows **~8192 bytes per bit of tier width** at a fixed tile.
   The observed failure was 109568 for K5, so K4 at that same configuration is
   `109568 − 8192 = 101376` — **exactly the opt-in limit, to the byte.** K3+K4
   mixed fits with zero headroom; K5 is the first tier that cannot fit at all.
2. K3 alone at `cta_m=4` sits at 99328, 2% under the limit. **The tile that
   works for a uniform-K3 model has no room left for a promoted tier.**

Reproduce:

```bash
python - <<'EOF'
from b12x.moe._shared.kernels.w4a16 import kernel as K
for bits in (3, 4, 5):
    print(bits, K._shared_memory_footprint(
        cta_m_blocks=4, tile_n=128, tile_k=128, scale_format="e4m3_k16",
        weight_layout=f"trellis{bits}_t256", weight_bits=bits))
EOF
```

### Mechanism

The tile is chosen from geometry only, with no knowledge of any tier's bit
width — the selector does not even take a bits argument:

```python
# vllm/model_executor/layers/quantization/exl3.py:1557
@staticmethod
def _mixed_trellis_tile_config(hidden_size: int, intermediate_size: int):
    ...
    if hidden_size % 512 == 0:
        return (128, 128, 32, 512)
    ...
```

It is called once at `exl3.py:1612`, stored into the layer's mixed state at
`exl3.py:1699`, and then **forced** into the compiler:

```python
# exl3.py:1896, inside _mixed_rank_sliced_runtime (def at :1828)
launch = api.compile_mixed_trellis(
    ...
    max_shared_mem=int(props.shared_memory_per_block_optin),
    force_tile_config=mixed["tile_config"],
    ...
)
```

On the b12x side that forced tile is applied to every tier
(`b12x/moe/_shared/kernels/w4a16/mixed_trellis.py`):

```python
814:  def make_kernel(num_experts: int, bits: int) -> W4A16FusedMoeKernel:
845:      driver=make_kernel(total_experts, tier0_bits),
846:      tier0=make_kernel(int(tier0_num_experts), int(tier0_bits)),
847:      tier1=make_kernel(int(tier1_num_experts), int(tier1_bits)),
```

`make_kernel` closes over the forced `fc1_tile_n` / `fc1_tile_k` and varies
only `trellis_bits`. `max_shared_mem` is passed in and is what the eventual
`ValueError` at `kernel.py:1132` is checked against — but by then the tile is
already fixed, so the check can only reject, never adapt.

Meanwhile the fitting machinery exists and is used elsewhere:
`_candidate_tile_fits` (`kernel.py:411`) and `_select_tile_config`
(`kernel.py:457`, which loops over candidates calling `_candidate_tile_fits`
at `:484`). The mixed path never reaches it.

### Suggested fix

Select and validate the tile against **`max(tier_bits)`**, not against
geometry alone and not against the base tier.

Two shapes, in increasing order of intrusiveness:

1. **Validate at selection time.** Give `_mixed_trellis_tile_config` the tier
   bit widths and have it reject a candidate whose widest tier does not fit,
   falling through to the next candidate. From the table, dropping to
   `cta_m=2, 128x128` makes K5 fit at 82432 — **19% headroom.** The capability
   is there; it just is not being chosen.
2. **Stop forcing.** Let `compile_mixed_trellis` run `_select_tile_config`
   with `weight_bits = max(tier_bits)` when the forced tile does not fit, and
   log the downgrade. This also fixes the case where a *future* tier is
   promoted into a layer whose tile was chosen when the layer was uniform.

Either way the error message at `kernel.py:1132` would be more useful if it
named the tier and the tile it was constructed with — the current text gives
the layout string but not which of the three `make_kernel` calls raised.

### Why this matters beyond one device

- It is a **late** failure. The weights load first (77.83 GiB/rank, 81.8 s
  wasted) and then every TP worker dies.
- It is **silent about the workaround.** A tile that fits exists, and the
  operator has no way to ask for it.
- The K3+K4 case fits with **zero bytes to spare** on SM120 at that
  configuration, so any future increase in per-tier footprint breaks the one
  mixed ladder that currently works.
- It bounds what #280 can claim on consumer/workstation Blackwell: the PR
  title says K3/K4/K5, and on SM120 the K5 rung will not construct.

### Scope note

The encoded K5 weights are fine — this is a runtime kernel limit, not a
problem with the checkpoint. K5 segments work on a device with a larger
shared-memory budget, or once the tile is chosen correctly.

Full write-up, including the derivation of the 8192-bytes-per-bit slope:
`research/fungible-quant/runs/m5-serve/k5-shared-memory-limit.md` in
`malaiwah/vllm-voipmonitor`.
