# Tensor-loader compatibility of the fungible-quant integration

Date: 2026-08-11. Author: loader-compat sweep (CPU-only; GPUs were in use by a
live serve and the 0c encode campaign, so nothing here was booted).

**Question asked.** Are our fungible-quant changes compatible with *every*
tensor-loader variant vLLM/GG supports — not just the default safetensors
path? Specifically the "instant tensors loader": GG PR #281
(`126039af3`, branch `origin/codex/gg-instanttensor-zero-copy-contract-20260810`)
adds an InstantTensor borrowed-buffer mode where `INSTANTTENSOR_COPY=0`
forwards `copy=False` to `instanttensor.safe_open`.

**Short answer.** Our own code is clean in both directions: the progressive
loader and `fragments.py` are *owning producers* (every tensor they hand out
keeps its own backing alive by refcount), and the only FQ code that consumes
tensors it did not produce — `swap.ResolverFragmentSource.read_expert` —
copies before it returns. Both are now pinned by CPU tests, including one that
drives the real swap engine through a payload buffer that gets recycled the
instant the call returns.

**But** the loader question does not resolve in our favour overall, and the
reason is not ours: **the EXL3 quant methods are an owning-only consumer.**
They retain every loaded tensor from the moment it is yielded until
`process_weights_after_loading`, and the two operations they use to "copy" it
(`.contiguous()` and `.to(device)`) are both identity functions when the tensor
is already contiguous and already on the target device. So *any* borrowed-buffer
loader silently corrupts an EXL3 checkpoint — with or without fungible quant.
That rules out `instanttensor` with `INSTANTTENSOR_COPY=0`, and rules out
`fastsafetensors` whenever the world size is > 1 (which is every TP serve we
run). This is a GG/exl3.py issue that FQ inherits; §5 has the concrete fix.

---

## 1. Compatibility matrix

Scope of each verdict: "for an EXL3 rank-sliced, mixed-bitrate MoE checkpoint
with `VLLM_FQ_ENABLE=1`". "Demonstrated" means something was actually executed;
"inferred" means read from source with the mechanism identified but not run.

| `--load-format` | Verdict | Why | What would have to change |
|---|---|---|---|
| `auto` | **works** | Resolves to `hf` (no `consolidated*.safetensors` in our checkpoints); default lazy safetensors. Demonstrated on GPU by `m5-serve` (assembled K3/mixed serves). | — |
| `hf` | **works** | Same path as `auto`. | — |
| `safetensors` | **works** | Lazy `safe_open` + `get_tensor`; the returned tensor keeps the file mapping alive by refcount — verified empirically (tensor still readable after the `with` block exits and a `gc.collect()`). This is *why* EXL3's retain-then-copy pattern has always worked. | — |
| `safetensors` + `--safetensors-load-strategy eager` | **works (inferred)** | `load(f.read())` produces owned CPU tensors. | — |
| `safetensors` + `--safetensors-load-strategy prefetch` | **works (inferred)** | Prefetch only warms page cache; the yield path is unchanged. | — |
| `safetensors` + `enable_multithread_load` | **works (inferred)** | `load_file(...)` returns owned tensors per shard. Rejected in combination with a non-lazy strategy by `DefaultModelLoader.__init__`. | — |
| **`progressive`** (ours) | **works** | Our loader. Demonstrated on GPU TP4 by `runs/loader-v2/report.md` (PASS, token-identical to the assembled serve). Owning producer: proven again here by CPU test — tensors stay correct after the generator, resolver and spec are dropped and GC runs. Caveats in §4. | — |
| `runai_streamer` | **works (inferred)** | `runai_safetensors_weights_iterator` yields `tensor.clone()` (weight_utils.py:1124) — owned by construction, since upstream #43464. Requires an object-storage or local dir; untested against EXL3 here. | — |
| **`instanttensor`, `INSTANTTENSOR_COPY=1`** (default) | **works (inferred)** | `copy=True` — "yielded tensors are clones that own their memory and outlive the context" (instanttensor 0.1.9 `_impl.py` docstring). Owning. Note PR #281 is **not** merged into `fq/m1-stats-collector`; on our branch `safe_open` is called without `copy=`, which defaults to `True`. | — |
| **`instanttensor`, `INSTANTTENSOR_COPY=0`** | **incompatible** | Borrowed. instanttensor 0.1.9: views "into an internal ring buffer **reused during iteration** and freed on `__exit__`". EXL3 retains every tensor until `process_weights_after_loading` (§2) → not a race, a certainty: the trellis slabs are built from recycled memory. Also on-device, so `.to(device)` in `process_weights_after_loading` is a no-op rather than the rescue copy. | §5.1 — copy in `Exl3Parameter` / `Exl3MoEParameter.load_exl3_weight` when the tensor is borrowed. |
| `fastsafetensors`, world size 1 | **works (inferred)** | fastsafetensors 0.3.3 `parallel_loader.py:197` sets `need_clone = pg.size() == 1 …` → the library clones for us. vLLM passes `SingleGroup()` when torch.distributed is not initialised. | — |
| **`fastsafetensors`, world size > 1** | **incompatible (inferred)** | `need_clone` is False, and `_consume_single_batch` calls `batch.fb.close()` in its `finally` — i.e. the batch's device buffer is freed as soon as the consumer resumes past the batch's last tensor. `file_buffer.py:159`: "The returned tensor must not be used after `close()` unless the caller cloned". EXL3 retains. Every TP>1 serve we run hits this branch. | §5.1, same fix. |
| `sharded_state` | **incompatible** | Loads by `param_data.copy_(tensor)` over `model.state_dict()`. EXL3 parameters are **zero-sized** (`torch.empty(0, dtype=uint8)`); their payload lives in `param.exl3_tensors` before processing and in `layer.exl3_mixed_trellis` (a plain dict attribute) after — neither is in `state_dict()`. `ShardedStateLoader.save_model` therefore also cannot *produce* a usable checkpoint. | A save/load adapter for `exl3_tensors` + `exl3_mixed_trellis`. Large; no demand. |
| `runai_streamer_sharded` | **incompatible** | Same class (`ShardedStateLoader`), same `state_dict()` hole. | as above |
| `tensorizer` (vLLM-tensorized) | **incompatible** | `serialize_vllm_model` / `deserialize_tensorizer_model` round-trip module parameters; same `state_dict()` hole as `sharded_state`. | as above |
| `tensorizer` (HF artifact) | **untested** | Falls back to `model.load_weights(tensorizer_weights_iterator(...))`, which yields from `TensorDeserializer(..., device="cpu")`. Probably owning, but the deserializer's buffer policy was not verified and the artifact would have to be produced from our checkpoint first. | verify `TensorDeserializer` tensor ownership |
| `dummy` | **incompatible** | `initialize_dummy_weights` iterates `model.state_dict().values()`; the EXL3 params it finds are zero-sized, so `param.exl3_tensors` stays empty and `process_weights_after_loading` raises `Missing EXL3 MoE tensors for …`. Profiling shortcut is unavailable for EXL3, FQ or not. | an EXL3-aware dummy path |
| `modelexpress` | **untested** | Delegates to `modelexpress.engines.vllm.loader.MxModelLoader`; the package is not in the GG v20-r33 rootfs, so the tensor lifetime is unknown. Treat as borrowed until proven otherwise. | install + audit its iterator |
| `bitsandbytes` | **N/A** | Forced by `--quantization bitsandbytes` (`arg_utils.create_load_config`), mutually exclusive with `exl3`. | — |
| `npcache` | **N/A** | `allow_patterns = ["*.bin"]`; EXL3 checkpoints are safetensors → `Cannot find any model weights`. | — |
| `pt` | **N/A** | `allow_patterns = ["*.pt"]`; same. | — |
| `mistral` | **N/A** | Wants `consolidated*.safetensors` + `consolidated.safetensors.index.json`. | — |

There is no `LoadFormat` enum in this tree; the authority is the `LoadFormats`
`Literal` and the `_LOAD_FORMAT_TO_MODEL_LOADER` dict, both in
`vllm/model_executor/model_loader/__init__.py`. A new CPU test asserts the two
agree and that `"progressive"` is in both — that one line of ours is the
likeliest casualty of an upstream rebase.

---

## 2. The mechanism: owning vs borrowed tensors

Every loader in the tree yields `(name, tensor)`. What differs is **how long
the tensor is valid**, and vLLM has no contract that states it. Two families:

* **Owning** — the tensor keeps its storage alive by refcount. Default lazy
  safetensors (the mmap is held by the tensor's storage), `eager`, `runai`
  (explicit `.clone()`), `fastsafetensors` at world size 1 (library clone),
  `instanttensor` with `copy=True`, and our `progressive` stream.
* **Borrowed** — the tensor is a view into a buffer the loader recycles on the
  next yield and frees on context exit. `instanttensor` with `copy=False`,
  `fastsafetensors` at world size > 1.

PR #281 states the contract explicitly and even labels the tensors:

```python
if not copy_tensors:
    # Parameter loaders consume borrowed tensors synchronously.
    # Loaders retaining a tensor past this yield must materialize
    # owned storage before InstantTensor advances its ring buffer.
    tensor._vllm_instanttensor_borrowed = True
```

The EXL3 quant methods do not satisfy that contract:

```python
# vllm/model_executor/layers/quantization/exl3.py:792   (Exl3Parameter)
self.exl3_tensors[shard_id] = loaded_weight.contiguous()

# exl3.py:1216   (Exl3MoEParameter, non-preallocate branch)
self.exl3_tensors[key] = loaded_weight.contiguous()

# exl3.py:1402-1405   (Exl3MoEMethod.process_weights_after_loading)
param.exl3_tensors[key] = tensor.to(device=device, non_blocking=True).contiguous()
```

Both "copies" are identity operations on a tensor that is already contiguous
and already on `device` — asserted by `test_contiguous_and_same_device_to_are_not_copies`:

```python
t.contiguous() is t                                   # True
t.to(device=t.device, non_blocking=True) is t         # True
```

So the retention window is `[yield … process_weights_after_loading]`, i.e. the
whole load plus the whole post-processing pass. A ring buffer recycled per
yield is corrupt long before that.

Which parameters take the retaining branch matters, and it is worst exactly for
us. `Exl3MoEMethod.create_weights` (exl3.py:1369-1375):

```python
preallocate = rank_sliced and suffix in (
    {"suh", "svh"} if getattr(layer, "exl3_mixed_bitrate", False)
                   else {"suh", "svh", "trellis"})
```

* uniform rank-sliced: `trellis` is preallocated and genuinely copied
  (`target.copy_(loaded_weight, non_blocking=True)`, exl3.py:1250); only `mcg` /
  `mul1` retain — small, but they are the codebook constants, so corruption is
  silent and total.
* **mixed bitrate — the fungible-quant configuration** — `trellis` drops out of
  the preallocate set (rows differ in width per K), so the bulk of the weights
  takes the retaining branch.

`instanttensor` yields on the CUDA device
(`device = current_platform.current_device()`), which removes the one accidental
rescue: with a CPU-side loader, `.to(device=cuda)` in
`process_weights_after_loading` *is* a real copy and would have saved us.

`_prepare_mixed_rank_sliced_weights` does clear `param.exl3_tensors` once the
tier objects own compact copies (exl3.py:1705-1710), which confirms the window
closes there — and only there.

---

## 3. FQ touchpoints, one by one

**(a) `model_loader/__init__.py` — lazy `_progressive_loader_cls()`.**
Compatible with every loader by construction: it is one dict entry consulted
only when `load_format == "progressive"`. Two notes.
The dict is annotated `dict[str, type[BaseModelLoader]]` but our entry is a
*function*; `get_model_loader` only calls it, so this is fine at runtime, and
`register_model_loader`'s `issubclass` check applies only to new registrations.
The laziness is load-bearing (an eager import makes
`exl3_fungible.progressive_loader` unimportable on its own) and is now pinned by
`test_progressive_loader_is_resolved_lazily`.

**(b) `progressive_loader.py` / `progressive.py` — the segment stream.**
Owning producer. `_shard_tensor` builds `torch.frombuffer(memoryview(mm), …)`
views; each view holds a reference to the memoryview, which holds the mmap, so
the mapping outlives the generator frame. Verified two ways:
`test_progressive_stream_tensors_outlive_the_generator` (drop generator +
resolver + spec, `gc.collect()`, bytes still correct) and directly
(`del mm` then read → still correct). One change landed here: the shard *file
object* is now closed as soon as the mapping exists (`with open(shard, "rb")`),
since `mmap(2)` keeps its own dup of the descriptor — measured mid-stream, the
load window now costs one fd per mapped shard instead of two.

**(c) `v1/worker/gpu_worker.py` — `maybe_init_fq_state`.**
Loader-agnostic. It runs in `initialize_from_config`, strictly after
`load_model` (and therefore after `process_weights_after_loading`), and only
walks `runner.model.modules()` binding routers. It never sees a loader tensor.

**(d) `fragments.py` — safetensors headers + ranged bytes.**
This was flagged as the highest risk, and it is the opposite: `fragments.py` is
a *producer*, not a consumer. It parses headers itself and materializes views
over payloads it owns (`_LocalSegment`'s own mmap for local segments,
`bytes` for HF ranged reads / cache hits). It never receives a tensor from any
vLLM loader, so no loader can hand it a borrowed buffer.
`test_fragment_views_outlive_the_resolver` pins the lifetime: views stay correct
after the resolver is dropped and GC runs.

Two things worth recording rather than fixing:

* `_LocalSegment` deliberately never closes its file/mapping ("fragments hand
  out zero-copy memoryviews whose lifetime the GC ties to this mapping").
  `FragmentResolver._local_segments` caches one per `(dir, layer, K)`, so the
  cost is bounded by layers × K per manifest dir (≈ 184 fds + mappings for a
  92-layer K3/K4 model), held for the process lifetime. Bounded, not a leak.
* `Fragment.payload` is `bytes | memoryview`; `materialize` calls
  `memoryview(fragment.payload)` and every `torch.frombuffer` view keeps it
  alive. Retaining a `Fragment` retains its payload — which is the intent.

**The one place FQ consumes a tensor it did not produce** is
`swap.ResolverFragmentSource.read_expert` (swap.py:362-403), and it copies:

```python
t = dest.dest_tensor(proj, comp)
...
t.copy_(src)          # swap.py:402
```

`src` is never stored. `test_fragment_reader_copies_out_of_borrowed_payloads`
drives it with a resolver that serves every fragment out of one scratch buffer,
poisons that buffer when the next fragment arrives and on close, and stamps
`_vllm_instanttensor_borrowed` on each view; the stage still matches a
locally-staged reference afterwards and no stage buffer aliases the payload.
`test_swap_through_borrowed_payloads_is_byte_identical` does the same through
the whole `SwapEngine.apply` and compares layer fingerprints against a
`LocalSegmentSource` run. A control test
(`test_retaining_consumer_is_corrupted_by_borrowed_payloads`) reproduces the
EXL3 retain pattern against the same fake resolver and shows every tensor
poisoned — so the fake is known to model the hazard, and our passing tests mean
something.

---

## 4. `--load-format progressive`: known gaps

None of these are borrowed-buffer issues; they are features of
`DefaultModelLoader` that `ProgressiveModelLoader` does not implement. Recording
them so the matrix's "works" is not read as "works for everything".

1. **`checkpoint_weight_name_prefixes` is ignored.** `DefaultModelLoader`
   honours it (`glm4_moe_mtp.py`, `deepseek_mtp.py` and friends set it so an MTP
   head reads only its own checkpoint tensors); the progressive stream has no
   equivalent filter. Perf, not correctness — but see (2).
2. **Draft/MTP models inherit the load format.** `draft_load_config` defaults to
   the target's `LoadConfig`, so `--load-format progressive` applies to the
   speculator too, with `ProgressiveSpec.from_env(draft_model_config.model)`
   falling back to the *draft's* path for `dense_source` when
   `VLLM_FQ_DENSE_SOURCE` is unset. **Untested.** Escape hatch:
   `--speculative-config '{"draft_load_config": {"load_format": "auto"}}'`.
3. **No `secondary_weights`.** Multi-source models (Ultravox, Kimi-Audio,
   RoBERTa) would silently load only the primary source. Not applicable to
   GLM-5.2; would be a real bug if progressive were used elsewhere.
4. **No EP weight filter.** `DefaultModelLoader._init_ep_weight_filter` skips
   non-local experts before reading; progressive filters by TP rank
   (`_rank_ok`) only. EXL3 MoE already refuses expert parallelism
   (`Exl3MoEMethod.create_weights` raises `NotImplementedError` under
   `use_ep`), so this is unreachable today.
5. **No `track_weights_loading`.** Off for quantized models anyway.
6. **Local directories only.** `ProgressiveSpec` requires `dense_source` to be a
   directory, so S3/GCS model URIs cannot be combined with progressive — and
   `VllmConfig` independently restricts object-storage URIs to
   `{modelexpress, runai_streamer, runai_streamer_sharded}`. Pinned by
   `test_progressive_spec_rejects_a_non_directory_dense_source`.
7. **Extra-config keys are loader-specific.** `progressive` accepts exactly
   `{manifest_dir, policy, dense_source}` and rejects everything else, including
   the default loader's `enable_multithread_load` / `num_threads`. Switching
   `--load-format` while keeping `--model-loader-extra-config` fails fast, which
   is the right behaviour; pinned by test.

---

## 5. What would need to change

### 5.1 Make EXL3 a copying consumer (the only change that matters)

One line in each of the two retaining branches of
`vllm/model_executor/layers/quantization/exl3.py`. Sketch:

```python
def _own(t: torch.Tensor) -> torch.Tensor:
    """Materialize owned storage for loaders that lend their buffers."""
    if getattr(t, "_vllm_instanttensor_borrowed", False):
        return t.clone()
    return t.contiguous()
```

used at exl3.py:792 (`Exl3Parameter.load_exl3_weight`) and exl3.py:1216
(`Exl3MoEParameter.load_exl3_weight`). That covers `instanttensor` exactly,
because PR #281 labels its tensors. It does **not** cover
`fastsafetensors` at world size > 1, which labels nothing; the honest options
there are (a) always `.clone()` in the retaining branches — one extra device
copy per non-preallocated tensor, freed as soon as the tier objects are built,
or (b) extend the `_vllm_*_borrowed` marker to
`fastsafetensors_weights_iterator` when `pg.size() > 1`, which is the
generalisable fix and probably belongs upstream of us.

Not applied here: `exl3.py` is GG's file, other agents are working in this tree,
and the change wants a GPU test we cannot run tonight.

### 5.2 Make the contract explicit

`_vllm_instanttensor_borrowed` is a per-library ad-hoc attribute. A neutral
`tensor._vllm_borrowed = True` (or a `WeightsIterator` capability flag on
`LoadConfig`) set by *every* borrowing iterator would let consumers do the right
thing once instead of per loader. Worth proposing when #281 lands.

### 5.3 If `sharded_state` / `tensorizer` are ever wanted

They need an EXL3 serialization adapter for `param.exl3_tensors` and
`layer.exl3_mixed_trellis`, since neither appears in `model.state_dict()`. No
current demand; listed so the "incompatible" cells have a price tag.

---

## 6. Tests

`/home/mbelleau/src/gg-vllm/tests/exl3_fungible/test_loader_compat_cpu.py`
— 12 CPU tests, no GPU, no network, no model downloads.

| Test | Pins |
|---|---|
| `test_progressive_is_registered_and_declared` | `LoadFormats` literal and `_LOAD_FORMAT_TO_MODEL_LOADER` agree; `progressive` in both (static AST parse — `import vllm` needs the compiled ext) |
| `test_progressive_loader_is_resolved_lazily` | no module-scope `exl3_fungible` import in `model_loader/__init__.py`; the import lives inside `_progressive_loader_cls` |
| `test_progressive_stream_tensors_outlive_the_generator` | progressive is an owning producer |
| `test_progressive_stream_keeps_one_fd_per_shard` | shard file object closed once mapped (fails without the change) |
| `test_fragment_views_outlive_the_resolver` | `fragments.materialize` views survive the resolver |
| `test_contiguous_and_same_device_to_are_not_copies` | the torch identity the "incompatible" verdicts rest on |
| `test_retaining_consumer_is_corrupted_by_borrowed_payloads` | control: the fake borrowed loader really does corrupt a retaining consumer |
| `test_fragment_reader_copies_out_of_borrowed_payloads` | `swap.ResolverFragmentSource` copies; no aliasing of the payload |
| `test_swap_through_borrowed_payloads_is_byte_identical` | a whole `SwapEngine.apply` through recycled payloads == a local-segment run |
| `test_progressive_extra_config_rejects_default_loader_keys` | loader-specific extra config |
| `test_progressive_ignores_checkpoint_weight_name_prefixes` | §4.1/§4.3 gaps stay documented, not folklore |
| `test_progressive_spec_rejects_a_non_directory_dense_source` | object-storage URIs fail fast |

Run:

```
cd /home/mbelleau/src/gg-vllm && CUDA_VISIBLE_DEVICES="" \
  /home/mbelleau/protensors-work/vllm-voipmonitor/research/fungible-quant/runs/gg-env/gg-run.sh \
  python -m pytest tests/exl3_fungible/ -q --noconftest
```

`154 passed, 11 skipped` at the time of writing (was `142 passed, 11 skipped`;
+12, no regressions). Other agents land tests in this suite concurrently, so
the absolute count drifts — the invariant is the +12 and the zero failures.

Code change shipped with them:
`vllm/model_executor/layers/quantization/exl3_fungible/progressive.py` — close
the dense-shard file object once the mapping exists, with the lifetime rule
spelled out in the comment (the *mapping* must outlive the generator because
yielded views are zero-copy; the *descriptor* must not, because `mmap(2)` keeps
its own dup).

---

## 7. Honest list of what was not done

* No serve was booted; GPUs were off-limits. Every "works (inferred)" cell is
  source reading plus a mechanism, not an execution.
* `instanttensor` was **not** run in either mode. PR #281 is not on
  `fq/m1-stats-collector`, and the copy=0 verdict rests on the instanttensor
  0.1.9 docstring plus the EXL3 retention reading. A 5-minute GPU check
  (`INSTANTTENSOR_COPY=0` on the Fruit proxy, expect garbage output or a
  codebook assertion) would upgrade "incompatible (inferred)" to demonstrated.
* `fastsafetensors` TP>1 likewise: read from `parallel_loader.py:197` and
  `_consume_single_batch`'s `finally`, not executed.
* `modelexpress` and the HF-artifact `tensorizer` path are genuinely
  **untested** — the packages are not in the GG rootfs.
* The progressive draft/MTP interaction (§4.2) is untested and is the most
  likely place for a surprise in an FQ + speculative-decoding serve.
