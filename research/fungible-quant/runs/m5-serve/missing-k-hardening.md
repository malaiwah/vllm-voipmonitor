# Missing-K hardening — a weight that is not available at the required bpw

**Question (operator).** *"What happens if a weight is not available in the
required K bpw? Sure thing, it needs not to crash."*

**Answer after this change.** It does not. The resolver serves the nearest
available lower K and says so loudly; if no K can be supplied it reports the
fragment as unavailable instead of raising; the swap engine pends that
promotion and keeps the incumbent tier; the M2 interval records the failure
and retries next interval. Four crash paths that *did* exist (uncaught
`JSONDecodeError` / `ValueError` / `OSError` / `IsADirectoryError` /
`UnicodeDecodeError` / `KeyError` escaping into the loader and the swap
stager) are closed, each with a test that fails against `HEAD`.

Scope: `vllm/model_executor/layers/quantization/exl3_fungible/{fragments,
swap,loop,lazy_encode}.py` + `tests/exl3_fungible/test_missing_k_cpu.py`
(27 new CPU tests). No GPU, no network, no model downloads.

---

## 1. Call-site trace — what happens when the error propagates

`FragmentResolver.resolve()` is the only thing that raises
`FragmentUnavailableError`; `is_unavailable_error()` is the swap engine's
classifier for "supply failure, droppable" vs "structural, fatal".

| # | Call site | Before | After |
|---|---|---|---|
| 1 | `progressive.py:343` `resolver.resolve(layer, expert, int(k))` — boot weight stream | Raises out of the generator → `model.load_weights` → **engine dies at boot** | Unchanged code (see §5), but the resolver itself no longer raises for corrupt local dirs / poisoned cache / dead mirrors, and `available_k()` + `project_bits_to_available()` now exist to make boot ask only for Ks that exist |
| 2 | `fragments.py:1075` `expert_tensors()` → `self.resolve(...)` | Raises | Unchanged (strict API); `best_tensors()` added as the never-raising twin |
| 3 | `swap.py:367` `ResolverFragmentSource.read_expert` → `resolver.resolve(...)` | Classified: `is_unavailable_error` → `FragmentUnavailable`; **everything else re-raised as fatal** (an `OSError` from an evicted cache, a `ValueError` from index/segment skew, a `JSONDecodeError` from a torn index all counted as "structural") | Prefers `resolve_best()`, which cannot raise: *every* supply-side failure, including the unclassified ones above, becomes `FragmentUnavailable` and is therefore droppable |
| 4 | `swap.py:872` `stage(..., on_unavailable="drop")` | Drops the pair; `raise` (the default) fails the whole stage | Same, plus the default is now readable from `VLLM_FQ_ON_UNAVAILABLE` so a serve can make pending the global policy without every call site remembering |
| 5 | `swap.py:1060` `apply()` → `self.stage(plan)` when no batch was pre-staged | Raises **before** `with quiesce:` — no device write has happened | Unchanged, and now documented: staging is host-only and strictly pre-quiesce |
| 6 | `loop.py:559` `_maybe_apply` → `ok = bool(self.apply_fn(...))` | Unguarded. An exception aborted `run_interval`: **no decision record, no proposal in `history/`, no metrics** for that interval. `step()` caught it, so the engine lived, but the audit trail and the out-of-band reload proposal were lost | Guarded: logged, `apply_failures` incremented, `applied=False`, and the interval still explains, persists and exports |
| 7 | `loop.py:410` `step()` → `run_interval()` | Already catch-all — the engine loop was never at risk | Unchanged (outermost belt) |
| 8 | `fragments.py:718` `_get_encode_queue` / `_enqueue_encode` | Already guarded — boot never blocks on, or dies from, the encode queue | Unchanged; the queue *loader* itself is now total (§4) |

Measured on `HEAD` (`scratchpad/prove_old.py`, `prove_old2.py`,
`prove_loop_old.py`):

```
corrupt local index-k4.json                            -> JSONDecodeError    (escapes resolve())
index/segment body_offset skew                         -> ValueError         (escapes resolve())
cache write fails (ENOSPC) after a verified fetch      -> fragment discarded, FragmentUnavailableError
corrupt cached segment header                          -> JSONDecodeError    (escapes resolve())
queue file is a directory                              -> IsADirectoryError
queue file holds a torn/binary line                    -> UnicodeDecodeError
encoder template names an unknown placeholder          -> KeyError
apply_fn raises                                        -> decisions written = 0, proposals written = 0
```

`ValueError`, `JSONDecodeError` and `OSError` are **not** in
`_UNAVAILABLE_ERROR_NAMES`, so on `HEAD` each of them was fatal in the swap
stager even with `on_unavailable="drop"`, and fatal at boot unconditionally.

## 2. Worst case — can a live swap tear a layer?

**No, and it could not before either.** The reason is structural, and is now
written down in both docstrings: `SwapEngine.stage()` does *all* fragment IO
into pinned host staging, and `apply()` only opens `with quiesce:` after
staging returned. A fragment that disappears mid-swap therefore fails during
staging, where the only mutable state is scratch (`orderings` / `pre_orderings`
copies) that is discarded on the exception path. Confirmed by
`test_midswap_disappearance_drops_the_pair_and_leaves_no_trace`: no slab op,
no rotation op, no map op, no `staged_layers`, tier orderings byte-identical.

Two second-order cases checked:

* **`fail_atomic=True`.** The undo pass reads each destination slot a second
  time, so a disappearance can land on read 3 or 4 of a pair. The pair still
  drops before `pair_idx` advances, so the half-filled staging buffers are
  never referenced by an op list and are fully overwritten by the next pair.
  Pinned by `test_midswap_disappearance_under_fail_atomic_rolls_back_cleanly`
  (two-pair plan, second pair's undo read vanishes, first pair still applies,
  K4 cardinality preserved).
* **The `FragmentUnavailable` that escapes `stage()` under the `raise`
  default** used to abort `run_interval` (row 6). That is what got fixed in
  `loop.py`, not in `swap.py` — the fail-closed default is deliberate (see
  `test_trust_rejection_raises_by_default`) and is preserved.

One residual wart, left alone: an all-dropped batch still bumps
`engine.generation` and still runs `memo_hook`/`policy_store.commit`. Harmless
(`staged_layers` is empty) as long as callers build the persisted policy from
`staged.plan`, which the docstring already requires.

## 3. What changed

### `fragments.py`

**Two entry points, deliberately different about failure.**

* `resolve()` — unchanged strict contract: explicit `VLLM_FQ_K_FALLBACK`
  ladder only, raises when nothing supplies the fragment. Tooling, audits and
  the trust tests keep their fail-closed behaviour.
* `resolve_best()` — **never raises**. Requested K → nearest available lower
  K → `None`. Returns the decision chain through `chain_out` so callers can
  quote the real reason. `best_tensors()` is the materialising twin.

**Automatic nearest-lower-K ladder.** `k_universe()` discovers the Ks a
deployment can plausibly supply (manifest `k_values`, `index-k*.json` in every
local segment dir, every K already consulted on a source; `DEFAULT_K_UNIVERSE`
if nothing advertises). `fallback_ladder(k)` orders them **lower-first,
nearest-first** — a lower bitrate always fits the memory the higher one
needed. Upward substitution is opt-in (`VLLM_FQ_K_FALLBACK_UP=1`): a higher K
costs memory, and on SM120 K5 does not serve as a mixed tier at all
(`k5-shared-memory-limit.md`). `VLLM_FQ_K_FALLBACK=off` disables substitution;
an explicit comma list still wins verbatim.

**Loud on degradation.** Every substitution now emits, on top of the existing
structured chain line:

```
FQ DEGRADED L17/e204: K4 unavailable, serving K3 instead (origin=local) (encode queued #3)
```

and a total miss emits `FQ UNAVAILABLE ... — keeping the incumbent tier` at
ERROR.

**Crash paths closed** (each was an uncaught exception out of `resolve()`):

| Failure | Now |
|---|---|
| corrupt / truncated `index-k{K}.json` | that dir is a MISS, warned once per `(dir, layer, K)`, memoized; `local_error` counter |
| index vs segment `body_offset` skew, unreadable segment | same |
| unreadable attestation file | that dir skipped, next dir tried |
| fragment-cache entry evicted or unreadable mid-read | cache step skipped, source chain still walked; `cache_error` |
| corrupt cached **segment header** | entry discarded and **re-fetched** (self-healing), not a permanent poison; `cache_error` |
| cache write fails (ENOSPC / read-only) | the fetched-and-verified fragment is still **served**, uncached; `cache_write_error` |
| any exception from a source | `_try_source` has an outer guard, so "never raises" is literal, not aspirational |
| `_tensor_table_for` reaching a dead mirror | per-source guard, try the next |

**Boot-side projection.** `probe()` / `available_k()` answer "could this
`(layer, expert, K)` be supplied?" without transferring a payload;
`project_bits_to_available(resolver, bits_by_layer)` rewrites a policy's
per-expert Ks down to what exists and reports `(projected, substitutions,
missing)`. See §5 for why this matters and what still has to be wired.

New counters (all in `resolver.stats`): `local_error`, `cache_error`,
`cache_write_error`, `resolve_error`.

### `swap.py`

* `ResolverFragmentSource` prefers `resolve_best()` when the resolver exposes
  it (duck-typed, with a signature probe for third-party resolvers that lack
  `chain_out`). Consequence: unclassified failures become droppable supply
  failures rather than fatal structural ones. Genuinely structural faults —
  a missing tensor in a resolved fragment, a shape mismatch, a foreign
  MCG codebook — stay fatal, as `test_structural_faults_stay_fatal_even_when_dropping`
  requires.
* `stage(on_unavailable=None)` resolves its default from
  `VLLM_FQ_ON_UNAVAILABLE` (`raise`|`drop`, default `raise`). **Recommended
  serve setting: `VLLM_FQ_ON_UNAVAILABLE=drop`.**

### `loop.py`

* `_maybe_apply` treats the apply backend as untrusted: exceptions are logged,
  counted in `apply_failures` (also surfaced on the decision record as
  `apply_failures`), and downgraded to `applied=False`. `tier_of`,
  `policy_doc` and `current.json` only advance on a clean `True`, so the
  incumbent tiering stays live and authoritative. The interval still explains,
  persists the decision, writes the proposal to `history/` and exports metrics.

### `lazy_encode.py`

* `EncodeQueue` loading is **total**: torn last line, truncated file, invalid
  UTF-8, a JSON value that is not an entry, a path that is a directory or
  unreadable — all degrade to "skip that line" / "empty queue", counted in
  `corrupt_lines` and reported on stderr. Appends and dedup keep working.
* `render_cmd` is lenient about unknown `{placeholders}` (renders them
  verbatim) so one typo in `VLLM_FQ_ENCODER_CMD` cannot abort a drain.
* `drain()` guards per entry and around the encoder launch: one unusable entry
  or a missing encoder binary is a reported failure, not an aborted drain.

## 4. Lazy-encode wiring, verified end to end on CPU

`test_miss_lands_in_the_queue_file_and_drains_to_a_sane_command`: a resolver
miss appends `{"layer":3,"expert":2,"k":4,"reason":"unavailable",...}` to the
persisted queue; a second `EncodeQueue` (i.e. another process) reads it back;
`drain(dry_run=True)` validates the BF16 index and the Hessian capture dir and
renders

```
#1 L3/e2 K4 reason=unavailable DRY-RUN OK bf16=ok(index) capture=ok(layer_003) cmd: python .../fruit_encode_driver.py --encode --bits 4 --layers 3 --src <bf16> --capture-dir <capture> --workers 1 --gpus 0
```

Also pinned: missing BF16/capture reports `BLOCKED` instead of encoding; a
corrupt/partial/binary queue keeps its parseable entries and stays writable; a
queue path that is a directory yields an empty queue and a resolver that still
resolves; a bad encoder template and an encoder that cannot start are both
reported, not raised.

## 5. Residual gap — boot, and why it is not closed in this commit

A progressive boot is only crash-free if the **tier bitmap the model was
configured with agrees with the Ks the loader streams**: `exl3.py`
`_load_rank_sliced_bitrates` sizes the slabs from that bitmap, so a fragment
substituted at a different K would fail a shape check deep in the weight
loader instead of crashing at the resolver. Today `python -m
...exl3_fungible.progressive` writes the bitmap straight from the policy,
with no availability check.

The fix is a pre-flight projection: run `project_bits_to_available()` before
writing the bitmap so the policy only ever asks for Ks that exist, then boot
cannot miss at all. Both halves of that (`probe`, `available_k`,
`project_bits_to_available`) landed here and are tested
(`test_project_bits_to_available_matches_what_boot_would_stream`); the
one-line wiring into `progressive.main()` is **not** in this commit because
another agent had uncommitted work in flight in `progressive.py`
(`progressive_weights_iterator`) and staging that file would have picked up
their change. Next step: call the projection from `main()` behind
`--project-availability` (default on) and write the bitmap + effective policy
from the projected bits.

Unaffected either way: **boot never blocks on an encode.** Enqueue is a single
O(1) append behind two layers of guards (`_get_encode_queue`,
`_enqueue_encode`), and the queue loader can no longer throw.

## 6. Operator settings

```bash
VLLM_FQ_ON_UNAVAILABLE=drop   # a pending promotion never fails an interval
VLLM_FQ_K_FALLBACK=auto       # default; nearest available LOWER K
# VLLM_FQ_K_FALLBACK=3        # or pin the ladder explicitly
# VLLM_FQ_K_FALLBACK=off      # or refuse to substitute at all
# VLLM_FQ_K_FALLBACK_UP=1     # allow upward substitution (NOT for SM120 K5)
```

Watch for `FQ DEGRADED`, `FQ UNAVAILABLE`, `FQ apply failed`, and the
`fallback_substituted` / `unavailable` / `local_error` / `cache_error` /
`cache_write_error` / `resolve_error` counters in `resolver.stats`.

## 7. Performance

Everything here is on the boot / interval path, never on the per-token path:
`step()` between intervals is untouched, and `resolve*` is called during weight
load and during swap staging only. The auto ladder adds one memoized directory
glob per resolver and a small set union per call, and is only *walked* on a
miss. The added guards are `try` blocks around IO that already dominated the
cost. New counters are plain integer increments on the same paths.

## 8. Tests

`tests/exl3_fungible/test_missing_k_cpu.py` — 27 tests, CPU only:

* resolver with nothing → `resolve_best` returns `None`, miss queued, strict
  `resolve()` still raises; a resolver *bug* also degrades to `None`
* only a lower K → nearest-lower substitution, `FQ DEGRADED` logged,
  `requested_k` recorded, encode queued; upward substitution opt-in; `off`
  honoured; explicit ladder honoured; `available_k` / projection
* HF source that times out → local lower K still served; nothing local → clean
  `None`; a corrupt cached header self-heals; a mirror dying mid-header is
  droppable, not fatal
* corrupt local index, index/segment skew, unwritable cache
* mid-swap disappearance (plain and `fail_atomic`), env-configurable default,
  resolver `OSError` classified as supply not structure
* apply backend raising → interval still completes and persists; `step()`
  keeps stepping across repeated failures
* lazy encode: miss → queue file → drain command; blocked inputs; corrupt /
  binary / torn queue; queue path is a directory; bad template; dead encoder

Suite: **169 passed, 11 skipped** (142 pre-existing + 27 new; the
`test_loader_compat_cpu.py` in that directory is another agent's uncommitted
work in progress and is excluded).

```bash
cd /home/mbelleau/src/gg-vllm && CUDA_VISIBLE_DEVICES="" \
  research/.../gg-run.sh python -m pytest tests/exl3_fungible/ -q --noconftest
```
