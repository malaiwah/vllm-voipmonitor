# K6: Per-expert in-place writes in the SparkInfer/b12x mixed-trellis path

**Sources read.** All claims are cited `file:line`.

- `SI` = SparkInfer clone at `/tmp/claude-0/-home-user-vllm-voipmonitor/eca492a8-ddf2-5fbf-8795-7c2d1229909b/scratchpad/sparkinfer` (github.com/malaiwah/sparkinfer, shallow, single commit `36cade0` on `master`).
- `GG` = the GG vLLM fork branch `gg/dev/gilded-gnosis` (available as a remote in `/home/user/vllm-voipmonitor`); its `vllm/model_executor/layers/quantization/exl3.py` was extracted to the scratchpad for reading (`scratchpad/exl3-gg.py`; line numbers below refer to the file as it exists on that branch).

**Provenance caveat (important).** The module the GG fork actually imports for the mixed path — `b12x.moe._shared.kernels.w4a16.mixed_trellis` (GG `exl3.py:225-227`), which exports `build_tiered_maps`, `combine_trellis_rotations`, `compile_mixed_trellis`, `make_mixed_trellis_buffers`, `run_mixed_trellis` (GG `exl3.py:237-245`) — **does not exist in this SparkInfer clone**. `SI sparkinfer/moe/_shared/kernels/w4a16/` contains only `__init__.py`, `host.py`, `kernel.py`, `prepare.py`, `route_pack.py`; a repo-wide grep for `mixed_trellis` returns zero hits. The GG deployment pins a b12x build that carries this extra module (or a newer/private revision). However:

1. `api.prepare_weights` **is** `prepare.prepare_trellis256_moe_weights` (GG `exl3.py:228-230, 243`), and that exact function is present, in full, in this clone (`SI .../w4a16/prepare.py:1594`). Everything about weight aliasing/layout below is therefore **verified in source**, not inferred.
2. The clone contains the mixed kernel's direct ancestor: the two-tier `W4A16FusedMoeHybridKernel` (`SI .../w4a16/kernel.py:5758`), `compile_w4a16_fused_moe_hybrid` (`kernel.py:7828`), `run_w4a16_moe_hybrid` (`kernel.py:11240`) and `build_w4a16_tier_local_map` (`kernel.py:10681`), whose descriptor encoding `(tier << 8) | local` matches the GG mock of `build_tiered_maps` bit-for-bit. Claims about the mixed kernel's runtime-map behavior are inferred from this ancestor plus the GG call sites, and are marked **[inferred]** where they exceed what the clone shows.

---

## Verdict on K6: **YES (qualified)** — per-expert row writes at stable device addresses are feasible with zero layout transform; the missing piece is *API surface*, not *memory layout*.

`prepare_trellis256_moe_weights` **aliases** the caller's slabs; it never copies, permutes, or repacks a single byte. Each expert of a tier occupies contiguous, checkpoint-native regions inside the tier slab (three regions: gate plane, up plane, w2), so a swap engine can overwrite ONE expert's 3.375–4.5 MiB (K3/K4 at H=6144, I=512) with three `copy_`s of checkpoint-native EXL3 tensors plus tiny rotation-row writes — no on-the-fly repacking math at all. The kernel streams weight bytes from gmem on every launch and reads routing maps from device tensors at runtime, so content mutation between launches (or between graph replays, given stable tensor identity) takes effect without recompile.

Qualifications:

- No per-expert write **API** exists; the write is a raw tensor mutation into the prepared tier's storage (which is legitimate: the prepared object is a zero-copy view, verified below).
- This holds for **fixed tier cardinalities and fixed bits per tier**. Moving an expert *between* tiers (K3→K4) changes both tiers' expert counts → new shapes, new `tier_signature`, megakernel recompile and new arena (GG `exl3.py:1840-1856, 1882-1899`). Membership *permutation* at fixed cardinality (swap which global expert occupies which slot) is a row write + in-place map rewrite, no recompile.
- In the GG flow the original per-expert source tensors are freed after tier assembly (GG `exl3.py:1703-1710`), so the tier slab itself is the only live copy — which is fine: it is directly writable.
- Writes must be ordered against in-flight kernels (same stream, or event-ordered side stream + quiesced apply point); nothing in the kernel double-buffers weights across launches, but there is no torn-write protection within a launch.

---

## 1. K6 in detail: what `prepare_weights` actually does

### 1.1 Zero-copy, by explicit contract

`prepare_trellis256_moe_weights(w13, w2, ..., w13_layout="trellis3_t256_proj", ...)` docstring: *"Supplying both `w13` and `w2` is the production path: **no bytes are copied or permuted**; each tensor is only viewed as contiguous int32 words and flattened."* (`SI prepare.py:1618-1623`). The implementation is `_trellis256_flat_native_view`, whose only data operation is `tensor.view(torch.int32).reshape(-1)` after validating dtype (int16/int32), shape, contiguity, and 16-byte alignment (`SI prepare.py:1556-1591`, the view at `:1591`). The returned `PreparedW4A16MoeWeights.w13/.w2` are those flat views (`SI prepare.py:1963-1967`).

Test-level confirmation: `assert prepared.w13.data_ptr() == w13.data_ptr()` and the same for `w2` (`SI tests/moe/test_fused_moe_trellis.py:625-626`). The GG-side unified-plan wrapper (`fused_moe.prepare_weights` → same function, `SI sparkinfer/moe/fused_moe/_impl.py:4915-4943`) adds only validation and metadata.

Contrast: the other two tier formats DO repack at prepare time — NF3 packs code planes into a tile_n-specific flat-span layout (`SI prepare.py:1410-1455`) and NVFP4 goes through scale permutation/swizzle (`SI prepare.py:190-255, 987`). **The trellis format is the only one where prepare is an identity on bytes** — which is exactly the format both tiers of the GG mixed path use.

### 1.2 Slab layout — checkpoint-native, expert-major, no interleaving

For `w13_layout="trellis3_t256_proj"` (the mixed/production FC1 layout, forced at GG `exl3.py:1676` and `SI _impl.py:4930`):

- **FC1 (w13)**: one projection-major backing `[2, E, H/16, I/16, 16*bits] int16` — plane 0 = gate, plane 1 = up (`SI prepare.py:1629-1632, 1797-1803`). The GG fork constructs it as `torch.stack(stack-of-experts per shard).contiguous()` from the per-expert checkpoint tensors of shape `[H/16, I/16, 16*bits]` (GG `exl3.py:1617-1627, 1633-1639`).
- **FC2 (w2)**: plain expert-major `[E, I/16, H/16, 16*bits] int16` (`SI prepare.py:1632, 1810-1814`; GG `exl3.py:1628-1632, 1640-1645`).

The innermost `[K/16, N/16, 16*bits]i16` tile order is **the native EXL3 tile encoding as stored in the checkpoint** — the test oracle `_reconstruct_native` decodes exactly that shape per expert (`SI tests/moe/test_fused_moe_trellis.py:361-390`), and prepare calls it "native EXL3 shape" (`SI prepare.py:1582`).

The kernel's global-memory addressing proves the per-expert regions are contiguous and un-permuted:

- **Plain layout** (w2, and packed-FC1): `b_src = expert_u32 * expert_idx + k16_tile * (N/16 * tile_u32) + n_tile_offset + word`, with `expert_u32 = (K/16) * (N/16) * 8*bits` int32 words per expert (`SI kernel.py:3470-3472, 3503-3510`). Expert stride = whole-expert size ⇒ each expert is one contiguous block, internally `[K/16][N/16][8*bits i32]` — byte-identical to the checkpoint tensor.
- **Projection-major FC1**: `b_src = proj * plane_u32 + expert_idx * proj_expert_u32 + ...` with `proj_expert_u32 = (K/16) * (I/16) * tile_u32` and `plane_u32 = num_experts * proj_expert_u32` (`SI kernel.py:3485-3501`). ⇒ each expert owns **two** contiguous regions, one per projection plane, each again checkpoint-native inside.

**Layout-transform answer: there is no transform.** No tile permutation, no interleaving beyond the `[2, E, ...]` plane split, no scale co-packing (trellis has no per-weight scales — the kernel const-expr-elides scale loads and shares a 4-byte dummy, `SI kernel.py:3578-3581`; `SI prepare.py:1934-1955`).

### 1.3 Per-expert write recipe and sizes

Per expert at bitrate `b`, region sizes are exactly `K*N*b/8` bytes per matrix (no overhead):

| Region | Address (byte offset into tier slab) | Size @ H=6144, I=512 |
|---|---|---|
| gate | `w13_base + (0*E + e) * (H/16)(I/16)(16b)*2` | `H*I*b/8` = 1.125 MiB (K3) / 1.5 MiB (K4) |
| up | `w13_base + (1*E + e) * (H/16)(I/16)(16b)*2` | same |
| w2 | `w2_base + e * (I/16)(H/16)(16b)*2` | same |

Total 3.375 MiB (K3) / 4.5 MiB (K4) — matching the design's per-expert budget. A swap is:

```python
w13_slab.view(2, E, H//16, I//16, 16*b)[0, slot].copy_(new_gate)   # checkpoint-native bytes
w13_slab.view(2, E, H//16, I//16, 16*b)[1, slot].copy_(new_up)
w2_slab.view(E, I//16, H//16, 16*b)[slot].copy_(new_w2)
gate_suh[slot].copy_(...); up_suh[slot].copy_(...)                  # see §4
intermediate_rotations[slot].copy_(...); down_svh[slot].copy_(...)
```

Address stability: the prepared tier objects hold the only references to the slabs after GG releases the sources (GG `exl3.py:1703-1710`), so `data_ptr` is stable for the layer's lifetime; the kernel re-reads gmem via `cp.async` on every launch (`SI kernel.py:3511-3515`), so no stale weight caching exists between launches. "Repack one expert on the fly": the artifact IS the wire format — the swap engine's repack cost is a memcpy.

**Whole-tier rebuild is the only *currently-exposed* path** (GG `exl3.py:1613-1700`: `torch.stack(...).contiguous()` per tier + `prepare_weights` per tier + `build_tiered_maps`), but nothing in the kernel or prepare layer requires it for content replacement — verified by the alias chain above.

---

## 2. `global_to_combined` and `descriptor_map`

The `mixed_trellis` module that produces them is absent from the clone, so shapes/semantics come from (a) the GG unit-test mock, (b) the in-clone hybrid ancestor, which uses the identical descriptor encoding.

- **`build_tiered_maps(tier0_ids, tier1_ids, device=...)` → `(global_to_combined, descriptor_map)`** (GG `exl3.py:1689-1691`), built purely from the two global-ID lists.
- **`global_to_combined`: int32 `[num_global_experts]`**, global expert id → *combined* slot, where combined ids are tier0-major concatenation (tier0 locals `[0, n0)`, tier1 locals `[n0, n0+n1)`). GG mock: tiers `(0,2)/(1,3)` → `[0, 2, 1, 3]` (GG `tests/quantization/test_exl3.py:337-343`). **[inferred from mock]** Its runtime role matches the single-tier `expert_map` consumed by the Triton route-packing kernels, which load it per element on device: `ids = tl.load(expert_map + safe_ids)`, invalid → −1 → route dropped (`SI route_pack.py:48-52, 178-181, 246-250, 286-289`).
- **`descriptor_map`: int32 `[n0+n1]`**, combined slot → descriptor `(tier << 8) | local_expert_id`, negative = unmapped. GG mock: `[0, 1, 256, 257]` (GG `test_exl3.py:341-343`). Encoding **verified in-clone** in the hybrid ancestor: `table[gid] = (tier << 8) | local` (`SI kernel.py:10714`, doc `:10688-10693`), decoded in-kernel as `tier = descriptor >> 8; local_expert = descriptor & 0xFF` (`SI kernel.py:5923-5926`). (In the ancestor a single map indexed by global id does both jobs; mixed_trellis splits it in two so route packing can histogram over the dense combined id space. **[inferred]**)

**Read fresh every launch — yes (verified for the ancestor, structurally inferred for mixed).** In the hybrid, both `global_topk_ids` and `tier_local_map` are ordinary tensor arguments passed on every launch of the torch custom op (`SI kernel.py:11040-11076, 11355-11370`) and are **indexed inside the kernel per mn-tile**: `gid = global_topk_ids[route_block_idx]; descriptor = tier_local_map[gid]` (`SI kernel.py:5921-5926`). The host reads only `numel()` of the map (`SI kernel.py:11386`) — no `.item()`, no content-dependent host branching. Therefore mutating map **contents** (not identity) between steps changes routing; this holds under CUDA graphs because the pointers baked into the graph are the tensors whose contents you mutate. GG passes both maps on every `run_mixed_trellis` call (GG `exl3.py:1955-1966`), and its own compile step refuses to run under capture (GG `exl3.py:1860-1864`), consistent with maps-as-data / launch-as-graph-node. Caveat for the swap engine: `build_tiered_maps` returns *new* tensors (mock; and the in-clone builder constructs on CPU then `.to(device)`, `SI kernel.py:10696-10717`), so for graph safety the swap must `copy_` results into the live map tensors rather than rebind — exactly what `implementation/03-testing-validation.md:29-30` already assumes.
- Related, verified single-tier machinery (the uniform rank-sliced path): `route_expert_map` / `output_expert_map` are contiguous int32 `[route_E]` bind-time tensors (`SI sparkinfer/moe/fused_moe/_impl.py:6947-6962`), consumed on-device by route packing and by the top-k sum kernel (`expert = expert_map_flat[raw_expert]`, `SI kernel.py:6518-6523`); the fused_moe test drives global→local mapping purely through their contents under CUDA graph capture (`SI tests/moe/test_fused_moe_trellis.py:642-664, 719-725`).

---

## 3. Two-tier limit: structural in the kernel, not a Python-API accident

Verified on the in-clone hybrid ancestor (the mixed kernel is its extension):

- The launch ABI carries exactly two tiers' weight-tensor sets (`t0_w13...t1_w2_global_scale`, `SI kernel.py:6030-6056`), and the per-tile dispatch is a compile-time two-way branch instantiating tier0's and tier1's decoders as separate const-expr arms (`SI kernel.py:5893-6027`). A third tier means new ABI parameters, a third decoder instantiation (more smem/registers — smem is already `max(tier0, tier1)`, `SI kernel.py:5880`), and passing the fail-closed spill gate (`SI kernel.py:8041-8046`).
- The descriptor format has headroom: 8-bit local field (≤256 experts/tier, enforced `SI kernel.py:5865-5868`) with tier in the high bits — 2+ tiers *encode* fine; the limit is the kernel's compiled two-arm dispatch, plus GG's hard `len(tiers) != 2` check (GG `exl3.py:1591-1595`).
- Both tiers must agree on all schedule-relevant geometry (size_m, H, I, top_k, tile config, cta_threads, ...; `SI kernel.py:5804-5860`) since tier0 drives one shared persistent schedule (`SI kernel.py:5765-5767, 6187-6190`). Any tier count/bits change flows into `__cache_key__` (`SI kernel.py:5882-5891` incl. each tier's key with `num_experts`, `trellis_bits` at `SI kernel.py:998-1019`) → different compiled kernel.
- **K2 (2-bit) support**: excluded at every layer — `_TRELLIS256_BITS = (3, 4, 5, 6)` (`SI kernel.py:130`), prepare rejects bit-widths outside 3–6 (`SI prepare.py:1548-1552, 1642-1651`), and the PTX dequant primitive raises for bits ∉ {3,4,5,6} (`SI sparkinfer/_lib/intrinsics.py:6185-6203`). The storage/addressing math (`16*bits` words/tile) generalizes to 2 arithmetically; the work is a new decode arm in the intrinsics + validation relaxation, i.e. a kernel project, matching `00-overview.md`'s D2 note.
- Note the in-clone hybrid also *forbids* trellis rotation features outright (`SI kernel.py:5796-5799`) and its element-count helper has no trellis branch (`SI kernel.py:7811-7825`) — direct evidence that the b12x `mixed_trellis` module GG pins is a real kernel-side extension (trellis tiers + per-tier bits + rotations in the two-tier grid), not a thin wrapper over what this clone ships.

---

## 4. suh/svh/rotations/scales: per-expert rows, independently updatable

Prepared-tier rotation tables (all fp16, contiguous, on the weight device; validated `SI prepare.py:1830-1912`):

- `gate_suh`, `up_suh`: `[E, H]` or broadcast `[1, H]` (kquant shared-su artifacts; `SI prepare.py:1852-1863`).
- `intermediate_rotations`: `[E, 3*I]` = `[svh_gate(I) | svh_up(I) | suh_down(I)]` per expert (`SI prepare.py:1886-1891`; layout doc `SI kernel.py:5632-5639`).
- `down_svh`: `[E, H]` or `[1, H]` (`SI prepare.py:1893-1896`).

Kernel indexing is a plain per-expert row lookup, so one row can be rewritten independently:

- suh: `s_base = expert * hidden_size + col0` (broadcast: stride 0) (`SI kernel.py:5271-5274, 4688-4690`).
- intermediate rotations: `s_base = expert * (3*I) + col0` against the persistent `[E,3I]` table (`SI kernel.py:5332-5335, 5354-5359`).
- down_svh: `svh_flat[sbase]` with the same broadcast-vs-per-expert switch (`SI kernel.py:6404-6406, 6533-6540`).

Prepare holds these by reference too (dataclass fields, `SI prepare.py:1984-1987`) — but note the GG mixed path materializes **per-tier compact copies** via `index_select(...).contiguous()` and `torch.cat` (GG `exl3.py:1653-1664`), so the writable objects are the per-tier tables inside each prepared tier, indexed by tier-local slot. `combine_trellis_rotations(*prepared_tiers)` (GG `exl3.py:1698`) packages them for the mixed launch **[not in clone; consumption inferred]** — if it concatenates into fresh storage rather than aliasing, the swap engine must write the *combined* object (or rebuild it, ~KBs); resolve against the pinned b12x build. Broadcast (`[1,H]`) suh/svh cannot be updated per expert by construction — per-expert updates require per-expert tables (both-or-neither enforced, `SI prepare.py:1866-1870`; GG slices per-expert rows from rank backing, so GG tiers are per-expert).

Scales: trellis tiers have **no** per-weight/per-group scales to update — `w13_scale/w2_scale` are a shared 4-byte dummy and `*_global_scale` is all-ones `[E]` fp32 (`SI prepare.py:1934-1958`); the bf16→"fp16-value" convention is baked into the decode path, not per-expert state.

---

## 5. Arena / workspace: sized by geometry, keyed by signature — content rewrites are free

- **Buffers** (`make_mixed_trellis_buffers(launch, device, sms)`, GG `exl3.py:1903-1907` **[module not in clone]**; in-clone ancestor `make_w4a16_packed_buffers`/`plan_w4a16_buffers`, `SI host.py:273-413`): every element count derives from `m(capacity) × topk`, `fc1_cols=2I`, `H`, route slots over `num_experts`, and `sms` (`SI host.py:283-325`). **Weight contents, bit-widths, and map contents never enter buffer sizing.** Route capacity in GG depends only on `total_experts = n0 + n1` (GG `exl3.py:1874-1881`). ⇒ Rewriting tier contents (same shapes) never requires an arena resize.
- **Compile cache**: keyed by structure only — `("w4a16_fused_moe_hybrid", device, kernel.__cache_key__)` (`SI kernel.py:7917-7924`) where the key contains map_slots + per-tier `(num_experts, trellis_bits, weight_layout, tile config, ...)` (`SI kernel.py:5882-5891, 998-1019`). ⇒ same shapes/bits = cache hit, **no recompile** for content rewrites or map-content mutation. Changing tier cardinality or bits changes `tier_signature` → GG cache miss → `compile_mixed_trellis` + fresh buffers into a module-global dict with **no eviction** (GG `exl3.py:73-76, 1844-1857, 1923`) — the known ~1 GiB leak per re-tiering; and compile is refused under capture (GG `exl3.py:1860-1864`), making cardinality/bits changes restart-level, as PLAN.md already concludes.
- **Kernel workspace**: barrier/counter int32 array sized `sms*4+2` (`SI prepare.py:178-187`; validated `SI kernel.py:11001-11008`); the persistent prepared object keeps only a 1-element placeholder and the live workspace is carved from the caller arena at bind (`SI _impl.py:4939-4942, 9762-9768`). Content-agnostic.

---

## 6. Existing harnesses to reuse for a swap test

1. **`SI tests/moe/test_w4a16_hybrid_moe.py`** — the two-tier harness. Small geometry (H=6144, I=512, 4+6 experts, `:38-47`), builds the descriptor table via `build_w4a16_tier_local_map` (`:211-216`), runs the one-grid hybrid (`:222-238`), and checks against a serial per-tier oracle with other-tier routes masked to −1 (`:141-199`). This is the closest scaffold for the K6 swap test: run → mutate one expert's slab bytes and/or map contents in place → run again → compare against the recomputed serial oracle. (Its tiers are NVFP4+NF3; on a b12x build with `mixed_trellis`, substitute trellis tiers + the mixed API, same structure. Note the map-mutation variant works in this clone today.)
2. **`SI tests/moe/test_fused_moe_trellis.py`** — the trellis-specific pieces: the zero-copy `data_ptr` assertions (`:625-626`, template for the alias check in `open-questions.md` §4), the standalone numpy EXL3-MCG decoder `_reconstruct_native` (`:361-390`) usable as a ground-truth oracle after writing one expert's bytes, contents-driven global→local mapping via `route_expert_map`/`output_expert_map` (`:642-664`), and CUDA-graph capture/replay including m-below-capacity and NaN-poisoned-arena checks (`:719-725, 803-830, 917-943`) — exactly the discipline a "mutate then replay" test needs.
3. **`SI benchmarks/benchmark_w4a16_hybrid_moe.py`** — graph-replayed decode benchmark of the two-tier grid at GLM-5.2 TP4 geometry (64+192 experts, H=6144, I=512, top-k 8; `:1-9, 44-52`) vs the serial two-launch baseline, with correctness gates before timing. Reuse to measure swap-induced disruption (replay latency around a mid-serve row write).
4. **GG `tests/quantization/test_exl3.py:280-383`** — host-side mock of the mixed prepare flow; documents `build_tiered_maps` output values and the tier partition/slab shapes without needing a GPU. Good for validating the swap engine's map arithmetic.

---

## Summary table

| Question | Answer | Confidence |
|---|---|---|
| K6 per-expert row write | YES (qualified): zero-copy alias, per-expert contiguous checkpoint-native regions; write = 3 memcpys + rotation rows; no API yet, fixed cardinality/bits only | prepare/layout **verified**; mixed-kernel consumption inferred from ancestor |
| Maps read at launch | Yes, device tensors indexed in-kernel per tile; contents mutable, graph-safe if identity preserved (mutate via `copy_`) | **verified** on ancestor hybrid; mixed inferred |
| Two-tier limit | Structural (compiled two-arm dispatch + 2-tier ABI) *and* GG-enforced; descriptor encoding has headroom; K2 blocked in intrinsics/prepare/kernel validation | **verified** |
| Rotations per expert | `[E,H]`/`[E,3I]` rows indexed by expert id; independently writable; broadcast variant is not per-expert; `combine_trellis_rotations` aliasing unresolved | **verified** (kernel indexing); combine step unresolved |
| Arena/recompile on content rewrite | Never: buffers sized by geometry, compile keyed by structure; cardinality/bits change = recompile + leaked arena (GG cache has no eviction) | **verified** |
| Harness | `test_w4a16_hybrid_moe.py` (structure), `test_fused_moe_trellis.py` (oracle + capture discipline), `benchmark_w4a16_hybrid_moe.py` (perf) | **verified** |

**Action item fallout for the plan.** The decisive uncertainty left is not layout (settled: no repack, contiguous) but *which b12x revision GG pins*: obtain the build containing `b12x/moe/_shared/kernels/w4a16/mixed_trellis.py` and confirm (a) `build_tiered_maps` output exactly matches the mock semantics, (b) `run_mixed_trellis` performs no host-side reads of map contents, and (c) whether `combine_trellis_rotations` aliases or copies tier rotation tables. All three are one-file reads against a module whose call-site contract is already fully characterized above.
