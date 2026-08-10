# M3 — Brutal apply: live reload-under-quiesce of the mixed-K allocation

Date: 2026-08-10. Substrate: Fruit proxy mixed-K serve (TP4, GPUs 0-3, port
8801, GG v20-r33 rootfs, CUDA graphs ON, `fp8_ds_mla` KV). Checkpoints
assembled from Progressive Tensors segments (`/home/mbelleau/fq-0c/fruit-segments`).

**Outcome: RUNG B — the real M3 — demonstrated on a LIVE HTTP serve.**
At a trigger, the running engine was quiesced (`POST /pause?mode=wait`,
drain), every mixed layer's expert allocation was rebuilt in place from a
new same-cardinality policy checkpoint (`POST /collective_rpc ->
fq_reload_experts` on all 4 TP workers), and the engine resumed — in
**0.47 s total stall, with zero request drops** under continuous traffic,
and **bit-identical post-reload logits vs a fresh boot of the same policy**
(the 04-milestones T8/T4-style gate, passed exactly, twice).

## 1. What was built

`tools/fq_reload.py` (+ `test_fq_reload.py`, 10 tests green):

- **`FqReloadWorker`** — a vLLM worker extension class
  (`--worker-extension-cls fq_reload.FqReloadWorker`; module symlinked into
  `/home/mbelleau/gg-extra`, which `gg-run.sh` puts on PYTHONPATH). Injected
  into `vllm.v1.worker.gpu_worker.Worker` at boot (worker_base.py:282);
  `worker_extension_cls` is compute-hash-ignored, so warm compile caches
  survive. Two string-args RPC methods (the `/collective_rpc` HTTP endpoint
  passes strings only):
  - `fq_reload_experts(new_model_dir, dry_run)` — the M3 apply path;
  - `fq_expert_state()` — per-layer membership digest for cross-rank
    agreement evidence.
- Driver subcommands (stdlib-only): `permute-policy`, `swap`
  (pause→rpc→resume, timed), `state`, `probe` (3-prompt teacher-forced
  prompt-logprob capture, fq_probe methodology, endpoint variant),
  `compare`, `traffic` (zero-drop evidence), `bench`, `watch-health`.

### The apply path (per mixed layer, all shapes identical by construction)

M3's contract is "rebuild the mixed layers from the artifact pair under the
new policy — in place, same shapes, startup code reused". Implementation:
re-run the *content* half of `exl3.py _prepare_mixed_rank_sliced_weights`
against the new checkpoint dir into staging tensors, then `copy_` into the
live device tensors so every CUDA-graph-baked pointer stays valid:

1. **tier slabs** — the prepared tiers' flat int32 `w13`/`w2` views.
   b12x `prepare_trellis256_moe_weights` is zero-copy (K6 audit), and the
   r33 decode + prefill tier objects alias the SAME storage (asserted via
   `data_ptr()` at swap time — `AssertionError` otherwise).
2. **rotations** — the COMBINED `gate_suh`/`up_suh`/`intermediate`/
   `down_svh` tensors at combined-slot (tier-then-membership) indices,
   per pre-m4-checks consequence #2. (r33 exl3.py builds these combined
   tensors itself and hands tier-slice VIEWS to prepare — so combined
   writes propagate everywhere; broadcast-suh/svh layouts are rejected,
   out of v1 scope.)
3. **maps** — `global_to_combined` / `descriptor_map` content `copy_`
   (graph-safe: launch args indexed in-kernel; pre-m4 checks #1/#2).
   Occupancy stays == capacity (fixed cardinality), so none of the
   absence-marking hazards of pre-m4-checks #1 apply.
4. **host metadata** — `mixed["tier_ids"]`, `layer.exl3_layer_bitrates`.

Commit order per layer follows 02-swap-engine (slabs → rotations → maps →
host). Fixed per-layer tier cardinality is enforced (hard error otherwise):
`tier_signature`, the compiled launches and the captured CUDA graphs stay
valid; membership is the only degree of freedom. Weight source = the
assembled checkpoint on NVMe (the checkpoint-form tensors are freed
post-prepare — exl3.py:1706-1710-equivalent — so the artifact is the only
source of truth; per-rank staging ≈ 427 MiB, read via self-contained
safetensors mmap).

## 2. The policy change (permute within same cardinality)

`policy-fruit-mixed-042b.json` = `permute-policy` on the 042 solve policy:
per layer, the **8 lowest-benefit K4 experts demoted to K3** and the **8
highest-benefit K3 experts promoted to K4**, benefit = (ε_K3−ε_K4)·φ_norm
from the 0c campaign work dirs (`work-k{3,4}-tr3`). 10 layers × 16 moved
experts = 160 membership changes; `n_k4_per_layer` unchanged
({3:71, 4:42, 5:79, 6:103, 7:125, 8:113, 9:106, 10:143, 11:152, 12:141}).
Assembled with `fq_assemble.py --reflink` in **3.8 s** →
`/home/mbelleau/fq-0c/fruit-mixed-042b` (3.7 GB, 12288 regions/layer
reflinked).

## 3. RUNG A — restart-swap (the honest brutal floor)

Serve restarted from `fruit-mixed-042` onto `fruit-mixed-042b` (same
serve-fruit.sh args + `VLLM_SERVER_DEV_MODE=1` +
`--worker-extension-cls fq_reload.FqReloadWorker`):

- **Downtime: 88.0 s** last-healthy → first-healthy (0.5 s poll,
  `health-restart.jsonl`; warm torch.compile/Triton caches — compile cache
  hit confirmed in the log).
- Per-layer tier log lines match the new policy exactly, all 4 ranks
  (`EXL3 mixed Trellis model.layers.L.mlp.experts: tiers=((3,n3),(4,n4))`,
  `serve-042b.log`).
- Coherent generation; decode **488.0 tok/s** (vs 483.4 pre-restart).

## 4. RUNG B — live swaps on the running serve

Continuous traffic: 1 completion request (32 tok) every ~0.3 s.
Trigger sequence: `POST /pause?mode=wait&clear_cache=true` (drain) →
`POST /collective_rpc {"method":"fq_reload_experts", ...}` →
`POST /resume`.

| step | direction | pause_s | rpc_s | resume_s | **total stall** | traffic |
|---|---|---|---|---|---|---|
| dry-run | →042 (validate only) | — | 0.440 | — | 0.44 | n/a |
| swap #1 | 042b → 042 | 0.039 | 0.426 | 0.001 | **0.466 s** | 93/93 OK, 0 drops, max latency 0.273 s |
| swap #2 | 042 → 042b | 0.040 | 0.369 | 0.001 | **0.410 s** | 91/91 OK, 0 drops, max latency 0.478 s |

All 10 layers × 4 ranks report `swapped, moved=16` each time. The
`/pause` drain + in-place rewrite means **zero requests dropped and zero
mixed-provenance tokens** (mode=wait; the M3 "seconds-long stall" budget is
beaten by ~10×: the stall is dominated by the 0.4 s per-rank staging read +
H2D, exactly as 02-swap-engine predicts for a full-model rewrite of
427 MiB/rank).

### The gate: post-reload logits == fresh-boot logits (passed exactly, twice)

3-prompt teacher-forced prompt-logprob compare (356 scored tokens total):

- **gate #1**: serve booted@042 (pre-restart) vs serve live-swapped→042:
  `identical: true`, max |Δlogprob| = **0.0** on every token of every
  prompt — across two different processes.
- **gate #2**: serve booted@042b vs the SAME process after the round-trip
  042b→042→042b: `identical: true`, max |Δ| = **0.0**; and
  `fq_expert_state` policy sha returns exactly to the boot value
  (`29bb0a958019cd1f`; the 042 state hashes `44248b1d317ce819` in between)
  with all 4 ranks agreeing at every step.

### Performance standard

| metric | 042 fresh boot | 042b fresh boot | 042b after 2 live swaps |
|---|---|---|---|
| decode tok/s (512-tok greedy, bs1) | 483.4 | 488.0 | 483.8 |
| GPU memory/GPU | 30371 MiB | 30371 MiB | 30371 MiB |

Zero memory growth (all writes land in existing storage; staging tensors
are transient) and no throughput change (identical launches/graphs).

### Quality delta of the policy change (042 vs 042b, teacher-forced)

| prompt | mean lp 042 | mean lp 042b | Δmean | max tok Δ | tokens differing |
|---|---|---|---|---|---|
| robot story | −1.6955 | −1.6972 | −0.0016 | 0.33 | 116/117 |
| tea recipe | −2.5241 | −2.5352 | −0.0110 | 0.46 | 115/115 |
| counting | −2.2047 | −2.1998 | +0.0049 | 0.92 | 122/122 |

Per-token logprobs differ on essentially every token (the swap really
changes 160 expert encodings) while the aggregate quality is near-equal —
expected, since the permutation trades adjacent ranks of the benefit
ordering. Greedy text visibly shifts (e.g. the tea completion changes
continuation between policies).

## 5. What this retires / what remains for M4

Retired per 04-milestones M3: the decide→apply lifecycle on a live engine,
policy projection into device state, quiesce choreography through the
production pause surface, CUDA-graph safety of content-only rewrites at
serve scale (graphs ON throughout; post-swap decode bit-exact). The serve
ran on `fruit-mixed-042b` with the extension loaded (further swaps = one
HTTP call) until it was stopped externally at ~19:11 by another session on
this box, which took GPUs 0-3 for a separate port-8802 experiment — after
all measurements here were complete.

Not yet M4: this is the full-rewrite reload (every mixed layer rewritten,
0.4 s stall), not the incremental row-write engine (<1 engine-step stall,
NVMe→pinned→side-stream staging, probe/rollback). Deltas M4 needs on top:
per-expert row writes instead of whole-slab copy_, staging outside the
pause, swap-list inverse rollback, probe wiring.

Operational notes:
- `/pause`+`/collective_rpc` need `VLLM_SERVER_DEV_MODE=1` (compile-hash
  ignored, warm restart safe). The extension class and its two methods are
  inert without an explicit RPC.
- `mode=wait` cannot be used on an inproc engine (EngineCore raises); the
  production multiproc serve accepts it. `keep` would freeze in-flight
  requests across the swap (mixed-provenance KV) — acceptable for M3 but
  `wait` is strictly cleaner at these stall sizes.
- The 042↔042b probes share no prefix-cache state (pause clears caches).

## 6. Files

- Tool + tests: `tools/fq_reload.py`, `tools/test_fq_reload.py`.
- Policy: `policy-fruit-mixed-042b.json` (copy; original in
  `/home/mbelleau/fq-0c/`), checkpoint `/home/mbelleau/fq-0c/fruit-mixed-042b`.
- Evidence in this dir: `swap{-dryrun,1-042b-to-042,2-042-to-042b}.json`,
  `probe-*.json`, `gate-*.json`, `quality-delta-042-vs-042b.json`,
  `traffic-swap{1,2}.jsonl`, `bench-*.json`, `mem-*.txt`,
  `health-restart.jsonl`, `restart-downtime.json`, `state-*.json`,
  `serve-042b.log`.
