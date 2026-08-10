# Multi-K encode campaign — K2/K3/K4/K5 segments for GLM-5.2 — 2026-08-10

Michel's directive: use idle GPUs to encode the original GLM-5.2 at
K∈{2,3,4,5} and publish segments. K3+K4 is the v1 loop; K2 unlocks the
fast progressive-base story, K5 the hottest-experts headroom. Encode now,
execute later where needed (K5 runs on today's kernel —
`_TRELLIS256_BITS=(3,4,5,6)`; K2 execution is the M6 kernel item).

**Capability check (done):** encoder smoke PASSED at K2 and K5 on SM120
(byte-exact round trip, lockstep==sequential) — same for K3/K4 earlier.

## Fruit rehearsal (first)

When the Fruit capture seals: encode K2/K3/K4/K5 (one K per free GPU,
~1.8 h each), assemble K3/K4, extract per-expert proxy-error curves from
the done-JSONs for all four Ks → the 0c variance/N_L solve gets a 4-point
ε curve instead of 2. Validates the whole multi-K pipeline for ~7 GPU-h.

## Full-model campaign (the flagship)

- **Capture source**: the K3-serving model (BF16 = 1.5 TB > 768 GB total
  VRAM; activations from the live quant are the lazy-encode design's
  accepted bias — 07). Engine: capture-instrumented TP4 on one quad.
- **Encode source**: per-expert BF16 read from disk layer-by-layer
  (STReader) — no full-model load. Hessians built per window from capture.
- **Storage math** (1 M tokens × 6144 hidden × 75 layers ≈ 923 GB payload
  vs ~880 GB free): capture in TWO half-campaigns (layers 3–40, 41–78,
  ~490 GB each) or reduce target tokens; encode all four Ks per window
  BEFORE deleting its capture (never re-capture between Ks — Hessian
  identity across Ks is the point).
- **Compute**: ~13.3 GPU-h per K (2.5 s/expert × 19,200) → 4 Ks ≈ 53 GPU-h
  ≈ overnight on the box with serve stopped, or ~18 h wall on one quad.
- **Segment sizes**: K2 ~173 GB (encode ~4.8s/expert — DP table 16x K5's; K2 tier ≈ 26 GPU-h full model, corrected from 13), K3 ~260 GB, K4 ~347 GB, K5 ~433 GB
  (~1.21 TB total) → rolling publish-and-delete per layer keeps disk
  bounded; attestations carry `encode-of` predicate with hessian_id =
  capture manifest hash, encoder sha `e9a85a47…`, per-expert digests.
- **Publish**: same HF repo family as the K3 seed
  (`malaiwah/GLM-5.2-EXL3-FQ-segments`), files `layer-LLL.kK.safetensors`
  + attestations, model card updated with the K ladder and provenance
  chain. Our own K3 `encode-of` complements the brandonmusic `repack-of`
  seed (hessian-consistent with K2/K4/K5 siblings).

## Sequencing

1. Fruit capture (running) → Fruit 4-K encodes → ε extraction + 0c solve.
2. M1 integration + T1-on-GLM-5.2 evidence (serve quad).
3. Stop serve → full capture half A → 4-K encode fleet on 8 GPUs → half B
   → publish rolling. Ping Michel with budget before launch.

## Opportunistic execution (Michel, 2026-08-10)

The full campaign is NOT all-or-nothing: segments are content-addressed
and per-layer, so **whatever gets quantized and uploaded is immediately
durable value; the rest backfills over time**. Execution order therefore
goes hottest-layers-first (0c variance ranking), rolling per-layer publish
+ local delete, resumable at layer granularity across rental sessions.

## Disk expansion (Michel's offer, 2026-08-10) — recommend /home ≈ 5 TB

Storage-driven compromises removed if /home grows from 3 TB:
- HF cache 1.8 TB + full 1M-token capture 0.92 TB + 4-K segments 1.21 TB
  + workdirs/rootfs ≈ **4.2 TB working set** → 5 TB gives margin.
- **Single-pass capture** (no half-campaign split): saves a serve-stop/
  restart cycle and keeps the capture contiguous.
- **Capture is preserved** after the encodes (~13 GPU-h of product):
  future K6/K7 encodes, re-encodes, and Hessian-blend experiments reuse it
  free — otherwise it must be regenerated on GPUs each time.
- Segments stay local post-publish as primed cache for mixed-boot gates
  and 0d ladders (no re-download, no upload coupling in the critical path).

## REVISED: no disk expansion (Michel, 2026-08-10) — fully streaming campaign

Fixed 3 TB. The campaign becomes a rolling pipeline, layer-window at a time
(which is the fungible/progressive philosophy applied to its own build):

1. **Prime, don't re-encode, wherever the community already paid**: K3 base
   = brandonmusic seed (published); K4 hot fragments range-read from
   willfalco 3.36bpw (~130 GB, repack-of) — task #22. 3.42bpw per the
   layout inspection's verdict.
2. **Opportunistic encode ring** (per window of ~8 layers): stream-capture
   window (~100 GB) → encode K2..K5 for those layers (~35 GB segments)
   → publish + attest → delete window capture + local segments → next
   window. Peak transient footprint ~150 GB — fits free space with wide
   margin, forever.
3. Priority order: hottest layers first (routing mass from the capture);
   the tail follows "over time" across rental sessions — segments are
   content-addressed and idempotent, every window is durable value.
4. The kept asset under 3 TB: the per-layer Hessian statistics (small) and
   published segments on HF; raw capture windows are transient by design
   (regenerable deterministically from plan+seed if ever needed).

## Complement-encode matrix (Michel, 2026-08-10)

With the segment format nailed, idle GPUs encode ONLY what community
salvage won't prime. Per layer (256 experts):

| Tier | Primed by community | We encode (idle GPUs) |
|---|---|---|
| K3 | ALL (brandonmusic seed, published) | none — encode-of reproducibility samples only |
| K4 | 3.42bpw expansion: 108/layer (L4-77), 50 (L3); 3.36 secondary | the ~148/layer complement (policy headroom beyond primed hot set) |
| K2 | none exists | all 256 (progressive fast-load base) |
| K5 | none exists | all 256 (hot-tier headroom; or top-N by 0c benefit if trimming) |

≈660 expert-encodes/layer instead of 1024 — the ring's window step drops
to ~27 GPU-min/layer-window-K-mix at 2.5 s/expert. Ring launches per
window as stream-capture seals it; K4-complement priority ordered by the
window's routing mass (hot-adjacent first — cheapest future-promotion
insurance).

## Attestation rung 3 — equivalence-of (Michel, 2026-08-10)

Three proof strengths in the segment family:
1. `repack-of` — byte-identity with a pinned source (transport fidelity).
2. `encode-of` — recorded recipe ⇒ independent re-encode byte-matches ⇒
   countersignable (reproducible-builds rung; fires for all OUR encodes).
3. `equivalence-of` (NEW) — for legacy fragments with unrecorded recipes:
   when we encode the same expert from BF16, decode BOTH fragments and
   attest their reconstruction errors against the same BF16 ground truth,
   side by side. Signed payload: {subject fragment sha, counterpart
   fragment sha, bf16 tensor materials, eps_subject, eps_counterpart,
   decode method + tolerance}. 3-way: BF16 ⇔ our encode ⇔ community
   fragment — proves the hydrated community segment is VALID (bounded ε
   vs the real thing) without claiming unattainable byte-identity
   (their Hessians were never recorded; cross-stack activation drift).
Produce these during the K3/K4 window passes (#24) — the decode + BF16
compare is nearly free at encode time. Publish alongside the primed
fragments' attestations (#22).

## Capacity as an operator knob (Michel, 2026-08-10)

`VLLM_FQ_CAPACITY_UTILIZATION` (default 1.0) — the gpu-memory-utilization
analog for expert-tier headroom. Semantics: at startup, per-layer tier
capacity C = ceil(occupancy N / util), bounded by E and by the global byte
budget; util=1.0 reproduces v1 exactly (cap == n, trade-only); util=0.9
pre-provisions ~11% spare upper-tier rows per layer, unlocking
displacement-free upgrades within the two-ledger check (per-layer capacity
ledger + global byte ledger, L2 mechanics — kernel side verified bitwise
by the occupancy test). One knob, headroom becomes a slider instead of an
arithmetic exercise. Lands with the M2 loop wiring: policy schema already
reserves capacity fields (fq-policy/2), decide() gains the two-ledger
branch, decision_log gains "upgrade (free)" vs "swap (trade)" vocabulary.
