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
- **Segment sizes**: K2 ~173 GB, K3 ~260 GB, K4 ~347 GB, K5 ~433 GB
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
