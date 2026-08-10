# 06 — Decision checklist & effort summary

Answers: can we proceed on defaults? what needs deciding? how much work?

## 1. Settled (no action): D1–D9 (`00-overview.md`), K6 → row-write path
(`02-swap-engine.md`), L2 door-openers as v1 defaults (`05` §5), all runtime
knobs (`01` §4 — Phase 0 refines, never gates).

## 2. Open decisions

### Need the operator's call before M0

| # | Decision | Default | Trade |
|---|---|---|---|
| P1 | Artifact provenance: fresh K3+K4 pair from one campaign vs reuse existing K3 + encode K4 only | Fresh pair | Reuse ≈ halves encode bill, valid only if Hessian/calibration provenance matches (`hessian_id` consistency) |
| P2 | Encode venue + budget | Rented, one-time (~40 GPU-h, ~5–10 h wall parallelized) | Only cash outlay before M3 |
| P3 | v1 K variants: {3,4} vs {3,4,6} | {3,4} | +K6 now: ~519 GB NVMe + hours; enables per-layer bit-pair (L1) without a second campaign |

### Defaults to accept-or-veto (no research needed)

| Decision | Default |
|---|---|
| Artifact storage | Unsharded, per-expert-addressable (slice at load; 16-col tile granularity) |
| Code home | `exl3_fungible/` inside the GG vLLM fork |
| MTP-78 | Pinned out of swap set (already uniform-K3 via tail_tr3) |
| Starting generic policy | Uniform 108 K4/layer (3.42 bpw point) until 0c's global solve replaces it |
| Probe set | Fixed-seed ~32 prompts from the 4-axis corpus, held out |
| ε source | Stock exllamav3 calibration mix; live traffic reweights φ/w only (TASA) |
| Policy schema | `fq-policy/2` with capacity fields, v1 writes `cap == n` |

## 3. Effort to v1 (one GG-familiar contributor, part-time)

| Track | Effort | Parallel with |
|---|---|---|
| Phase 0 (knob-setting) | 3–5 d | everything |
| M0 artifacts | ~1 wk elapsed (GPU-hours dominate) | M1 |
| M1 collector | 3–4 d | M0 |
| M2 policy + dryrun | ~1 wk | — |
| Pre-M4 b12x checklist (4 one-file checks) | ~1 d | M3 |
| M3 brutal apply | 3–4 d | — |
| M4 atomic swap | 2–3 wk | — |
| M5 soak + release | ~1 wk | — |

**Total ≈ 6–8 weeks.** Early exits that are still products: **M2** (~2.5 wk
in): specialize-with-reboot. **M3** (~3 wk): full loop, seconds-long apply.

## 4. First actions once P1–P3 are answered

1. Post the #49702 comment (deadline 2026-08-18; draft in `../drafts/`).
2. Start 0a (pandas on the existing trace) and M1 (collector) — neither
   depends on P1–P3.
3. Kick off the P2 encode campaign → M0.
