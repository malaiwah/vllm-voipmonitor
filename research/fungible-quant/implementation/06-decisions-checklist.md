# 06 — Decision checklist & effort summary

Answers: can we proceed on defaults? what needs deciding? how much work?

## 1. Settled (no action): D1–D9 (`00-overview.md`; D3 revised to D3′ by
`07-lazy-encode.md` — lazy K4 encode-and-cache, K3 base artifact only),
K6 → row-write path (`02-swap-engine.md`), L2 door-openers as v1 defaults
(`05` §5), all runtime knobs (`01` §4 — Phase 0 refines, never gates).

## 2. Open decisions

### Need the operator's call before M0

| # | Decision | Default | Trade |
|---|---|---|---|
| P1 | K3 base provenance: fresh encode vs reuse existing K3 checkpoint | Reuse if `hessian_id` consistency holds | Under D3′ only the K3 base ships; the measure campaign still runs once for ε curves + kept Hessians |
| P2 | ~~Encode venue~~ **dissolved by D3′** (`07-lazy-encode.md`) — K4 encodes lazily on the serving box | `VLLM_FQ_K4_SOURCE=lazy` | `artifact` mode remains available for boot-complete deployments |
| P3 | Ks the lazy encoder may target: {3,4} vs {3,4,6} | {3,4} | Config not campaign now; +K6 just widens the cache |

**Build note (2026-08-10) — P1–P3 settled by the build.** **P1**: the K3
base is a *repack* of `brandonmusic@9297b9f1` (byte-identity verified,
79/79 shards) — no fresh K3 encode was needed. **P2**: stays dissolved, but
`artifact` mode turned out cheap anyway (~13 GPU-h for a full K4 overlay,
not 41). **P3**: the answer is **{2,3,4,5}**, not {3,4} — the encoder accepts
bits 2–5, K5 runs on today's kernel, K2 is encode-now/execute-later, and the
0c campaign produced all four from one hessian-identical capture
(`../runs/0c-campaign/MULTI-K-PLAN.md`). Also note the encoder itself is not
stock exllamav3: `convert_model` cannot load `GlmMoeDsaForCausalLM`, so the
canonical encoder is the sha-pinned `encode_tr3_v31.py` (`e9a85a47…`) bundle
shipped inside the K3 repo (`../runs/0c-campaign/PIVOT.md`).

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
