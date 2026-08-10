# T1 — stats-collector graph-freeze test — PASS — 2026-08-10

The load-bearing assumption of the fungible-quant loop (03-testing-
validation.md): a capture_fn bound to `BaseRouter` before CUDA-graph
capture keeps recording during replay.

## Rig

SIQ-Fruit proxy (GlmMoeDsaForCausalLM, 10 MoE layers × 256 experts,
top-8), 1× SM120, graphs ON (default compile/capture path), hook chained
at the production binding site (`_bind_routed_experts_capturer`) — which
required `enable_return_routed_experts=True` (KEY INTEGRATION FACT: the
binding site is gated on that flag; discovered after three hollow runs.
The shipped `exl3_fungible/integration.py` therefore binds via its own
env-gated call, not that flag).

## Evidence

| Run | prompt+gen tokens | per-layer routing sums | naive expected (×8) | delta |
|---|---|---|---|---|
| 32-decode | 143 | **1128 on all 10 layers** | 1144 | **16** |
| 64-decode | 271 | **2152 on all 10 layers** | 2168 | **16** |

- Counts grow **monotonically across generations inside graph replay**
  (10800 → 21520) — a frozen capture would flatline.
- All 10 layers agree **exactly** at both run lengths — padding garbage or
  partial capture would break cross-layer identity.
- The naive-arithmetic offset is **constant (16 routings = 2 tokens × 8)**
  across a 2× change in generation length — fixed scheduler-boundary
  accounting (final sampled tokens of the two generate() calls are not
  re-forwarded), not a scaling leak.

**Corrected referee**: per-layer total == (prompt + gen − 2·num_generate
calls·…) — operationally: constant-boundary offset, validated at two run
lengths. PASS on all criteria: binding ✓, nonzero ✓, monotonic-in-replay ✓,
absolute counts explained exactly ✓.

## Consequences

- M1's collector mechanism is validated end-to-end at the production
  binding point under FULL cudagraph capture.
- Failure modes eliminated by construction of the evidence: capture-time
  freeze, replay staleness, padded-batch pollution, per-layer divergence.
- Remaining M1 gate item: decode-overhead measurement (<0.5% at cc8) on
  the GLM-5.2 serve with `VLLM_FQ_ENABLE=1` — scheduled next serve cycle.
