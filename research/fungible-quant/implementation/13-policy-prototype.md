# 13 — Policy engine prototype & T2 property tests (CPU de-risk)

Runnable prototype of the §3 policy engine from `01-artifacts-policy-stats.md`,
with the full T2 property-test suite from `03-testing-validation.md`. CPU-only,
pure NumPy + stdlib; no vLLM or torch imports.

## Files

- `research/fungible-quant/poc/fq_policy.py` (187 lines) — the engine:
  - `score(stats, eps, cfg)` — `(eps_k3 - eps_k4) * mass**beta * count**alpha`
    (§3.2), NaN/inf-sanitized.
  - `decide(stats, eps, tier_of, pins, dwell, cfg) -> [(L, e_out, e_in), ...]`
    — desired set (pinned-K4 forced in, pinned-K3 forced out, rest top-N_L by
    score), symmetric difference paired best-enterer-with-worst-leaver, then
    guards in spec order: dwell → hysteresis → per-layer cap → model-wide cap
    → ordered emit. All sorts use total-order keys (score, then expert id):
    bit-identical output across runs.
  - `apply_swaps` / `inverse` — membership transition and its exact inverse
    (§3.3 rollback primitive); `apply_swaps` validates swap direction.
  - `project(bits_per_expert, n_k4, order)` — §1.2 projection of a policy with
    different N_L onto running cardinality: the policy's own K4 members first
    (ranked by its own ordering), then its best K3 members if the running
    budget is larger.
  - Pins are per-(layer, expert) ints in {0, 3, 4}; `pinned: {"78": "all"}`
    maps to a row of PIN_K3 (never upgraded / excluded from the swap set).
- `research/fungible-quant/poc/test_fq_policy.py` (207 lines) — 10 pytest
  tests, also runnable as a plain script (`python3 test_fq_policy.py`, asserts
  in `__main__`) if pytest is unavailable.

## T2 coverage → test mapping

| T2 property | Test |
|---|---|
| Determinism (bit-identical repeat runs, copied inputs) | `test_determinism` |
| Budget invariant: exactly N_L K4 per layer after apply | `test_budget_invariant_after_apply`, `test_decide_rejects_budget_mismatch` |
| Pin respect (no downgrade of PIN_K4, no upgrade of PIN_K3; forced entry lands) | `test_pin_respect`, `test_pin_forced_entry` |
| Dwell respect (young experts never move; all-young ⇒ no swaps) | `test_dwell_respect` |
| Cap respect (per-layer + model-wide, saturation checked exactly) | `test_cap_respect` |
| Hysteresis (1.2× gain blocked at h=1.25, admitted at h=1.1; raising h only ever removes swaps) | `test_hysteresis_threshold_and_monotonicity` |
| Inverse restores membership | `test_inverse_restores_membership` |
| Projection of differing-N_L policy onto running cardinality (shrink/grow/extremes, exact top-N check, result re-enters `decide`) | `test_projection_onto_running_cardinality` |

## Test output (2026-08-10, Python 3.11.15, numpy 2.4.6, pytest 9.1.1)

```
test_fq_policy.py::test_determinism PASSED                               [ 10%]
test_fq_policy.py::test_budget_invariant_after_apply PASSED              [ 20%]
test_fq_policy.py::test_pin_respect PASSED                               [ 30%]
test_fq_policy.py::test_pin_forced_entry PASSED                          [ 40%]
test_fq_policy.py::test_dwell_respect PASSED                             [ 50%]
test_fq_policy.py::test_cap_respect PASSED                               [ 60%]
test_fq_policy.py::test_hysteresis_threshold_and_monotonicity PASSED     [ 70%]
test_fq_policy.py::test_inverse_restores_membership PASSED               [ 80%]
test_fq_policy.py::test_projection_onto_running_cardinality PASSED       [ 90%]
test_fq_policy.py::test_decide_rejects_budget_mismatch PASSED            [100%]
============================== 10 passed in 0.27s ==============================
```

## Deviations / decisions beyond the spec text

1. **Score exponents** `alpha` (count) and `beta` (mass) are unspecified in §3.2;
   defaulted to 1.0 each, exposed in cfg, to be set by Phase 0a.
2. **Pin-forced pairs** bypass dwell (for the forced member only — the
   displaced counterpart's dwell still protects it) and hysteresis (a pinned-K4
   expert with a poor score must still enter), and sort ahead of free pairs
   (gap = +inf) under the caps. Under tight caps a pin can therefore take more
   than one interval to land — pins are eventually consistent, never violated
   in the wrong direction (PIN_K4 never emitted as e_out, PIN_K3 never as e_in).
3. **Pairing is greedy** (best enterer ↔ worst leaver, zip); a pair dropped by
   a guard is not re-matched with another partner. Simpler, deterministic, and
   the budget invariant holds regardless (swaps are always pairs).
4. **Budget mismatch is a hard error** in `decide` (D1: cardinality fixed at
   startup); `project()` is the sanctioned repair path, verified round-trip in
   the projection test.
5. **Hysteresis monotonicity** is tested on positive-score inputs; with
   negative leaving scores the literal spec condition `s_in > h·s_out` is not
   monotone in h. Not reachable with eps3 > eps4 (the manifest guarantee), but
   worth a clamp (`max(s_out, 0)`) if that guarantee is ever relaxed.
6. **Verification loop (§3.3 probe, jaccard guard) and cadence (§3.4)** are out
   of scope here — they need a serving engine; `inverse()` provides the
   rollback primitive the probe loop consumes.

Conclusion: the §3.2 decision procedure is implementable deterministically in
<200 lines of NumPy and satisfies every T2 property on the first green run; no
spec contradictions surfaced. AI-assisted (Claude); research de-risk artifact,
not a vLLM contribution.

**Build note (2026-08-10):** this prototype is now the CPU twin of in-tree
code — `exl3_fungible/policy.py` (SwapPlan diff/inverse algebra) and
`store.PolicyStore` (write-temp + atomic rename on commit) are exercised by
the M4 swap engine's contract tests (`../runs/m4-swap/report.md`). Two of the
"out of scope" items got real inputs during the build: ε is no longer a stub
— `../tools/fq_eps.py` turns encoder-emitted per-expert rel-RT-MSE into the
4-point curves and the global budget solve
(`../runs/0c-campaign/report.md`), and the solve's output
(`policy-fruit-mixed-042.json`, `n_k4_per_layer` 42…152) was minted, booted
and live-swapped. §3.3's probe loop is still unbuilt, though `fq_probe`
(32 held-out prompts, teacher-forced logprobs) and the reload driver's
`probe`/`compare` subcommands now provide its measurement half.
