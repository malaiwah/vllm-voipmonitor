# Four-axis convergence — the corpus question, settled

Every MTP78 axis replayed separately against a live GLM-5.2 serve with real
gate mass recorded, scored against the human-built 3.42bpw Coder reference.
3,057 prompts per axis, **12,228 issued, 12,228 succeeded, 0 failed**.

Chance floor 0.2652. Human-human ceiling 0.6710 (the same author's 3.40bpw
non-coder sibling from identical calibration).

| axis | prompt tokens | wall | **count** | mass |
|---|---:|---:|---:|---:|
| axis1_general | 3,581,199 | 1233 s | 0.3988 (1.50x, 59%) | 0.3730 |
| axis2_legal | 1,100,565 | 399 s | **0.4240 (1.60x, 63%)** | 0.4119 |
| axis3_code_agentic | 2,631,231 | 891 s | 0.4210 (1.59x, 63%) | 0.3770 |
| axis4_reasoning_termination | 201,994 | 127 s | 0.4223 (1.59x, 63%) | 0.4153 |

75/75 layers scored in every row.

## Three findings, two of them against expectation

**1. Hit count beats gate mass — on all four axes.** Between +0.007 and +0.044.
This reverses the hypothesis that motivated binding real gate mass: weighting a
route by the router's confidence makes selection *worse* here, consistently.

A plausible mechanism, not yet tested: GLM-5.2's `GroupedTopKRouter`
renormalises (`norm_topk_prob`), so per-token weights sum to a constant. Mass
then redistributes emphasis toward experts that win their group decisively,
while the reference was built to protect experts that are *frequently* needed.
Frequency, it turns out, is closer to what the human optimised for.

The capability stays (opt-in, `VLLM_FQ_GATE_MASS=1`) — it is the right signal
to have measured, and the honest outcome is that count wins today.

**2. The code axis is NOT special.** Legal (0.4240), reasoning-termination
(0.4223) and code (0.4210) sit within **0.003** of each other — well inside
run-to-run noise. Only `general` trails, at 0.3988.

The reference is the *Coder* variant, so the intuitive prediction was that
coding traffic would match it best. It does not. What separates the top three
from `general` is not subject matter but **distinctiveness**: a corpus that
concentrates routing recovers the human's picks, and a diffuse one does not.
This is the claim I declined to make on one axis of evidence, and the evidence
now contradicts the flattering version of it.

**3. Volume is nearly irrelevant; concentration is what pays.**
`axis4_reasoning_termination` scored 0.4223 on **201,994 tokens in 127 s**.
`axis1_general` scored 0.3988 on **3,581,199 tokens in 1233 s** — **18x the
tokens, 10x the wall clock, a worse result.**

Operationally this is the most useful number here: a two-minute replay of
sharply-shaped traffic is worth more than twenty minutes of general chat.
Anyone deploying this does not need a long warm-up, they need a pointed one.

## What this does and does not establish

Established: routing frequency alone recovers **63% of the agreement two
humans reach with each other**, at **1.6x** chance, from ~2 minutes of
well-chosen traffic, on all 75 layers.

Not established: that any of this improves output quality. These runs observe
and score selection overlap; no expert was promoted, and no eval was run
against a re-tiered model. That is the next thing, and it needs K4 fragments
for the served layers — the campaign is producing them now (8/75 published).
