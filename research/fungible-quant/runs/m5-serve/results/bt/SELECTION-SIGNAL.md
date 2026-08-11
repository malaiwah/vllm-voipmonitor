# Which signal should pick the experts? — evidence from the encoder's own data

Michel asked what the MTP78 quant used to select and activate experts. The
answer turned out to be "nothing — all 256 experts, same bitrate", and chasing
*why* produced the clearest validation of the fungible-quant premise so far,
by way of a result that first looked like a refutation.

Data: `expert_routed_count` and `expert_rel_rt_mse` from the encoder's own
`layer-*.done.json`, for **all 75 main-stack layers** of our K4 encode, plus
layer 78 from `malaiwah/GLM-5.2-EXL3-TR3-MTP78` (7.29M tokens of captured
routing).

## 1. The owner's quant does not allocate bits per expert

`3bpw-keep0/` — the validated flagship — encodes **256/256 experts at 3.0 bpw**.
The 7.29M-token routing capture fed **LDLQ Hessian calibration**: each expert's
quantizer is fitted on that expert's own traffic. It was never a bit-allocation
input.

The one variant that differentiates, `keep64`, protects "the 64 highest
trellis-roundtrip-error experts in BF16" — selecting by **quantization error**,
not routing frequency. It is also unusable on the current runtime (no mixed
BF16+trellis expert path), and the base checkpoint's convention is
`nvfp4_keep_per_layer: 0`.

## 2. Hot experts quantize BEST — strongly and universally

| | median over 75 layers |
|---|---|
| Spearman(routed_count, rel_rt_mse) | **−0.7786** |
| layers with negative correlation | **75 / 75** |
| overlap: top-26 by routing vs top-26 by error | **0 / 26** (chance 2.6) |

The mechanism is the encoder's, not the model's: LDLQ fits each expert on its
own routed traffic, so an expert seen 1.16M times gets a well-conditioned
Hessian and an expert seen 8,526 times does not. **More traffic ⇒ better
calibration ⇒ lower error.**

At this point the obvious reading is that promoting hot experts spends bits
where the error is *already smallest*, i.e. backwards. **That reading is
wrong**, and it is wrong for a reason worth stating.

## 3. Frequency dominates, so frequency is the right signal

Expected contribution to output error goes roughly as **frequency × error**.
The two spreads are not comparable:

| | median |
|---|---|
| routing count spread (max/min) | **35.8×** |
| rel-RT-mse spread (max/min) | **1.37×** |

A 36× lever beats a 1.4× lever. Ranking experts by `count × error`:

| top-26 by contribution vs… | median overlap |
|---|---|
| top-26 by **routing frequency** | **0.96** |
| top-26 by **quantization error** | **0.00** |

**Selecting by routing frequency is 96% identical to selecting by expected
error contribution.** Selecting by error alone would be almost exactly wrong —
zero overlap with the experts that actually matter.

So the fungible-quant design picks the right signal, and the encoder's own
telemetry says so. `keep64`'s error-based selection optimises a different and
defensible objective (worst-case weight fidelity), but it is not the objective
that minimises expected output error.

## 4. The ceiling is still modest, and it is the real constraint

| | median |
|---|---|
| top-26 share of routing mass | 24.6% |
| top-26 share of error contribution | **24.1%** |

A per-layer budget of 26 K4 experts out of 256 addresses roughly **a quarter**
of the expert-quantization error. Perfect selection cannot exceed that; the
remaining ~76% sits in the other 230 experts.

Combine with the earlier live measurement — the desired top-K set churns ~39%
between adjacent intervals, and 15× more sampling barely moves it — and the
picture is coherent:

- **the signal is right** (frequency ≈ contribution, 0.96),
- **the ceiling is ~24%** of expert-quantization error at this budget,
- **the online estimate is unstable** because the distribution is flat enough
  that top-K boundaries are near-ties.

## 5. What this changes

**Keep.** Ranking by routing frequency. It is not a heuristic standing in for
something better — it *is* the something better, to within 4%.

**Drop.** Any plan to rank by quantization error, or by `expert_rel_rt_mse`.
Zero overlap with contribution. (`score_convergence.py` already refuses to rank
by `expert_rel_rt_mse` for a different reason — circularity — and that refusal
now has a second justification.)

**Reconsider.** Online per-interval re-tiering. The signal is right but the
per-interval estimate is not stable enough to act on, and the ceiling does not
leave much room to pay for churn. Offline corpus-level selection (Scenario 1)
uses the same signal with a stable estimate.

**Report.** That hot experts quantize best is a genuine property of
LDLQ-calibrated encoders and would mislead anyone who ranked on error without
weighting by frequency. It cost this analysis two steps to see; it is worth one
paragraph in the PR so it costs the next person none.
