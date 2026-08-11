# How the activations were recorded — and what was done with them

Source: `brandonmusic/GLM-5.2-EXL3-TR3v4-3.5bpw-MTP78`,
`reproducibility/r10/`. This answers the question directly and supersedes the
inference in `SELECTION-SIGNAL.md`.

## The capture

`capture/CAPTURE_PASS_SUMMARY.json`, schema `r7-capture-pass-v1`:

| | |
|---|---|
| prompts | 1,773 |
| tokens | **1,049,589** |
| layers captured | **75** (3–77) |
| recipe | `tr3-v4-r7-draft-1` |
| device | single `cuda:0` |

Bound by digest to `corpus_plan_sha256`, `carrier_inventory_sha256`,
`source_inventory_sha256`, `dispatch_audit_sha256`, `runtime_fingerprint`,
`routed_scaling_factor`, and `capture_pass_sha256`. Every probe result is
written to a `ProbeLedger` **before the next probe runs**, and the ledger
refuses to load if any of seven binding hashes drift
(`state`, `capture`, `search`, `source_inventory`, `numeric_environment`,
`runtime_inventory`, `backend_fingerprint`).

So: a live forward pass over a planned corpus, capturing per-layer MoE inputs
and routed expert ids, with the whole environment pinned by hash.

## What it is used for — a measured knapsack, not a heuristic

`r7_encoder/allocation.py`:

```python
class SensitivityCurve:
    tensor_id: TensorId          # (layer, expert, projection)
    mass: Decimal                # routing mass
    loss_by_bits: Mapping[int, Decimal]

    def gain(self, bits):
        return self.mass * (self.loss_by_bits[FLOOR_BITS] - self.loss_by_bits[bits])
```

`r7_encoder/r10_allocation.py` then solves an **exact-budget DP knapsack**:
384 upgrade units per layer, floor 3 bits, candidates {3,4,5}, maximising
summed `gain`. Fixed-point arithmetic (`Decimal`, prec 50, `ROUND_HALF_EVEN`)
with an integer score scale, and a canonical tie rule — so the allocation is
bit-reproducible rather than merely deterministic-ish.

Two details worth stealing:

1. **Monotonicity is enforced against measurement.** 5 bits is admitted only
   when 4 beats 3 *and* 5 beats 4, both measured. The docstring says the legacy
   DP "could therefore award two units to a tensor whose 4-bit measurement was
   worse than its 3-bit measurement" — a real bug, found by looking at the
   curves rather than assuming they were monotone.
2. **The loss is held out.** `CandidateLoss` carries `fit_rows` and
   `holdout_rows`. The quantizer is fitted on one split and scored on another,
   so the curve is not measuring its own fit.

## Granularity: per TENSOR, not per expert

`TensorId(layer, expert, projection)` and 384 units/layer. Our own encode's
`slice_nmse` has **3,072 keys per layer** = 256 experts × 3 projections × 4 TP
ranks. So allocation is already finer-grained than the per-expert unit
fungible quant swaps in.

That closes task #39 ("FUTURE: tensor-level fungible quant") as *already
solved offline* — the open question is whether it can be done **online**, which
is a different and harder claim.

## What this does to my analysis

`SELECTION-SIGNAL.md` reconstructed the right objective from the encoder's
outputs: I showed `count × error` ranks experts 0.96 the same as routing
frequency, and 0.00 the same as error alone. **The R10 allocator optimises
exactly `mass × Δloss`.** Same objective, arrived at from the artifacts rather
than the source. That is a genuine independent confirmation, and I should say
plainly that the source was available and would have been the faster route.

Where R10 is strictly better than what I derived:

| | my reconstruction | R10 |
|---|---|---|
| error term | one `rel_rt_mse` per expert | **measured loss at 3/4/5 bits** |
| unit | expert | **tensor (expert × projection × rank)** |
| budget | fixed cardinality (26 K4/layer) | **exact 384 units/layer, DP-optimal** |
| monotonicity | assumed | **enforced against measurement** |
| overfitting | not addressed | **held-out rows** |

My `count × error` is a first-order approximation of `mass × Δloss` that
assumes the loss curve has the same shape for every tensor. R10 measures the
shape instead. On evidence that the error spread is only 1.37× while mass
spread is 35.8×, the approximation is decent — but "decent" is not "measured".

## Consequences for fungible quant

**The offline problem is solved, and solved well.** Anything the online loop
proposes should be judged against R10's allocation, not against a flat
baseline. That is the honest comparison and it is now available: 75 frozen
`R10_FROZEN_DECISIONS_LAYER_*.json` files.

**The remaining claim is narrower and clearer.** Not "we choose experts
better" — R10 already chooses better, with held-out measured curves and a
DP-optimal budget. The claim is: *the same allocation can be delivered as
downloadable per-expert segments, assembled without a repack, restarted warm
in 350 s, and re-tiered live within a fixed memory envelope.* Every one of
those is a mechanism result and every one of them is measured.

**A concrete next step.** `R10_FROZEN_DECISIONS_LAYER_*.json` gives the
per-tensor bit assignment for layers 4–77. Comparing our K4 policy against it
tells us how much of R10's advantage a per-expert, fixed-cardinality
approximation actually captures — and that is a far better Scenario 1 than
"does the loop rediscover a coder quant", because the reference is an optimal
allocation rather than another heuristic.
