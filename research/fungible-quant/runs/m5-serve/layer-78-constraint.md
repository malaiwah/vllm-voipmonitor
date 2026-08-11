# The layer-78 asymmetry

**One structural constraint has now produced four separate failures. It is
worth stating once, in the PR, rather than being re-discovered as four
unrelated patches.**

## The constraint

GLM-5.2 has 76 MoE layers: 3–77 in the main stack, plus **layer 78, the MTP
(multi-token prediction) layer**. It is a real MoE layer with 256 routed
experts and it is quantized like the others.

Two subsystems disagree about whether it exists:

| subsystem | sees layer 78? | why |
|---|---|---|
| Progressive loader | **required** | it streams weights for every layer in the bitrate map; omit 78 and the boot fails on a missing tensor set |
| Stats collector | **cannot bind** | it binds `MoERunner` modules with a `BaseRouter`; the MTP layer's router is not reached by the model runner's forward the way the main stack's is, so no activation counts ever arrive |

So the *loader's* domain is 76 layers and the *decision* domain is 75, and
every component that translates between them has to know which one it is in.

## The four failures

**1. Boot deadlock.** The loader demanded layer 78 in the policy; the
collector could not instrument it, so the loop refused a policy containing a
layer it could not observe. Neither side would yield. Fixed by separating the
loader's bitrate map from the decision domain — the policy carries 76 layers,
the loop intersects with what it can actually bind and logs the exclusion:

```
FQ loop: 1 policy layer(s) are not instrumented by the collector and are
excluded from decisions: [78]
```

**2. Silent decision-domain drift.** The loop had to be taught to intersect
rather than assume, or it would index a 75-row tier array with layer ids
running to 78.

**3. Split dense calibration.** The memory preflight calibrates the
policy-independent "dense" term by subtracting expert bytes from the measured
footprint. The loader measures 19,456 experts (76 layers) and calibrates
11.40 GiB; the decision domain covers 19,200 (75 layers) and the same
footprint implies 12.16 GiB. Both are correct for their own view — layer 78's
experts simply move across the expert/dense boundary — but the two numbers
look like a discrepancy until you know why.

**4. Rejected swap documents.** `build_target_doc` and `loop._doc_for` both
rebuilt `bits_per_expert` from the decision domain alone, dropping layer 78.
The resulting document described 75 layers against a running document of 76,
and `SwapPlan.from_policies` refused it:

```
cardinality_unbalanced — the target membership is not a same-cardinality
trade: policies cover different layers
```

An accurate complaint about a document we had malformed ourselves. Fixed in
both places by starting from the running document and overwriting only the
rows the change owns.

## The rule

> **Any structure keyed by layer must declare which domain it is in, and any
> translation between them must be explicit.**
>
> - Loader domain (76): bitrate maps, tier bitmaps, policy `bits_per_expert`,
>   anything the weight stream reads.
> - Decision domain (75): tier arrays, stats, scores, swap plans, occupancy
>   tables — anything indexed by the collector's row order.
> - Documents are written in the LOADER domain. A component working in the
>   decision domain must merge its rows into the running document, never
>   rebuild the document from its own rows.

Failure 4 is the one to design against: rebuilding a document from a partial
view produces something structurally valid and semantically wrong, so it fails
downstream in a component that is behaving correctly — which is where the
debugging time goes.

## Generality

This is not GLM-5.2 trivia. Any model with layers that are quantized but not
routed through the instrumented forward path — MTP heads, draft models,
speculative decoders, auxiliary towers — has the same shape. A design that
assumes "the set of layers I can measure" equals "the set of layers that have
weights" will break on all of them, and it will break *late*, in whichever
component first compares the two sets.
