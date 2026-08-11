# RETRACTED — the corpus rows in this table are INVALID

**Do not cite the MTP78 numbers below.** The replay driver scraped the corpus
loader's `--show` output, which is a HUMAN DISPLAY mode: it prefixes each row
with `[line_no] axis source`, TRUNCATES the text to 160 characters, and prints
a summary footer to the same stdout. So the "corpus" runs replayed mangled
160-character stubs with metadata glued to the front, plus a few footer lines
counted as prompts.

Two tells that were visible in the recorded evidence and that I did not catch:

* **909,790 prompt tokens / 12,237 prompts = 74 tokens each.** Real rows are
  2,616 and 4,832 characters (~650+ tokens). A 9x shortfall.
* The code-axis run reported **3,063 prompts for an axis holding 3,057 rows** —
  the 6 extras were the summary footer.

The SYNTHETIC rows (0.3603 at 18 records, 0.3789 at 117) are unaffected: they
never went through the corpus loader. Everything below is kept verbatim for
the audit trail; corrected numbers replace it in a later section.

---

# Convergence: does live routing rediscover a human's expert choices?

Measured on the real GLM-5.2 (not the proxy), from a live TP4 serve of a
checkpoint assembled by our own tool out of Progressive Tensors segments.

**Question.** A human built `willfalco/GLM-5.2-EXL3-TR3-3.42bpw` (the "Coder"
variant) by choosing, per layer, which experts deserve K4 instead of K3. If we
just *watch routing* at inference time and promote the busiest experts, do we
land on the same set?

**Reference points that make the number mean something.** Overlap is per-layer
Jaccard against the reference's K4 set, at the reference's own per-layer
cardinality — so this is a pure selection test, not a budget test.
- chance floor **0.2652** (analytic; sampled 0.2641 — they agree)
- human-human ceiling **0.6710** = the same author's 3.40bpw *non-coder* build
  from identical calibration data. Two competent humans agree at ~0.67, so
  that is the honest target, not 1.0.

## Results

| traffic | stats records | mean Jaccard | vs chance | vs human |
|---|---:|---:|---:|---:|
| synthetic math+code, short | 18 | 0.3603 | 1.36x | 54% |
| MTP78 code-agentic axis | 13 | 0.3597 | 1.36x | 54% |
| synthetic math+code, long | 117 | 0.3789 | 1.43x | 56% |
| **MTP78 full corpus** | **51** | **0.3938** | **1.48x** | **59%** |

All rows: 75/75 layers scored, ranking by routing **hit count**.

## What the comparison shows

**The corpus matters, and now we can say so fairly.** The full MTP78 replay
beat the long synthetic run — 0.3938 vs 0.3789 — with **fewer than half the
observation records** (51 vs 117). Earlier I could not separate corpus from
window length; this pair does, because the corpus won from behind.

**Window length matters too**: the same synthetic traffic went 0.3603 -> 0.3789
purely by observing 6.5x longer.

**The code axis alone did not beat the full corpus** (0.3597 at 13 records).
That is the weakest comparison here — its window was the shortest of all four —
so it is suggestive at best, and it is the run most worth repeating at a
matched window before anyone reads anything into it.

## The replay itself

| | |
|---|---|
| corpus | `reap_recall_calib.jsonl`, sha256 `cf247acc…` **verified** against the one the reference quant's manifest cites |
| prompts | 12,237 issued, **12,237 succeeded, 0 failed** |
| tokens | 909,790 prompt tokens in 610.9 s (~21 req/s sustained) |
| routing observed | 7.1M events, 19,173 of 19,200 (layer, expert) cells non-zero |

Sustained ~21 req/s for 10 minutes while a K2 encode campaign ran on the other
four GPUs — the coexistence design holding up over a long run, not just at boot.

## Honest limits

1. **Ranking by hit count, not gate mass.** The collector aliased mass to count
   (arrays byte-identical). Real gate mass is now implemented and opt-in
   (`VLLM_FQ_GATE_MASS=1`) but is NOT in these numbers. Tempered expectation:
   GLM-5.2 uses `GroupedTopKRouter` with renormalization, so per-token weights
   sum to a constant — mass will differ from count in *shape*, not magnitude.
2. **Hotness is not the reference's criterion.** The human blended
   reconstruction error with importance; we are using frequency alone. Some of
   the gap to 0.67 is a genuinely different objective, not a worse estimate of
   the same one.
3. **No promotion actually happened.** These runs observe; they do not re-tier.
   Live promotion needs K4 fragments for the served layers (currently 3-10).

## What it establishes

Routing frequency alone, on the right corpus, recovers **59% of the agreement
two humans reach with each other**, at **1.48x** the chance floor. The
selection signal is real and it is in the traffic — which is the premise the
whole fungible-quant argument rests on: if the runtime can find these experts,
a separate full download per bitrate is moving bytes nobody needed to move.

## Raw artifacts

The per-interval routing dumps are large (89 / 40 / 10 MB) and belong on HF,
not in git — GitHub warned on the 89 MB file before I moved them.

| file | sha256 (first 16) | size | location |
|---|---|---|---|
| `stats-synthetic.jsonl` | `22c478914b96832b` | 89 MB | `malaiwah/GLM-5.2-EXL3-FQ-segments` → `evidence/routing-stats/` |
| `stats.jsonl` (full corpus) | `8b20323ac86c4f44` | 40 MB | same |
| `stats-code-axis.jsonl` | `bb58a4b2e1ef4791` | 10 MB | same |

The derived `convergence-*.json` scores stay in git — they are small and are
what the claims above are read from.
