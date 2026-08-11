# Upstream issues filed from the fungible-quant audit

Filed against `local-inference-lab/vllm` (issues are enabled; `has_issues: true`).
Both were checked against every open issue and against the mixed-Trellis / loader
PR set for duplicates before filing — nothing overlapping, see the notes below.
Bodies as filed are reproduced from `SEPARATE-REPORTS.md` with the corrections
described under "What changed against the draft".

| # | URL | Summary |
|---|---|---|
| 282 | https://github.com/local-inference-lab/vllm/issues/282 | EXL3 retains every loader tensor until `process_weights_after_loading`, and its two "copies" (`.contiguous()`, `.to(same device)`) are documented no-ops, so borrowed-buffer load formats — `instanttensor` with `INSTANTTENSOR_COPY=0` (reachable since #281 merged as `5d2079094`) and `fastsafetensors` at world size > 1 — can silently corrupt any EXL3 checkpoint. Labelled INFERRED: source reading plus CPU control tests, no loader was booted. |
| 283 | https://github.com/local-inference-lab/vllm/issues/283 | Mixed-Trellis prefill route block and tile are picked from a GLM-5.2 tier-signature allowlist and from layer geometry, never from `max(tier_bits)`. A two-tier K3+K5 split falls outside the allowlist, gets block-64, and the K5 FC1 kernel needs 109,568 B against SM120's 101,376 B opt-in limit. Observed on the r33 image; concerns PRs #228 and #280, not the current base branch. |

## Verification basis

- vLLM line numbers in #282 are against `dev/gilded-gnosis` @ `fa033bd4e1b16d9d729ad94be2d87da5a13210ce`,
  which contains `5d2079094` (#281). Every cited line was re-read at that commit.
- b12x line numbers are against `local-inference-lab/b12x` @ `7cecbb2`.
- #283 additionally cites PR #228 @ `5ec9357` and PR #280 @ `8e7be4d`, because the
  code it concerns is not on the base branch.

## What changed against the draft in `SEPARATE-REPORTS.md`

Report (b) as drafted would have been wrong in two ways, and was rewritten:

1. **It is not a bug on `dev/gilded-gnosis` today.** The base branch compiles both
   the mixed decode and the mixed prefill state at `_MIXED_TRELLIS_ROUTE_BLOCK_SIZE = 8`
   (`exl3.py:76`, `:1893`, `:1910-1915`), i.e. `cta_m_blocks = 1`, where K5 needs
   59,520 B and fits easily. The block-64 prefill state and the failure arrive with
   PR #228 and are carried by #280.
2. **#280 already knows the number.** `_is_glm52_block32_tier_signature`
   (#280 `exl3.py:125-133`) carries a comment naming 109,568 B, and
   `_resolve_mixed_trellis_prefill_block_m` already drops to block-32 for the tier
   signatures it qualified. The real finding is that the mitigation is an
   *allowlist* — three exact `((3,N),(4,M))` splits plus a `(3,4,5)` special case —
   so a `(3,5)` split still gets block-64 and still dies. The issue is framed as
   "compute the condition instead of enumerating it", crediting #280's prior work.

Both drafts also had numeric provenance problems that were fixed:

- The draft's footprint table came from `_shared_memory_footprint`
  (`kernel.py:326`) called with the default `scale_format="e4m3_k16"`. The mixed
  path passes `e4m3_k32` (`mixed_trellis.py:834`), and that helper is a planner-side
  estimator that omits the route-metadata term — it does not produce the 109,568 in
  the traceback at any plausible configuration. The filed issue instead reconstructs
  `W4A16GemmKernel.__init__`'s own accounting (`kernel.py:1014-1135`), which
  reproduces 109,568 for K5 and 101,376 for K4 at block-64 exactly, and ships that
  reconstruction as a runnable CPU snippet.
- "K4 at the failing configuration is exactly the opt-in limit, to the byte" was a
  chained inference in the draft. It is now an exact result of the reconstruction,
  and is independently corroborated by b12x's own comment at
  `mixed_trellis.py:849-852`.
- The draft's suggested fix ("drop to `cta_m=2 / 128x128`") was rephrased: `cta_m`
  is derived from the route block (`kernel.py:1014`), so that is a block-32 choice,
  not a free tile choice — and it is a path the tree already has. The draft's
  fallback suggestion of reusing `_select_tile_config` was qualified, because
  `_candidate_tile_fits` rejects `tile_k < 64` (`kernel.py:444`) and the mixed FC2
  tile is `32x512`.

Report (a) survived verification intact; every cited line and both loader-library
quotes were re-checked against the installed `fastsafetensors` 0.3.3 and
`instanttensor` 0.1.9 in the r33 rootfs. Two presentation changes: the
`.contiguous()` / `.to()` identities are now attributed to the documented torch API
contract rather than only to our CPU test, and the report leads with an explicit
"this is INFERRED, not OBSERVED" statement plus the exact experiment that would
settle it.

## Follow-ups

- `PR-BODY.md` still references `SEPARATE-REPORTS.md`; replace those with #282 and #283.
- #283 is worth cross-referencing from #280 once someone with triage rights can do it
  (this account has pull-only permissions, so no labels were set on either issue).
