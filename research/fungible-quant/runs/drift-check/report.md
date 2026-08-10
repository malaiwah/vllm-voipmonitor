# Fresh-source drift check — 2026-08-10

Michel's standing rule: work against fresh GG + b12x HEADs. Clones under
`/home/mbelleau/src/` (blobless, single-branch).

| Repo | Branch | HEAD | vs audited/pinned | Drift |
|---|---|---|---|---|
| `local-inference-lab/vllm` | `dev/gilded-gnosis` | `e2666d9a` 2026-08-07 "refactor: complete b12x rename" | audit commit `e2666d9a` (gg-integration-surface.md), r33 base same | **NONE** — HEAD *is* the audited commit |
| `local-inference-lab/b12x` | `master` | `7cecbb2` 2026-08-09 | r33 base `9bbae678` | **3 commits**, all confined to `b12x/_lib/dense_gemm.py` + `pyproject.toml` (1.2.1 release, dense-FP6/FP8 GEMM fixes). Zero overlap with moe/trellis/maps surface |
| `brandonmmusic-max/exllamav3` | `a1-retile-sm120` | `704aefd` 2026-07-14 | r33 pin `704aefd7` | **NONE** — HEAD is the r33 pin |

## Consequence for the audit gap

The K6 audit's residual uncertainty existed because its clone **lacked
`mixed_trellis.py`**. Fresh b12x master HAS it:
`b12x/moe/_shared/kernels/w4a16/mixed_trellis.py`, plus the harness bases
named in 02 (`tests/moe/test_fused_moe_trellis.py`,
`tests/moe/test_w4a16_mixed_trellis.py`,
`benchmarks/benchmark_mixed_trellis.py`,
`benchmarks/validate_mixed_trellis_checkpoint.py`). Pre-M4 checks 1–3 are
therefore pure source reads against HEAD (== r33-relevant content, since the
3-commit drift never touches these files); only the occupancy<capacity check
needs a GPU run.

## Caveat

r33 ships *composed* trees (vLLM `fa13d334`, b12x `06db0f4b` = base + pinned
public PR heads per glm5.2_v20.md). Implementation targets GG HEAD
(= audited base); runtime validation happens in the r33 image env. Any
check that passes on HEAD but matters at runtime gets re-confirmed against
the image's installed tree once extracted (same file paths inside the
image's site-packages).
