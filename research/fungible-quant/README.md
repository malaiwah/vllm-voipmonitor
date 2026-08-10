# Fungible Quant — research archive

Research materials for the "fungible quant" proposal: periodic, workload-driven
reallocation of per-expert bit-width (EXL3 trellis K3/K4) in a live vLLM server,
analogous in cadence — but not in mechanism — to EPLB.

Target stack: GLM-5.2 / EXL3 on 4× RTX PRO 6000 Blackwell (SM120), TP4+DCP4, EP=1,
Gilded Gnosis vLLM fork + SparkInfer/b12x.

Produced 2026-08-10 by a multi-agent research campaign (27 agents: 5 prior-art
sweeps, 5 vLLM feasibility lenses with adversarial verification of every
hard/blocker claim, completeness critic, architect synthesis), assembled with
Claude Code. All findings are AI-generated research: **verify citations and
line references before relying on them in public artifacts.** A follow-up
verification pass over the arXiv IDs and vLLM issue numbers is recorded in
`verification.md`.

## Contents

| File | What it is |
|---|---|
| `PLAN.md` | The main deliverable: verdict, decomposition, chosen swap design (fixed-cardinality tier membership permutation), policy design, phased plan (0–5), upstreaming strategy, kill criteria. |
| `fungible-quant.html` | The same document typeset as a standalone page. |
| `open-questions.md` | Completeness-critic output: top unresolved questions with cheapest resolving experiments. |
| `prior-art.json` | Raw findings from the 5 prior-art lenses (mixed-precision allocation, MoE-aware quant, runtime-swap systems, vLLM history/duplicates, EXL3 trellis mechanics), with citations. |
| `feasibility-claims.json` | Raw claims from the 5 vLLM code lenses (weight lifecycle, runtime mutation precedents, cudagraph/compile constraints, memory/arena, stats/hooks), each with file:line evidence, risk rating, and adversarial verdicts on hard/blocker claims. |
| `vllm-ep-architecture.md` | Foundational report: EP config surface, process groups, TP-vs-EP MoE dispatch, all2all backends, DP+EP coupling, constraints. Against tree `99267c23`. |
| `vllm-eplb-internals.md` | Foundational report: EPLB concepts, mapping tables, load-statistics substrate, rebalance algorithm/execution, trigger cadence, model interface. |
| `gg-fork-ep-dcp-deltas.md` | Foundational report: what the `dev/gilded-gnosis` fork changed re: EP/EPLB/MoE/DCP (163 GG-authored commits vs upstream drift); EXL3/b12x EP gates. |
| `verification.md` | Post-hoc verification of arXiv IDs and vLLM issue/PR numbers cited in PLAN.md. |
| `drafts/` | Draft upstream comments (RFC #49702, RFC #48920) prepared for human review — **not posted**; to be reviewed, edited, and posted by the human contributor per AGENTS.md accountability rules. |
| `implementation/` | **Implementation-ready spec targeting GG** (decision: build the full dynamic loop in the fork). `00-overview` (architecture, decision log D1–D9), `01-artifacts-policy-stats` (formats, collector, policy engine, knobs), `02-swap-engine` (row-write design, commit protocol — finalized against the K6 audit), `03-testing-validation` (T1–T9), `04-milestones` (M0–M6 build order), `gg-integration-surface.md` + `k6-sparkinfer-mixed-trellis.md` (agent-audited code maps with file:line cites). |

## Key line references

- vLLM merge-base for all `file:line` cites: `99267c23` (this repo's `origin/main`).
- GG fork branch inspected: `gg/dev/gilded-gnosis` @ `e2666d9a` (remote `gg` →
  `local-inference-lab/vllm`).

## Immediate next steps (from PLAN.md §5)

Phase 0 — measurement only, no vLLM changes:
- 0a: stability analysis (Kendall τ of top-N experts across windows) on the
  existing 7.3M-token layer-78 trace — pandas, zero GPU. **This is the cheapest
  decisive experiment.**
- 0d: the go/no-go — two quants at identical bpw, stock vs workload-blended
  calibration.
- 0f(vi): comment on vLLM RFC #49702 (EPLB platform backend) while the feedback
  window is open.
