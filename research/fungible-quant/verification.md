# Citation verification — 2026-08-10

Post-hoc verification of the citations in `PLAN.md`, performed against arxiv.org
abstracts and github.com/vllm-project/vllm public pages. Method: every arXiv ID
fetched and matched against the claimed title/topic/detail; every issue/PR
number fetched and matched against the claimed title/state/content.

Verdicts: **CONFIRMED** (ID resolves, claim supported by abstract/page) ·
**PARTIAL** (resolves and matches, but a specific quoted detail is body-level
and not verifiable from the abstract/page) · **REFUTED** (wrong or mismatched).

## arXiv papers (18/18 resolve; 0 refuted; corrections listed below)

| ID | Verdict | Actual title | Note |
|---|---|---|---|
| 2511.15015 | CONFIRMED | Dynamic Expert Quantization for Scalable MoE Inference (**DynaExq**) | UConn+UCSC, Nov 2025. Router-trace hotness, budget-constrained allocation, async promote/demote, stable expert handles — all in abstract. "EMA"/"hysteresis" are body details. |
| 2506.02006 | PARTIAL | **MorphServe**: … Runtime Quantized Layer Swapping and KV Cache Resizing | Runtime quantized layer swap confirmed. Pinned-CPU staging and the **~6 ms/layer** figure are body details — verify PDF before quoting. |
| 2410.06270 | PARTIAL | Mixture Compressor for MoE LLMs (**MC-MoE**) | ICLR 2025; LP over (error, routing score, frequency) confirmed. "~1 s solve" is a body detail. |
| 2604.06515 | PARTIAL | Efficient Quantization of MoE with Theoretical Generalization Guarantees | RPI+IBM confirmed. Uses router-L2-norm **plus intra-neuron variance** (not router norms alone). Abstract says "negligible overhead"; "no GPU" and "beats activation-frequency baselines" need PDF verification. |
| 2506.13329 | CONFIRMED | **EAQuant** | Expert-aware calibration balance is an explicit contribution. |
| 2505.03804 | CONFIRMED | **MoEQuant** | EBSS + affinity-guided quantization as claimed. |
| 2410.14649 | CONFIRMED | **EvoPress** | ICML 2025; explicitly refutes additive per-layer error. |
| 2502.06786 | PARTIAL | **Matryoshka Quantization** | **Correction: ICML 2025 poster; the "oral" was ICLR 2025 SLLM *workshop*, not an ICLR main-conference oral.** |
| 2402.10517 | PARTIAL | **Any-Precision LLM** | ICML 2024 confirmed. Bit-plane layout and INF-perplexity Table 2 claim are body/table details. |
| 2505.05799 | CONFIRMED | **MxMoE** | Auto-generated mixed-precision GroupGEMM, "up to 29.4% over uniform" — in abstract. |
| 2607.02893 | CONFIRMED | **VBQ** | "Freeze into a fixed recipe and reuse without further search" — supported. |
| 2607.00908 | PARTIAL | **TASA** (alignment–diversity tradeoff) | Kendall τ ≈ 0 confirmed in abstract. The 50%/75% mixing ratios are body details. |
| 2608.04048 | PARTIAL | **RRQ** | **Correction: "Recurrent Residual Quantization"** (not "residual refinement"). Reusable 2-bit base + sequential residuals confirmed. |
| 2510.10467 | CONFIRMED | **AnyBCQ** | Binary bit-plane multi-precision; 1.2× over SOTA multi-precision. |
| 2606.04980 | CONFIRMED | **AlphaQ** | HT-SR spectra, calibration-free, budget-constrained — all in abstract. |
| 2509.02512 | CONFIRMED | **MoPEQ** | Hessian-trace per-expert sensitivity. Note: evaluated on VLMs, not text-only LLMs. |
| 2501.07139 | CONFIRMED | **FlexQuant** | Elastic quantized ensemble; framing is edge devices. |
| 2411.01433 | PARTIAL | **HOBBIT** (ID located by verifier) | Mixed-precision dynamic expert loading confirmed, up to 9.93× decode. "Cumulative gate-norm"/"LHU" terminology not in abstract — verify PDF before quoting. |

**Blanket caveats:** the five 2026-dated IDs (2604.06515, 2606.04980, 2607.00908,
2607.02893, 2608.04048) resolve and match but have no corroboration found beyond
arXiv itself. Any specific number quoted from a paper body (6 ms, 1 s, 50%/75%,
INF PPL) must be checked against the PDF before appearing in an upstream RFC.

## vLLM issues / PRs / RFCs (24/24 resolve; 0 refuted; corrections applied to PLAN.md)

Checked via rendered github.com pages (the REST API returned 403 through the proxy).

| Ref | Verdict | Actual title | State |
|---|---|---|---|
| RFC #49702 | CONFIRMED | [RFC]: Add an EPLB Platform Backend for out-of-tree accelerators (freyfwt) | open |
| RFC #48920 | CONFIRMED | [RFC]: Unify the Weight Loading Lifecycle Across Initial Load and Reload (aoshen02) | open |
| PR #48908 | CONFIRMED | [Refactor][Model Loader] Unify the weight loading lifecycle | **closed, unmerged** |
| RFC #50281 | CONFIRMED | [RFC]: Per-layer online quantization configuration (fxmarty-amd) | open |
| PR #50401 | CONFIRMED | Add per-layer online quantization configuration (**Ganeshkusalkar**, not fxmarty-amd) | open |
| PR #51285 | CONFIRMED | Targeted online quantization configuration based on user patterns (fxmarty-amd) | open |
| PR #51392 | CONFIRMED | Online quantization with partially pre-quantized checkpoints (fxmarty-amd) | open |
| RFC #38256 | CONFIRMED | [RFC]: Incremental MoE Expert Offloading — GPU Cache + Async Pipeline (e1n00r) | open |
| PR #37190 | CONFIRMED | Run MoE models exceeding VRAM via expert CPU offloading with GPU cache (~1479 additions after rebase) | open |
| #49198 | CONFIRMED | Recency-based progressive mixed-precision KV cache — lifecycle quote verified verbatim | open |
| #51567 | CONFIRMED | EPLB fails to transport E8M0 expert state; fix = storage-preserving `tensor.view(torch.uint8)` | open |
| #41955 | CONFIRMED | Mixed INT4/INT8 GPTQ MoE models crash on initialization (AssertionError in fused_marlin_moe) | open |
| #21336 | CONFIRMED | vLLM crashes with --enable-sleep-mode on Blackwell PRO 6000 — created 2025-07-21 | open |
| #48680 | CONFIRMED | sleep-mode OOM, ~4.85 GiB cumem private MemPool on SM120/NVFP4 | open |
| PR #50168 | PARTIAL | LUT-B quantization accuracy prototype (mgoin, draft). **bpw range 3.125–3.522** (not 3.516); 52.08% = baseline variant, 75.51% = best | open (draft) |
| #46611 | CONFIRMED* | 60 s shared-memory broadcast timeout. *Quantization named in vLLM's generic error message as a possible cause, not a confirmed diagnosis | open |
| RFC #32028 | CONFIRMED | [RFC]: EPLB Implementation Refactoring (ilmarkov) | open |
| #19896 / #11416 / #3203 / #296 / #2645 | CONFIRMED | EXL3 / QTIP / exl2 / ExLlama / ExLlamaV2-kernel requests — all closed as not planned (several stale-closed) | closed |
| Discussion #5802 | CONFIRMED* | RFC: FP8 Quantization Schema in vLLM update. *Real, but GitHub Discussions was **deprecated Mar 2025** in favor of discuss.vllm.ai — not an active design venue | — |

### The deadline claim

**CONFIRMED verbatim.** RFC #49702's body states the feedback window as:
**"Two weeks, through August 18, 2026."** The 2026-08-18 deadline in PLAN.md §6
is directly established by the issue text.

### Corrections applied to PLAN.md (commit history has details)

1. #48920 does **not** contain the phrase "existing model geometry" — replaced
   with its actual scope sentence ("…do not replace the base checkpoint…").
2. PR #48908 described as closed-unmerged; engagement belongs on the RFC.
3. LUT-B bpw upper bound 3.522; GSM8K figures attributed to baseline/best variants.
4. PR #50401 authorship corrected.
5. #46611 attribution softened to error-message wording.
6. #38256 "atomic swap" softened to "async H2D pipeline" (not evidenced verbatim).
7. Duplicate-check venue corrected: GitHub Discussions is deprecated; the
   unsearched venue is the **discuss.vllm.ai forum**.
