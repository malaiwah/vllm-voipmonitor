# runs/ — the evidence index

Every claim in `../implementation/` that says "measured" points here. One
session, 2026-08-10, on an 8× RTX PRO 6000 (SM120) box (224 cores, 1.5 TB
RAM, 3.0 TB persistent `/home`), inside a managed container with no nested
runtime and no sudo — the GG stack runs from an extracted rootfs via
`gg-env/gg-run.sh`.

**Start here if you have 60 seconds:** the four rows in bold below are the
load-bearing ones — the collector really records inside CUDA graphs, a
mixed-K checkpoint really boots, a live serve really swaps its allocation in
0.4 s, and the atomic map mutation really survives graph replay.

**Start here if you have 10 minutes:** `../implementation/14-build-findings.md`
consolidates everything these reports taught us that the pre-build spec
(docs 00–13) got wrong.

## The evidence

| Run dir | Question it answers | Verdict | Headline number |
|---|---|---|---|
| `drift-check/` | Are we building against the sources we audited? | **No drift that matters** | GG HEAD **==** the audited `e2666d9a`; b12x +3 commits, all dense-GEMM-only; exl3 HEAD == the r33 pin |
| `pre-m4-checks/` | Do the 4 kernel assumptions the swap engine rests on actually hold? | **4/4 + occupancy CLOSED** | rotations **COPY** (⇒ write the *combined* tables); occupancy<capacity **7/7 bitwise-equal**, leakage control 1024/1024 NaN |
| `encode-bench/` | What does one expert cost to encode? | **3× cheaper than planned** | **2.55 s** K3 / **2.48 s** K4 per expert; K3≈K4 within 3 %; ~71 experts/h at a 5 % budget |
| **`t1-graph-freeze/`** | **Does a capture_fn keep recording inside CUDA-graph replay?** | **PASS** | counts grow 10800→21520 in replay; 10/10 layers agree exactly; constant **16-routing** boundary offset at two run lengths. Key fact: the binding site is gated on `enable_return_routed_experts` — **3 prior runs were hollow** |
| `m0-seed/` | Can an existing community quant become attested segments? | **Yes, verbatim** | 76 layers repacked, **278.6 GB** published, ed25519-signed `repack-of` per layer |
| `m0-assemble/` | Do segments + a recipe rebuild the original bytes? | **Byte-identical** | **79/79 shards sha256-identical** to `brandonmusic@9297b9f1` at full-model scale |
| `serve-baseline/report.md` | Does full GLM-5.2 K3 serve on this box with no container runtime? | **PASS** | boots TP4/DCP4 from the extracted r33 rootfs; **37.7 tok/s** single-request MTP0 |
| **`serve-baseline/fruit-mixed-report.md`** | **Does a true mixed-K checkpoint boot and generate?** | **PASS (M0 gate)** | per-layer K4 counts == policy exactly (42…152); **501.6 vs 503.1 tok/s** vs pure K3 ⇒ mixed-tier execution costs **~0 %**. Documents the 3-part loader metadata contract and the `fp8_ds_mla` KV trap |
| `probe-reference/` | What is the quality baseline to measure swaps against? | captured | 32 held-out prompts, teacher-forced, mean logprob **−1.7589** on the K3 serve |
| `0c-campaign/report.md` | How much does one more bit buy, and per expert or per layer? | **Per-expert wins; K2-abort does NOT fire** | ε ladder **0.0903 / 0.0231 / 0.0060 / 0.0016** = clean **~3.8× per bit**; Δε CV 0.047 but benefit **Gini 0.48** ⇒ routing mass is the signal |
| `0c-campaign/PIVOT.md` | Why can't we use stock exllamav3 to encode? | blocked, rerouted | `convert_model` asserts `Unknown architecture GlmMoeDsaForCausalLM`; canonical encoder is the sha-pinned `encode_tr3_v31.py` (`e9a85a47…`) bundle inside the K3 repo |
| `0c-campaign/capture-stream-report.md` | Can we capture a 1.5 TB BF16 model's activations on one GPU? | **PASS, bit-exact** | layer-major streaming, ~35 GB peak; **100.0000 %** ids/x match vs the sealed reference on all 10 layers, sha256-equal. Finds the CUDA-`pow` rotary 1-ulp trap and batch-shape row-instability |
| `0c-campaign/quant-342-layout-report.md` | What is inside the 3.42bpw community quant, and can we reuse it? | **Salvageable** | `shared_h_v1` confirmed **from bytes**; 36+12 tensors/layer; expansion to `per_expert_v1` is exact (+147,456 B/expert); **8,042 K4 fragments** reusable; 351.6 GB audited with **zero** full-shard downloads |
| `0c-campaign/MULTI-K-PLAN.md` | How do we get K2/K3/K4/K5 under a fixed 3 TB disk? | streaming ring | capture window → encode K2..K5 → publish → delete; peak transient ~150 GB; complement matrix drops the encode load to ~660/1024 experts per layer |
| `0c-campaign/ATTESTATION-V2.md` | What must an attestation carry to be re-verifiable? | design | DSSE/in-toto envelope; **determinism scope is mandatory** (stack-scoped reproducibility is a measured fact); predicates grow to 5 rungs |
| **`m3-reload/`** | **Can a running serve swap its whole mixed-K allocation?** | **PASS — the real M3** | **0.466 s / 0.410 s** total stall, **0 request drops**, post-reload logits **bit-identical** to a fresh boot (max \|Δlogprob\| = **0.0**, 356 tokens, twice). Restart floor for comparison: 88.0 s |
| **`m4-swap/`** | **Are the tier maps read as data, or baked into the graph?** | **T3 PASS — read as data** | mutate map contents under a captured graph → replay `torch.equal` to a fresh build; T4 row-write **PASS ×3** + bitwise rollback; window **0.061 ms**/pair (fixed overhead, toy payload) |
| **`m5-serve/assembly-report.md`** | **Can we build the two checkpoints the serve proof needs, on a full disk?** | **PASS — both** | pure K3 **81/81 shards bit-exact** vs `brandonmusic@9297b9f1`, **0 GB** physical (100 % shared extents); mixed K3/K5 differs on **exactly the 12** K5 layers, 768 K5 slots, **56.2 GB** physical. Per-rank TP4 **73.65 → 75.33 GiB** (77.0 → 78.8 % of a 95.6 GiB card). Confirms the 0c finding: `--reflink` shares **nothing** per-region, but whole-file clones share everything |
| `loader-v2/report.md` | Can we boot mixed-K straight from segments, skipping assembly? | **PASS** | greedy output **token-identical** to the assembled serve; streaming costs **+1.8 s**, tok/s at parity; **zero** per-policy disk (vs a 3.7 GB copy). ~30 s of the boot gap is a compile-cache miss, not loader cost |
| `loader-v2/trust-and-lazy-encode.md` | How does an operator choose who to trust, and what if a K is missing? | design + 77 CPU tests | `VLLM_FQ_TRUST_SIGNERS`/`_PREDICATES` armed only with an anchor; countersignatures accept; `VLLM_FQ_K_FALLBACK` substitutes a **marked** K and queues the encode — boot never blocks |

## Supporting / infrastructure dirs

| Dir | What |
|---|---|
| `gg-env/` | `gg-run.sh` — run anything inside the extracted r33 rootfs (validated: torch 2.12.0+cu132, vllm r33, b12x 1.1.0, cutlass DSL 4.6.0, 8× SM120). Every GPU number in this tree came through it |
| `dl-glm52-orig/` | `zai-org/GLM-5.2` @ `b4734de4` — 1.51 TB, 283 files verified |
| `dl-glm52-k3/` | `brandonmusic/GLM-5.2-EXL3-TR3-3.0bpw` @ `9297b9f1` — 295 GB, 82 safetensors, all blobs verified |
| `dl-gg-images/` | runtime-free ghcr puller + whiteout-aware rootfs extraction (r33) |
| `health/` | liveness-based job sweep (retires "stale tail" illusions in long-running logs) |

## Reading order for a newcomer

1. `t1-graph-freeze/report.md` — the assumption the whole loop rests on.
2. `serve-baseline/fruit-mixed-report.md` — what a mixed-K artifact *is*.
3. `m3-reload/report.md` — the loop closing on a live server.
4. `m4-swap/report.md` — why the fast path is not blocked.
5. `0c-campaign/report.md` — why per-expert allocation is worth doing.
6. `../implementation/14-build-findings.md` — everything the above changed
   about the spec.

## Caveats that apply to the whole table

- Every fidelity and quality gate above runs on a **proxy**: the SIQ-Fruit
  model (5.04B, 10 MoE layers × 256 experts, same expert cardinality as
  GLM-5.2) or, for T3/T4, a toy layer (E=32, H=I=128). The GLM-5.2 leg has
  capture (windows 1–2) and K2/K5 encodes for layers 3–10 only.
- Not run: T5 (torn-update injection), T6 (cross-rank soak), T7 (24 h
  soak), T8 as specified (kill -9 rehydration), T9 (quality ladder), M1 at
  TP4, and the M4 engine against a live layer.

## Artifacts that live outside this tree

- **Tools**: `../tools/` — `fq_repack`, `fq_assemble`, `fq_eps`, `fq_probe`,
  `fq_prime`, `fq_reload`, `oci_unpack`, `fruit-encoder/`.
- **Runtime code**: gg-vllm branch `fq/m1-stats-collector` (base
  `e2666d9a`) — `exl3_fungible/{stats,fragments,progressive,
  progressive_loader,swap,lazy_encode}.py`; key commits `0d6d54196`
  (loader v2), `a16c87f73` (swap engine), `b69feebca` (trust + lazy encode).
- **Public tools repo**: <https://github.com/malaiwah/progressive-tensors>.
- **Public segment repo**: `hf.co/malaiwah/GLM-5.2-EXL3-FQ-segments`
  (K3 base for layers 3–78, plus window-1 K2/K5 for layers 3–10). Model
  card copy: `m0-seed/hf-model-card.md`. **Access note (2026-08-10): the
  repo still answers 401 — it has not been flipped public yet**, so the
  quickstart in the public README cannot succeed for outside readers until
  it is.
- **Local big dirs** (not in git): `/home/mbelleau/fq-segments/`,
  `/home/mbelleau/glm52-segments/`, `/home/mbelleau/glm52-capture/`,
  `/home/mbelleau/fq-0c/`, `/home/mbelleau/rootfs/`, `/home/mbelleau/images/`,
  `/home/mbelleau/src/{gg-vllm,b12x,exllamav3}`.
