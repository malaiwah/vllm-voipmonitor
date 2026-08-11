# M5 serve harness — benchmark + evaluation tooling

Everything needed to measure the Progressive Tensors serve once it is up.
Prepared and rehearsed **without touching a GPU**: every tool below was run
end-to-end against `stub_server.py`, a fake OpenAI endpoint in this directory,
so the first time any of it meets the real serve is not also the first time it
runs.

Target endpoint: `http://127.0.0.1:8000`, served model name `GLM-5.2`
(`../serve-glm52.sh`, TP4 on GPUs 0-3).

## Constraints inherited from `../serve-glm52.sh`

These set every default here, and getting them wrong wastes a serve window:

| Serve setting | Default | Consequence for the harness |
|---|---|---|
| `--max-model-len` (`FQ_MAXLEN`) | 8192 | vLLM **rejects** any request with `prompt + max_tokens > 8192`. Bench contexts and eval `MAX_GEN` are sized under it. |
| `--max-num-seqs` (`FQ_MAXSEQS`) | 32 | Concurrency above 32 measures queueing, not throughput. |
| `--served-model-name` | `GLM-5.2` | Must match `MODEL=` exactly or every request 404s. |
| `VLLM_FQ_INTERVAL_STEPS` | 200 | At ~38 engine steps/s (from the 37.7 tok/s single-stream baseline) a swap decision point arrives every ~5 s, so even a 10-minute saturation run covers many. |

GLM-5.2 is a **thinking** model. If the serve is booted without a reasoning
parser, `<think>` text lands in `message.content` and the eval scorers see it.
The `flexible-extract` filter (last match wins) survives that; `strict-match`
may under-report. Both are reported — read `flexible-extract` as the headline
and treat a large gap between the two as a formatting artifact, not damage.

## Install state (already done)

| Path | What |
|---|---|
| `/home/mbelleau/bench/llm-inference-bench` | local-inference-lab decode bench, commit `86cf05c`, tool version `0.4.29` |
| `/home/mbelleau/venvs/bench` | `httpx 0.28.1`, `rich 15.0.0`, `psutil 7.2.2` — runs the decode bench and `saturate.py` |
| `/home/mbelleau/venvs/lmeval` | `lm_eval 0.4.12`, `datasets 5.0.1`, `aiohttp 3.14.3` |

Datasets are pre-downloaded and were verified to load with the network off
(`HF_HUB_OFFLINE=1`), so no eval can stall on a fetch mid-run:

| Dataset | Items | Cache |
|---|---|---|
| `openai/gsm8k` main/test | 1319 | `~/.cache/huggingface/datasets/openai___gsm8k` |
| `Idavidrein/gpqa` gpqa_diamond/train | 198 | `~/.cache/huggingface/datasets/Idavidrein___gpqa` |
| bench `gsm8k_test.jsonl` | 1319 | shipped in the repo's `data/` |
| bench `mmlu_pro_1000.jsonl` | 1000 | shipped in the repo's `data/` |
| bench `gpqa_diamond.jsonl` | 198 | `~/.cache/llm_decode_bench/datasets/` (sha256-pinned, built from the official password-protected zip; **must not be committed**) |

---

## 0. Preflight — `preflight.sh`

Four checks worth 30 seconds before an hour of GPU time.

```bash
./preflight.sh http://127.0.0.1:8000 GLM-5.2
```

Verifies `/v1/models` answers, the model name matches, `/metrics` is exposed
(needed by `../swap_evidence.py scrape`), and — the one that silently ruins
throughput numbers — that streamed chunks carry a cumulative
`completion_tokens`. Without it every tool falls back to counting chunks,
which is an estimate, not a token count. Exit non-zero on any failure.

## 1. Decode throughput — `decode_bench.sh`

Wraps **[local-inference-lab/llm-inference-bench](https://github.com/local-inference-lab/llm-inference-bench)**
`llm_decode_bench.py`. This is the operator-requested tool and the one the
rtx6kpro wiki numbers (`models/glm5.2_v20.md`) are produced with, so results
are directly comparable to the serve baseline.

**Why this one:** it was the only decode/throughput benchmark in the org — the
other 13 repos are engines (`vllm`, `sglang`, `LMCache`), images
(`blackwell-llm-docker`), NCCL forks, a quant toolkit, `qsrt`, `b12x`,
`llmconduit`, `fanpilot`, and the `rtx6kpro` wiki itself. Its description is
literal: "LLM inference decode throughput benchmark… Measures token generation
speed across concurrency levels and context lengths."

```bash
./decode_bench.sh http://127.0.0.1:8000 \
    ../evidence/decode/decode_live.json
```

**Measures**, per (concurrency × context) cell: Sustained Decode aggregate
tok/s over a `--duration` window with the concurrency held saturated by
restarting streams as they finish; per-request tok/s; TTFT, TTST, ITL and
request latency at avg/p50/p90/p99; prefill tok/s as `prompt_tokens / TTFT`
from the scout request for each non-zero context; and, when `/metrics` is up,
Prometheus validation counters plus effective concurrency (`(X/Y)*` when the
server cannot actually run all requested requests at once).

Aggregate tok/s comes from the stream's cumulative `completion_tokens`, which
is exact; `results[].aggregate_source` records which source was used
(`openai_continuous_usage` in the rehearsal).

**Runtime:** default matrix is 5 concurrencies × 2 contexts = 10 cells × 30 s
measured = 5 min of measurement; with per-cell readiness warmup and scout
prefill, budget **10-20 min wall**.

**Output:** one JSON file. Top level `metadata`, `startup_diagnostics`,
`nvidia_p2p_override`, `hardware_run_summary`, `event_log`, `prefill`,
`results`, `summary_table`, `burst_results`, `burst_summary_table`,
`methodology`. Each `results[]` entry carries 66 fields; the ones that matter:
`concurrency`, `context_tokens`, `aggregate_tps`, `aggregate_source`,
`per_request_avg_tps`, `ttft_{avg,p50,p90,p99}`,
`inter_token_latency_{avg,p50,p90,p99}`, `request_latency_*`,
`output_seq_len_*`, `effective_concurrency`, `client_output_tokens`,
`server_output_tokens`, `num_errors`, `measurement_seconds`.

Add `--run-burst` for a finite-burst (Burst / E2E) section as well; it roughly
doubles the runtime, which is why it is off by default.

## 2. Quality evals

Both are prepared. **Run GSM8K-250 first** — see the runtime note below, it is
the cheap one.

### Runtime model

```
wall_seconds ≈ items × mean_output_tokens / aggregate_tok_s
```

Grounded on `../serve-baseline/report.md`: **37.7 tok/s single-stream** on this
box (GLM-5.2 K3 3.0bpw, TP4/DCP4, PCIe fallback). Batching amortizes expert
reads, so aggregate at C=16 lands somewhere in **100-250 tok/s** — step 1 gives
the real figure, and these estimates should be recomputed from it.

| Eval | Items | Mean output (thinking model) | @100 tok/s | @150 tok/s | @250 tok/s |
|---|---|---|---|---|---|
| GSM8K subsample | 250 | ~800 | 33 min | 22 min | 13 min |
| GSM8K full | 1319 | ~800 | 2 h 55 m | 1 h 57 m | 1 h 10 m |
| **GPQA Diamond** | 198 | ~4000 | 2 h 12 m | 1 h 28 m | 53 min |

**This inverts the usual assumption.** GPQA Diamond looks affordable because it
is only 198 items, but they are graduate-level science questions and a
reasoning model spends thousands of tokens per item; GSM8K is grade-school
arithmetic with short traces. Cost is driven by *tokens*, not item count, so
the 250-item GSM8K subsample is roughly **4x cheaper** than GPQA Diamond
despite having more items. Budget accordingly: GSM8K-250 as the routine gate,
GPQA Diamond only when there is a spare hour or two.

### GSM8K — `eval_gsm8k.sh`

```bash
./eval_gsm8k.sh http://127.0.0.1:8000 ../evidence/eval/gsm8k

# full 1319-item set (~2-3 h)
ITEMS=0 ./eval_gsm8k.sh http://127.0.0.1:8000 ../evidence/eval/gsm8k-full

# 8-shot CoT instead of 0-shot CoT
TASK=gsm8k_cot ./eval_gsm8k.sh http://127.0.0.1:8000 ../evidence/eval/gsm8k-8shot
```

Task `gsm8k_cot_zeroshot` (0-shot CoT, `generate_until`, exact-match on the
final number, `strict-match` + `flexible-extract` filters), via lm-eval's
`local-chat-completions` against `/v1/chat/completions`. No model surgery, no
tokenizer download — the server applies the chat template.

**Subsampling is real and is labelled.** `ITEMS` defaults to **250 of 1319**,
picked by `subsample.py` with **seed 1234**, and the runner writes
`SUBSAMPLE.txt` plus `gsm8k_subsample_250_seed1234.json` into the output dir so
a subsampled result can never be mistaken for a full-set number. This uses
lm-eval's `--samples` (explicit indices), *not* `--limit`: `--limit` takes the
first N documents and the GSM8K test split is not shuffled, so a prefix is a
biased sample. 250 items gives roughly ±5 pp at 95% on an accuracy near 90% —
enough to catch a broken serve, not enough to resolve a 1-2 pp quant delta.
For that, `ITEMS=0`, or `BACKEND=bench` with `--compare-baseline` for paired
McNemar statistics.

### GPQA Diamond — `eval_gpqa.sh`

```bash
./eval_gpqa.sh http://127.0.0.1:8000 ../evidence/eval/gpqa
```

Task `gpqa_diamond_cot_zeroshot`: all **198** items, 4 options, generative CoT,
exact-match on the answer letter. Always the full set — at 198 items,
subsampling would only cost statistical power that is already coarse.

Generative CoT is used rather than the `multiple_choice` variants
(`gpqa_diamond_zeroshot`, `gpqa_diamond_n_shot`) on purpose: those score by
loglikelihood, which needs `/v1/completions` with `echo=true` + `logprobs`, and
scoring a reasoning model by the logprob of a bare letter measures the wrong
thing.

### `BACKEND=bench` — the same two evals through the decode bench

Both runners accept `BACKEND=bench` to run the identical item sets through
`llm_decode_bench.py`'s pinned dataset profiles instead of lm-eval:

```bash
BACKEND=bench ./eval_gsm8k.sh http://127.0.0.1:8000 ../evidence/eval/gsm8k-bench
BACKEND=bench ./eval_gpqa.sh  http://127.0.0.1:8000 ../evidence/eval/gpqa-bench
```

Same datasets (sha256-pinned), but it adds what lm-eval does not: Wilson 95%
intervals, per-category accuracy, completion-token percentiles, an explicit
TRUNCATED-vs-unparseable split, and `--compare-baseline` paired A/B with exact
McNemar p-values and per-item flip lists. For an FQ-on vs FQ-off comparison
this is the stronger instrument; for a number comparable to published
leaderboards, use lm-eval. Its `--profile-runs N` subsample is a deterministic
evenly-spread slice (verified: 6 items of GSM8K came back as
`gsm8k-0000, 0219, 0439, 0659, 0879, 1099`).

### Eval output schema

lm-eval writes into `<output_dir>/<model_name>/`:

- `results_<timestamp>.json` — `results.<task>` holds
  `exact_match,strict-match`, `exact_match,flexible-extract` and their
  `_stderr` twins; `n-samples.<task>` gives `{"original": 1319, "effective":
  250}`, which is the machine-readable proof of subsampling. Also
  `configs`, `versions`, `config` (full model args), `git_hash`,
  `total_evaluation_time_seconds`.
- `samples_<task>_<timestamp>.jsonl` — one row per item with `doc_id`, `doc`,
  `target`, `arguments`, `resps`, `filtered_resps`, `exact_match`, `metrics`,
  `doc_hash`, `prompt_hash`.

`BACKEND=bench` writes a single JSON with per-item results under the
completion-stats structure plus a `comparison` block when `--compare-baseline`
is passed.

## 3. Saturation load — `saturate.py`

Sustained concurrent load with **per-interval** machine-readable output, so
throughput can be read *during* live expert swaps rather than averaged over
them.

```bash
/home/mbelleau/venvs/bench/bin/python saturate.py \
    --base-url http://127.0.0.1:8000 --model GLM-5.2 \
    --concurrency 32 --duration 1800 --interval 5 --warmup 30 \
    --max-tokens 512 --metrics --echo \
    --out ../evidence/saturation/sat.jsonl
```

**Why not wrap something:** neither existing tool emits a timeline.
`llm_decode_bench.py` reports one aggregate per matrix cell and the inside of a
cell is opaque. vLLM's `benchmarks/benchmark_serving.py` is now a deprecation
shim (verified — it prints "moved to the vLLM CLI" and exits 1); the real tool
is `vllm bench serve` (`vllm/benchmarks/serve.py` in the GG tree at
`/home/mbelleau/src/gg-vllm`), which also reports a single aggregate per run
and needs torch + the GG container to import. `saturate.py` needs only `httpx`
and produces the time series the swap overlay requires.

**Token accounting** requests `stream_options.continuous_usage_stats`, so each
interval's token count is the sum of exact server-reported deltas. If the
server does not honour it, the script counts streamed chunks instead and says
so in `token_source` and in the summary `note` — a chunk count is an estimate
and is labelled as one. Rehearsal cross-check: client counted 2789 tokens where
`/metrics` counted 2790.

**Overlaying swap events:** every row carries `t` as epoch seconds, the same
clock `../swap_evidence.py scrape` writes, so joining the two JSONL streams is
a numeric merge with no clock translation. Run them together:

```bash
python ../swap_evidence.py scrape --base http://127.0.0.1:8000 \
    --out ../evidence/saturation/swaps.jsonl --interval 5 --duration 1800 &
/home/mbelleau/venvs/bench/bin/python saturate.py ... --out ../evidence/saturation/sat.jsonl
```

**Output** is JSONL: one `kind:"config"` row, one `kind:"sample"` row per
interval, one `kind:"summary"` row at the end. A sample row:

```json
{"kind":"sample","t":1786408389.0964358,"iso":"2026-08-11T00:33:09Z",
 "elapsed_s":6.028,"window_s":3.004,"warmup":false,
 "window_output_tokens":1426,"window_tok_s":474.73,
 "window_completed":24,"window_failed":0,"inflight":8,
 "launched_total":47,"completed_total":39,"failed_total":0,
 "output_tokens_total":2789,"token_source":"usage",
 "ttft_ms":{"n":24,"p50":41.96,"p90":43.55,"p99":44.82},
 "itl_ms":{"p50":16.36,"p90":16.38},
 "latency_ms":{"p50":1024.84,"p90":1027.20},
 "errors":{},"server_running":8.0,"server_waiting":0.0,
 "server_gen_tokens_total":2790.0,"server_window_tok_s":475.06}
```

`server_*` fields appear only with `--metrics`. TTFT is booked into the window
where the first token actually arrived, not where the request finished — a
request that runs for minutes would otherwise report its TTFT into the wrong
window, which is precisely the window being lined up against a swap event.

**Runtime** is exactly `--duration`. Size it against
`VLLM_FQ_INTERVAL_STEPS=200`: 20-30 min at concurrency 16-32 covers many swap
decision points with margin.

## 4. Rehearsal without a GPU — `stub_server.py`

A fake OpenAI-compatible endpoint: `/v1/models`, `/v1/chat/completions`
(streaming with correct chunked framing and continuous usage, and non-
streaming), `/v1/completions` (with `echo`+`logprobs` for loglikelihood tasks),
`/metrics`, `/health`.

```bash
/home/mbelleau/venvs/bench/bin/python stub_server.py --port 8971 \
    --tok-per-s 60 --n-tokens 60
```

It validates plumbing only — connection, streaming, concurrency, parsing,
scoring, output files — and says nothing about the model. Everything in this
README was verified against it:

| Tool | Rehearsal result |
|---|---|
| `preflight.sh` | passes; correctly exits 1 on a wrong model name |
| `decode_bench.sh` | C=1 → 59.7 tok/s, C=4 → 238.0 tok/s against a 60 tok/s/stream stub |
| `eval_gsm8k.sh` | 12-item seeded subsample scored, both filters, samples JSONL written |
| `eval_gpqa.sh` | 198 items loaded, `--limit` honoured, both filters scored |
| `BACKEND=bench` | GSM8K profile ran, Wilson CI + per-item table + JSON written |
| `saturate.py` | 8 × 60 tok/s → 471.6 tok/s measured; client/server token counts agreed to 1 |

## Order of operations on the real serve

1. `preflight.sh` — 30 s.
2. `decode_bench.sh` — 10-20 min. Gives the aggregate tok/s that makes every
   estimate below exact instead of a range.
3. `saturate.py` + `swap_evidence.py scrape` together — 20-30 min. This is the
   throughput-during-swaps evidence.
4. `eval_gsm8k.sh` (250-item subsample) — 20-35 min.
5. `eval_gpqa.sh` — 1-2 h, only if the window allows.

Steps 2-5 must not overlap: they all measure the same GPUs.
