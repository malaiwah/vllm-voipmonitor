#!/usr/bin/env python3
"""T1 — stats-collector graph-freeze test (03-testing-validation.md).

The load-bearing assumption of the whole fungible-quant loop: a capture_fn
bound to BaseRouter before CUDA-graph capture keeps firing (graph-safely)
during replay, with counts matching eager mode.

Rig: malaiwah/GLM-5.2-SIQ-Fruit-Instruct-bf16 — real-weight
GlmMoeDsaForCausalLM proxy (13 layers, 10 MoE, 256 routed experts, top-8),
1 GPU, in-process engine. The FQ counting hook is chained at the exact
production binding site (GPUModelRunner._bind_routed_experts_capturer) so
it lands BEFORE graph capture, mirroring where exl3_fungible/stats.py will
sit.

Pass criteria:
  1. graphed counts == eager counts for the same prompts (exact),
  2. counts grow monotonically across generations in graphed mode,
  3. per-layer totals == (prompt+generated tokens) * top_k.

Usage: t1_test.py <fruit_model_dir> <workdir>
"""
import json
import sys
from pathlib import Path

MODEL = Path(sys.argv[1])
WORK = Path(sys.argv[2])

PROMPTS = ["The capital of France is", "def fibonacci(n):",
           "Once upon a time", "import torch"]
DECODE_TOKENS = 32
TOP_K = json.load(open(MODEL / "config.json"))["num_experts_per_tok"]

# FQ prototype state: layer_id -> int32 count buffer [E]
COUNT_BUFS: dict[int, "torch.Tensor"] = {}


def install_fq_hook() -> None:
    """Chain a graph-safe counting fn at the production binding site."""
    import torch
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    from vllm.model_executor.layers.fused_moe.layer import MoERunner
    from vllm.model_executor.layers.fused_moe.router.base_router import BaseRouter

    orig = GPUModelRunner._bind_routed_experts_capturer

    def patched(self, *a, **k):
        out = orig(self, *a, **k)
        for module in self.model.modules():
            if isinstance(module, MoERunner) and isinstance(module.router, BaseRouter):
                layer_id = module.layer_id
                n_experts = module.router.global_num_experts
                buf = torch.zeros(n_experts, dtype=torch.int32, device=self.device)
                COUNT_BUFS[layer_id] = buf
                prev = module.router.capture_fn

                def fq_fn(topk_ids, _buf=buf, _prev=prev):
                    # graph-safe: pure tensor ops, no host reads, no allocation
                    _buf.scatter_add_(
                        0, topk_ids.flatten().to(torch.int64),
                        torch.ones_like(topk_ids.flatten(), dtype=torch.int32))
                    if _prev is not None:
                        _prev(topk_ids)

                module.router.set_capture_fn(fq_fn)
        print(f"[T1] FQ hook chained on {len(COUNT_BUFS)} MoE layers", flush=True)
        return out

    GPUModelRunner._bind_routed_experts_capturer = patched


def run_mode(eager: bool) -> dict:
    import torch
    from vllm import LLM, SamplingParams

    COUNT_BUFS.clear()
    llm = LLM(model=str(MODEL), dtype="bfloat16",
              enforce_eager=eager, gpu_memory_utilization=0.45,
              max_model_len=512, max_num_seqs=4, trust_remote_code=True,
              disable_log_stats=True)
    for buf in COUNT_BUFS.values():
        buf.zero_()  # discard profiling/warmup/capture traffic
    sp = SamplingParams(temperature=0.0, max_tokens=DECODE_TOKENS)
    totals_progress = []
    outs = llm.generate(PROMPTS[:2], sp)
    totals_progress.append(sum(int(b.sum()) for b in COUNT_BUFS.values()))
    outs += llm.generate(PROMPTS[2:], sp)
    totals_progress.append(sum(int(b.sum()) for b in COUNT_BUFS.values()))

    prompt_tokens = sum(len(o.prompt_token_ids) for o in outs)
    gen_tokens = sum(len(o.outputs[0].token_ids) for o in outs)
    per_layer = {lid: [int(x) for x in buf.cpu().tolist()]
                 for lid, buf in COUNT_BUFS.items()}
    result = {
        "mode": "eager" if eager else "graphed",
        "prompt_tokens": prompt_tokens,
        "gen_tokens": gen_tokens,
        "totals_progress": totals_progress,
        "per_layer_sum": {lid: sum(v) for lid, v in per_layer.items()},
        "per_layer": per_layer,
        "n_moe_layers": len(COUNT_BUFS),
    }
    del llm
    torch.cuda.empty_cache()
    return result


def main():
    install_fq_hook()

    graphed = run_mode(eager=False)
    eager = run_mode(eager=True)

    report = {"graphed": graphed, "eager": eager, "checks": {}}
    c = report["checks"]
    c["moe_layers_found"] = graphed["n_moe_layers"] > 0
    c["counts_nonzero_graphed"] = sum(graphed["per_layer_sum"].values()) > 0
    c["monotonic_growth"] = (graphed["totals_progress"][1]
                             > graphed["totals_progress"][0] > 0)
    c["eager_equals_graphed"] = graphed["per_layer"] == eager["per_layer"]
    expected = (graphed["prompt_tokens"] + graphed["gen_tokens"]) * TOP_K
    c["per_layer_total_expected"] = {
        "expected_tokens_x_topk": expected,
        "per_layer_sums": graphed["per_layer_sum"],
        "all_match": all(v == expected for v in graphed["per_layer_sum"].values()),
    }
    c["T1_PASS"] = (c["moe_layers_found"] and c["counts_nonzero_graphed"]
                    and c["monotonic_growth"] and c["eager_equals_graphed"])
    WORK.mkdir(parents=True, exist_ok=True)
    (WORK / "t1_results.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in c.items()}, indent=1), flush=True)
    print("T1", "PASS" if c["T1_PASS"] else "FAIL", flush=True)
    sys.exit(0 if c["T1_PASS"] else 1)


if __name__ == "__main__":
    main()
