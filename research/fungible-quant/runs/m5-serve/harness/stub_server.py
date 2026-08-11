#!/usr/bin/env python3
"""A fake OpenAI-compatible endpoint, for testing the harness with no GPU.

The M5 harness has four moving parts (decode bench, two evals, saturation
driver) that must be known-good *before* the real serve exists, because the
serve run is expensive and we only get clean shots at it. This stub answers
``/v1/models``, ``/v1/chat/completions`` (streaming and not), ``/v1/completions``
(with ``echo`` + ``logprobs`` so lm-eval loglikelihood tasks work) and
``/metrics``, at a configurable fake token rate.

It is a plumbing check only: it proves the runners parse, connect, stream,
score and write output. It says nothing about the model.

    ./stub_server.py --port 8000 --tok-per-s 200
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

STATE = {"model": "GLM-5.2", "tok_per_s": 200.0, "n_tokens": 40,
         "running": 0, "gen_tokens": 0}
LOCK = threading.Lock()

# Enough of an answer that both eval scorers find something: a final-number
# line for gsm8k and an "The answer is (X)" line for the multiple-choice tasks.
ANSWER_TAIL = "\nThe answer is (A).\nAnswer: A\n42"


def fake_body(n: int) -> list[str]:
    words = ("step reasoning consider therefore however compute value result "
             "because next then finally").split()
    rng = random.Random(1234)
    toks = [" " + rng.choice(words) for _ in range(max(1, n))]
    toks.append(ANSWER_TAIL)
    return toks


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):  # noqa: D102 - silence per-request logging
        pass

    def _chunk(self, data: bytes) -> None:
        """One HTTP/1.1 chunked-transfer frame; empty data closes the body."""
        self.wfile.write(f"{len(data):X}\r\n".encode() + data + b"\r\n")
        self.wfile.flush()

    def _send(self, code: int, body: bytes, ctype="application/json") -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path.rstrip("/") in ("/v1/models", "/models"):
            self._send(200, json.dumps({
                "object": "list",
                "data": [{"id": STATE["model"], "object": "model",
                          "owned_by": "stub",
                          "max_model_len": 131072}]}).encode())
        elif self.path == "/metrics":
            with LOCK:
                text = (f'vllm:num_requests_running{{model_name="{STATE["model"]}"}} '
                        f'{STATE["running"]}\n'
                        f'vllm:num_requests_waiting{{model_name="{STATE["model"]}"}} 0\n'
                        f'vllm:generation_tokens_total{{model_name="{STATE["model"]}"}} '
                        f'{STATE["gen_tokens"]}\n')
            self._send(200, text.encode(), "text/plain")
        elif self.path.rstrip("/") in ("/health", "/v1/health"):
            self._send(200, b"")
        else:
            self._send(404, b'{"error":"not found"}')

    def do_POST(self) -> None:  # noqa: N802
        n = int(self.headers.get("Content-Length") or 0)
        try:
            req = json.loads(self.rfile.read(n) or b"{}")
        except json.JSONDecodeError:
            self._send(400, b'{"error":"bad json"}')
            return
        path = self.path.rstrip("/")
        if path.endswith("/chat/completions"):
            self._chat(req)
        elif path.endswith("/completions"):
            self._completions(req)
        else:
            self._send(404, b'{"error":"not found"}')

    # ---------------------------------------------------------------- chat
    def _chat(self, req: dict) -> None:
        want = min(int(req.get("max_tokens") or STATE["n_tokens"]),
                   STATE["n_tokens"])
        toks = fake_body(want)
        delay = 1.0 / STATE["tok_per_s"] if STATE["tok_per_s"] > 0 else 0.0
        with LOCK:
            STATE["running"] += 1
        try:
            if not req.get("stream"):
                time.sleep(delay * len(toks))
                with LOCK:
                    STATE["gen_tokens"] += len(toks)
                self._send(200, json.dumps({
                    "id": "stub", "object": "chat.completion",
                    "model": req.get("model", STATE["model"]),
                    "choices": [{"index": 0, "finish_reason": "stop",
                                 "message": {"role": "assistant",
                                             "content": "".join(toks)}}],
                    "usage": {"prompt_tokens": 100,
                              "completion_tokens": len(toks),
                              "total_tokens": 100 + len(toks)},
                }).encode())
                return
            cont = bool((req.get("stream_options") or {})
                        .get("continuous_usage_stats"))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            # HTTP/1.1 with no Content-Length needs explicit chunked framing,
            # or the client blocks until the connection closes and every
            # streaming metric (TTFT, ITL, completion) is wrong.
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            for i, tok in enumerate(toks, 1):
                time.sleep(delay)
                chunk = {"id": "stub", "object": "chat.completion.chunk",
                         "model": req.get("model", STATE["model"]),
                         "choices": [{"index": 0, "delta": {"content": tok}}]}
                if cont:
                    chunk["usage"] = {"prompt_tokens": 100,
                                      "completion_tokens": i,
                                      "total_tokens": 100 + i}
                self._chunk(f"data: {json.dumps(chunk)}\n\n".encode())
                with LOCK:
                    STATE["gen_tokens"] += 1
            self._chunk(b"data: [DONE]\n\n")
            self._chunk(b"")
        finally:
            with LOCK:
                STATE["running"] -= 1

    # --------------------------------------------------------- completions
    def _completions(self, req: dict) -> None:
        prompts = req.get("prompt")
        if isinstance(prompts, (str, list)) and not isinstance(prompts, list):
            prompts = [prompts]
        if not isinstance(prompts, list) or (prompts and isinstance(prompts[0], int)):
            prompts = [prompts]
        choices = []
        for idx, p in enumerate(prompts):
            if req.get("echo"):
                # lm-eval loglikelihood: needs token_logprobs and top_logprobs
                # spanning the echoed context plus the continuation.
                ntok = len(p) if isinstance(p, list) else max(2, len(str(p)) // 4)
                lp = [None] + [-math.log(2)] * (ntok - 1)
                choices.append({
                    "index": idx, "text": "", "finish_reason": "length",
                    "logprobs": {"token_logprobs": lp,
                                 "tokens": ["x"] * ntok,
                                 "top_logprobs": [{}] + [{"x": -math.log(2)}]
                                                 * (ntok - 1),
                                 "text_offset": list(range(ntok))}})
            else:
                want = min(int(req.get("max_tokens") or STATE["n_tokens"]),
                           STATE["n_tokens"])
                choices.append({"index": idx, "finish_reason": "stop",
                                "text": "".join(fake_body(want))})
        self._send(200, json.dumps({
            "id": "stub", "object": "text_completion",
            "model": req.get("model", STATE["model"]), "choices": choices,
            "usage": {"prompt_tokens": 100, "completion_tokens": 10,
                      "total_tokens": 110}}).encode())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--model", default="GLM-5.2")
    ap.add_argument("--tok-per-s", type=float, default=200.0,
                    help="fake per-request decode rate")
    ap.add_argument("--n-tokens", type=int, default=40,
                    help="fake completion length in tokens")
    args = ap.parse_args()
    STATE.update(model=args.model, tok_per_s=args.tok_per_s,
                 n_tokens=args.n_tokens)
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    srv.daemon_threads = True
    print(f"stub OpenAI endpoint on http://{args.host}:{args.port} "
          f"model={args.model} rate={args.tok_per_s} tok/s", flush=True)
    srv.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
