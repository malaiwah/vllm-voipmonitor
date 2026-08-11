"""Tests for the swap-evidence recorder.

The parser is the part that can silently lie: a metric name that moved between
vLLM versions, or a label regex that half-matches, would produce a plausible
but wrong timeline — exactly the kind of evidence that must not end up in a PR.
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import swap_evidence as se  # noqa: E402

EXPOSITION = """\
# HELP fq_swaps_total Fungible-quant expert tier swaps decided per MoE layer
# TYPE fq_swaps_total counter
fq_swaps_total{layer="3"} 12.0
fq_swaps_total{layer="7"} 5.0
# HELP fq_tier_occupancy Experts resident per (layer, K tier)
# TYPE fq_tier_occupancy gauge
fq_tier_occupancy{layer="3",tier="3"} 200.0
fq_tier_occupancy{layer="3",tier="5"} 56.0
fq_tier_occupancy{layer="7",tier="3"} 251.0
fq_tier_occupancy{layer="7",tier="5"} 5.0
# TYPE fq_jaccard gauge
fq_jaccard 0.9731
fq_probe_kld 0.0042
fq_policy_age_steps 81000.0
fq_rollbacks_total 0.0
# TYPE vllm:generation_tokens_total counter
vllm:generation_tokens_total{model_name="GLM-5.2"} 1234567.0
vllm:num_requests_running{model_name="GLM-5.2"} 16.0
"""


def test_parses_labelled_and_bare_samples():
    flat = se.parse_metrics(EXPOSITION)
    assert flat['fq_swaps_total{layer="3"}'] == 12.0
    assert flat["fq_jaccard"] == pytest.approx(0.9731)
    assert flat["fq_policy_age_steps"] == 81000.0


def test_comments_and_blank_lines_ignored():
    flat = se.parse_metrics("# HELP x doc\n\n# TYPE x counter\nx 1.0\n")
    assert flat == {"x": 1.0}


def test_non_numeric_value_is_skipped_not_crashed():
    flat = se.parse_metrics("broken NaNsense\ngood 2.0\n")
    assert "good" in flat and flat["good"] == 2.0


def test_fq_view_groups_by_layer_and_tier():
    view = se.fq_view(se.parse_metrics(EXPOSITION))
    assert view["swaps_by_layer"] == {"3": 12.0, "7": 5.0}
    assert view["swaps_total"] == 17.0
    assert view["tier_occupancy"]["3"] == {"3": 200.0, "5": 56.0}
    # layer 3 has 256 experts split across tiers — occupancy must be complete
    assert sum(view["tier_occupancy"]["3"].values()) == 256.0
    assert view["jaccard"] == pytest.approx(0.9731)
    assert view["rollbacks"] == 0.0


def test_fq_view_survives_a_serve_with_no_fq_metrics():
    """A baseline serve (VLLM_FQ_ENABLE=0) exports none of these."""
    view = se.fq_view(se.parse_metrics("vllm:generation_tokens_total 5.0\n"))
    assert view["swaps_total"] == 0
    assert view["jaccard"] is None
    assert view["tier_occupancy"] == {}


def test_prompt_families_are_distinct_and_formatted():
    seen = set()
    for family in se.FAMILIES:
        for i in range(6):
            p = se.build_prompt(family, i)
            assert "{a}" not in p and "{b}" not in p and "{c}" not in p
            seen.add(p)
    # distinct across families and indices — a repeated prompt would let the
    # prefix cache serve it and understate real routing pressure
    assert len(seen) == sum(min(6, 6) for _ in se.FAMILIES)


def test_prompt_is_deterministic_across_runs():
    assert se.build_prompt("math", 3) == se.build_prompt("math", 3)


def test_prompt_corpus_does_not_run_out_under_a_real_phase():
    """Three of four families interpolate only {a} (3..97): with a
    template-only corpus the ceiling was 3 x 95 = 285 prompts, and a 420 s
    phase at concurrency 24 re-issues each of them several times. Cached
    prefill routes no tokens, so the domain shift under-reports the very
    routing pressure it exists to create."""
    for family in se.FAMILIES:
        draws = [se.build_prompt(family, i) for i in range(5000)]
        assert len(set(draws)) == 5000, (
            f"{family}: only {len(set(draws))} distinct prompts in 5000 draws")


def test_prompt_is_deterministic_ACROSS_PROCESSES():
    """str.__hash__ is salted per interpreter, so seeding on hash((fam, i))
    gave every process a different corpus and made runs incomparable."""
    import subprocess

    code = ("import sys; sys.path.insert(0, %r)\n"
            "import swap_evidence as se\n"
            "print('|'.join(se.build_prompt(f, i) "
            "for f in se.FAMILIES for i in range(3)))"
            % str(Path(__file__).parent))
    outs = set()
    for seed in ("0", "1", "12345"):
        env = {**os.environ, "PYTHONHASHSEED": seed}
        outs.add(subprocess.run([sys.executable, "-c", code], env=env,
                                capture_output=True, text=True,
                                check=True).stdout)
    assert len(outs) == 1, "prompt corpus differs between interpreters"


# ------------------------------------------------------------- parser probes
def test_exemplars_and_timestamps_are_parsed_not_dropped():
    """OpenMetrics exemplars and the optional sample timestamp are legal;
    requiring the value to be the last token dropped the whole series."""
    flat = se.parse_metrics(
        'a_bucket{le="0.5"} 7 # {trace_id="abc"} 0.5 1520879607.789\n'
        "b_ts 42 1520879607789\n"
        "  c_indented 3\n")
    assert flat == {'a_bucket{le="0.5"}': 7.0, "b_ts": 42.0, "c_indented": 3.0}


def test_non_finite_samples_never_reach_the_timeline():
    """A 0/0 gauge legitimately exports NaN. One NaN in swaps_total turns
    every coordinate of that chart series into 'nan' — the SVG still parses
    and the line silently disappears — and json.dumps writes bare NaN, which
    is not valid JSON for any non-Python reader."""
    report: dict = {}
    flat = se.parse_metrics(
        'fq_swaps_total{layer="1"} NaN\n'
        'fq_swaps_total{layer="2"} 3\n'
        "fq_jaccard +Inf\nfq_probe_kld -Inf\n", report)
    assert flat == {'fq_swaps_total{layer="2"}': 3.0}
    assert sorted(report["nonfinite"]) == [
        "fq_jaccard", "fq_probe_kld", 'fq_swaps_total{layer="1"}']
    view = se.fq_view(flat)
    assert view["swaps_total"] == 3.0
    json.dumps(view, allow_nan=False)  # strict JSON, no NaN/Infinity tokens


def test_unparsable_lines_are_counted_not_silently_dropped():
    report: dict = {}
    se.parse_metrics('ok 1\nfoo{a="x}y"} 4\nnot a metric line at all\n', report)
    assert report["unparsed"] == 2


def test_series_matches_the_exact_metric_name_only():
    """startswith() also matched vllm:generation_tokens_created (a ~1.7e9
    unix timestamp) and every _sum/_count/_bucket of a same-prefixed
    histogram; summing those produced a plausible but wrong counter."""
    flat = se.parse_metrics(
        'vllm:generation_tokens_created{model_name="m"} 1770000000.0\n'
        'vllm:generation_tokens_sum{model_name="m"} 9.0\n'
        'vllm:generation_tokens_count{model_name="m"} 2.0\n'
        'vllm:generation_tokens_bucket{le="+Inf",model_name="m"} 2.0\n'
        'vllm:generation_tokens{model_name="m"} 500.0\n')
    assert se.series(flat, "vllm:generation_tokens") == [500.0]
    assert se.series(flat, "vllm:generation_tokens_total") == []


def test_decode_rate_is_never_negative_on_a_counter_reset(tmp_path,
                                                          monkeypatch):
    """An exporter restart makes the counter go backwards. (tok-prev)/dt is
    then a spike of minus-a-million tok/s, which fit() cannot see (it only
    looks at the max) and the chart draws off the bottom of the viewBox."""
    import http.server

    seq = iter([1000.0, 5.0, 5.0, 5.0])

    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            body = f"vllm:generation_tokens_total {next(seq, 5.0)}\n".encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    srv = http.server.HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    out = tmp_path / "reset.jsonl"
    stop = threading.Event()
    th = threading.Thread(target=se.scrape_loop,
                          args=(f"http://127.0.0.1:{srv.server_address[1]}",
                                out, 0.05, stop), daemon=True)
    th.start()
    time.sleep(0.4)
    stop.set()
    th.join(timeout=5)
    srv.shutdown()
    rows = [json.loads(x) for x in out.read_text().splitlines()]
    rates = [r["decode_tok_s"] for r in rows if "decode_tok_s" in r]
    assert rates, rows
    assert all(r >= 0 for r in rates), rates
    assert any(r.get("counter_reset") for r in rows), rows


def test_scrape_writes_rows_against_a_live_endpoint(tmp_path):
    """End-to-end against a real HTTP server, including the error path."""
    import http.server

    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            body = EXPOSITION.encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):  # silence
            pass

    srv = http.server.HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    out = tmp_path / "t.jsonl"
    stop = threading.Event()
    th = threading.Thread(target=se.scrape_loop,
                          args=(base, out, 0.05, stop), daemon=True)
    th.start()
    time.sleep(0.35)
    stop.set()
    th.join(timeout=5)
    srv.shutdown()

    rows = [json.loads(x) for x in out.read_text().splitlines()]
    assert len(rows) >= 2, rows
    assert rows[0]["fq"]["swaps_total"] == 17.0
    assert rows[0]["gen_tokens_total"] == 1234567.0
    assert rows[0]["requests_running"] == 16.0
    # counter is flat here, so the derived rate must be 0, not absent/garbage
    assert rows[1]["decode_tok_s"] == pytest.approx(0.0)


def _chat_stub():
    """A tiny /v1/chat/completions that answers with 10 completion tokens."""
    import http.server
    import socketserver

    class H(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self):  # noqa: N802
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            body = json.dumps({
                "choices": [{"message": {"content": "x"}}],
                "usage": {"completion_tokens": 10, "prompt_tokens": 5},
            }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    class TS(socketserver.ThreadingMixIn, http.server.HTTPServer):
        daemon_threads = True

    srv = TS(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"


def test_a_repeated_family_is_not_double_counted(tmp_path):
    """phase_end selected out of ONE shared results list with
    r["family"] == family, so a phase that repeats an earlier family reported
    that phase's requests too — measured: 180 requests and twice the real
    tok/s for a phase that actually issued 96."""
    srv, base = _chat_stub()
    try:
        out = tmp_path / "w.jsonl"
        totals = se.workload(base, "m", out,
                             [("math", 0.4), ("math", 0.4)],
                             concurrency=4, max_tokens=8,
                             stop=threading.Event())
    finally:
        srv.shutdown()
    ends = [json.loads(x) for x in out.read_text().splitlines()
            if json.loads(x).get("event") == "phase_end"]
    assert len(ends) == 2
    assert ends[1]["requests"] < ends[0]["requests"] * 2, ends
    assert totals["requests"] == ends[0]["requests"] + ends[1]["requests"]
    for e in ends:
        # tok_s and wall_seconds must come from ONE clock read, not two
        assert e["tok_s"] == pytest.approx(
            e["completion_tokens"] / e["wall_seconds"], rel=1e-12)


def test_a_workload_that_never_succeeds_exits_nonzero(tmp_path, monkeypatch):
    """All-failed requests still leave a complete, plausible timeline; the
    exit status is the only thing that can say it is not evidence."""
    out = tmp_path / "dead.jsonl"
    monkeypatch.setattr(sys, "argv", [
        "swap_evidence.py", "workload", "--base", "http://127.0.0.1:1",
        "--out", str(out), "--concurrency", "2", "--phase", "math:0.3"])
    assert se.main() == 3
    assert out.read_text().strip(), "timeline was written but the run failed"


def test_scrape_records_error_rows_when_endpoint_is_down(tmp_path):
    out = tmp_path / "down.jsonl"
    stop = threading.Event()
    th = threading.Thread(
        target=se.scrape_loop,
        # port 1 is reliably closed
        args=("http://127.0.0.1:1", out, 0.05, stop), daemon=True)
    th.start()
    time.sleep(0.2)
    stop.set()
    th.join(timeout=5)
    rows = [json.loads(x) for x in out.read_text().splitlines()]
    assert rows and all("error" in r for r in rows)
    # a dead endpoint must still produce a timeline row, so a gap in the
    # chart is visible as downtime rather than as missing data
    assert all(r["kind"] == "sample" for r in rows)
