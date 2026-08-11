"""Tests for the swap-evidence recorder.

The parser is the part that can silently lie: a metric name that moved between
vLLM versions, or a label regex that half-matches, would produce a plausible
but wrong timeline — exactly the kind of evidence that must not end up in a PR.
"""
from __future__ import annotations

import json
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
