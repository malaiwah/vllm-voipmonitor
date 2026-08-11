"""Regression tests for the shell logic in `run-demo1.sh`.

The runner is `set -euo pipefail`, which is what makes it trustworthy — and
also what makes three ordinary-looking lines into run-enders: a command that
returns non-zero in a plain `cmd; cmd` sequence aborts the whole script, with
no `die` message and no scoring, potentially after an hour of GPU time.

These tests execute the REAL lines, extracted out of `run-demo1.sh` by their
text, under the same shell options the script sets. Rewriting the line in the
test would test the test; extracting it means the test dies with the script if
somebody reintroduces the bug.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).parent
SCRIPT = HERE / "run-demo1.sh"
SRC = SCRIPT.read_text()
LINES = SRC.splitlines()

sys.path.insert(0, str(HERE))
import make_scenario1_policy as mk  # noqa: E402

PRELUDE = "set -euo pipefail\n"


def run_bash(body: str) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", "-c", PRELUDE + body],
                          capture_output=True, text=True, timeout=60)


def one_line(needle: str) -> str:
    """The single line of run-demo1.sh containing ``needle``.

    A leading ``^`` anchors the needle to the start of the line, which is how
    the top-level `kill -INT` is told apart from the one inside `cleanup()`.
    """
    if needle.startswith("^"):
        hits = [ln for ln in LINES if ln.startswith(needle[1:])]
    else:
        hits = [ln for ln in LINES
                if needle in ln and not ln.lstrip().startswith("#")]
    assert len(hits) == 1, f"expected exactly one line with {needle!r}, got {hits}"
    return hits[0]


def flat(text: str) -> str:
    """Collapse the runner's wrapped multi-line `die`/`say` messages."""
    return " ".join(text.split())


def gate_block() -> str:
    """The step-1 policy gate: from `read -r N_SLOTS` to its end marker."""
    i = next(k for k, ln in enumerate(LINES) if ln.startswith("read -r N_SLOTS"))
    ends = [k for k, ln in enumerate(LINES) if "policy gate end" in ln]
    assert ends, "run-demo1.sh lost the '# ---- policy gate end' marker"
    return "\n".join(LINES[i:ends[0]])


# ------------------------------------------------- step 9: scraper shutdown

DEAD_PID = 2147483646  # above every plausible pid_max; never a live process


def test_scraper_shutdown_survives_a_scraper_that_already_exited():
    """swap_evidence.py scrape RETURNS once --duration (FQ_SCRAPE_MAX,
    default 4 h) elapses. Killing a pid that is already gone exits 1, and in
    a bare `kill; wait` sequence that aborts the run at step 9 — after the
    replay and the eval, before convergence scoring and the charts."""
    out = run_bash(f'SCRAPE_PID={DEAD_PID}\n'
                   + one_line('^kill -INT "$SCRAPE_PID"') + "\n"
                   + one_line('wait "$SCRAPE_PID"') + "\n"
                   + 'echo REACHED_SCORING\n')
    assert out.returncode == 0, out.stderr
    assert "REACHED_SCORING" in out.stdout


def test_serve_pgid_probe_survives_a_serve_that_died_immediately():
    """`ps -o pgid= -p <dead>` exits 1; with pipefail the assignment fails and
    `set -e` kills the runner one line BEFORE `trap cleanup EXIT` is armed —
    so the operator gets no message and no serve.log tail."""
    out = run_bash(f'SERVE_PID={DEAD_PID}\n'
                   + one_line("SERVE_PGID=$(ps -o pgid=") + "\n"
                   + 'echo "REACHED pgid=[${SERVE_PGID}]"\n')
    assert out.returncode == 0, out.stderr
    assert "REACHED pgid=[]" in out.stdout


# ------------------------------- step 9: fragment-substitution honesty check

SUBS_CHECK = '\n'.join([
    one_line('SUBS=$(grep -aic "substitut"'),
    '[ "$SUBS" -eq 0 ] || echo "WARN_SUBSTITUTIONS $SUBS"',
    'echo "SUBS=[$SUBS]"',
])


def test_substitution_warning_is_silent_on_a_clean_log(tmp_path):
    """`grep -c` prints 0 and exits 1 on no match, so `$(grep -c ... || echo
    0)` yields "0\\n0" — `[ -eq ]` then errors and the fallback-bitrate
    WARNING fires on every clean run, which trains the operator to ignore the
    one line that says the occupancy table is lying."""
    (tmp_path / "fq-lines.log").write_text("FQ loop: armed\nswap L3 e1<-e2\n")
    out = run_bash(f'OUT={tmp_path}\n' + SUBS_CHECK)
    assert out.returncode == 0, out.stderr
    assert "SUBS=[0]" in out.stdout
    assert "WARN_SUBSTITUTIONS" not in out.stdout
    assert "integer expression expected" not in out.stderr


def test_substitution_warning_still_fires_when_there_are_substitutions(tmp_path):
    (tmp_path / "fq-lines.log").write_text(
        "substituting K3 for K4 L3/e1\nsubstitution L4/e2\nswap ok\n")
    out = run_bash(f'OUT={tmp_path}\n' + SUBS_CHECK)
    assert out.returncode == 0, out.stderr
    assert "SUBS=[2]" in out.stdout
    assert "WARN_SUBSTITUTIONS 2" in out.stdout


def test_substitution_counter_treats_a_missing_log_as_zero(tmp_path):
    out = run_bash(f'OUT={tmp_path}\n' + SUBS_CHECK)
    assert out.returncode == 0, out.stderr
    assert "SUBS=[0]" in out.stdout


# ----------------------------------------- step 1: the saturated-pool gate

E = 256
GATE_PRELUDE = (
    'say() { echo "SAY $*"; }\n'
    'die() { echo "DIE $*" >&2; exit 1; }\n'
    f'PY={sys.executable}\n'
)


def _policy(tmp_path, name, *, fill):
    """A seeded policy over one covered layer with a 10-fragment pool."""
    ref = tmp_path / f"{name}-ref.json"
    ref.write_text(json.dumps({
        "reference": {"repo_id": "synthetic"},
        "per_layer_k4_sets": {
            "4": {"n_k4": 10, "n_k3": E - 10,
                  "k4_experts": list(range(10))},
            "5": {"n_k4": 0, "n_k3": E, "k4_experts": []},
        }}))
    doc = mk.build(ref, E, "m" * 64, "seeded",
                   coverage={4: set(range(10))}, fill_fraction=fill)
    out = tmp_path / f"{name}.json"
    out.write_text(json.dumps(doc))
    return out


def test_a_saturated_fragment_pool_is_refused_before_boot(tmp_path):
    """FQ_FILL=1.0 fills every K4 slot the pool can back, so there is no
    expert left to promote INTO: the loop draws a flat line for the whole run
    and looks broken. The runner must refuse instead of spending the GPU."""
    policy = _policy(tmp_path, "saturated", fill=1.0)
    assert json.loads(policy.read_text())["provenance"]["saturated_layers"] == [4]
    out = run_bash(GATE_PRELUDE + f'POLICY={policy}\nFILL=1.0\n' + gate_block())
    assert out.returncode != 0
    assert "ZERO legal promotion targets" in flat(out.stderr)


def test_a_half_filled_pool_passes_the_gate(tmp_path):
    policy = _policy(tmp_path, "half", fill=0.5)
    out = run_bash(GATE_PRELUDE + f'POLICY={policy}\nFILL=0.5\n' + gate_block())
    assert out.returncode == 0, out.stderr
    assert "1 covered layers (1 tradable)" in flat(out.stdout)
    assert "WARNING" not in out.stdout


def test_a_zero_slot_policy_is_still_refused(tmp_path):
    """The pre-existing gate must survive the new one: no coverage at all is
    a different failure from a saturated pool, and keeps its own message."""
    ref = tmp_path / "ref.json"
    ref.write_text(json.dumps({
        "reference": {"repo_id": "synthetic"},
        "per_layer_k4_sets": {"4": {"n_k4": 10, "n_k3": E - 10,
                                    "k4_experts": list(range(10))}}}))
    doc = mk.build(ref, E, "m" * 64, "seeded", coverage={}, fill_fraction=0.5)
    policy = tmp_path / "empty.json"
    policy.write_text(json.dumps(doc))
    out = run_bash(GATE_PRELUDE + f'POLICY={policy}\nFILL=0.5\n' + gate_block())
    assert out.returncode != 0
    assert "zero K4 slots" in flat(out.stderr)


# ------------------------------------------------- the EXIT trap's exit code

def cleanup_block() -> str:
    """`cleanup()` verbatim, from its `cleanup() {` line to `trap cleanup`."""
    i = next(k for k, ln in enumerate(LINES) if ln.startswith("cleanup() {"))
    j = next(k for k, ln in enumerate(LINES) if ln.startswith("trap cleanup EXIT"))
    return "\n".join(LINES[i:j + 1])


# A pgrep pattern that cannot match anything, kept OUT of every command line
# the test itself runs: `pgrep -f` matches the full cmdline, so spelling the
# pattern inline makes the enclosing shell match itself and the test passes
# for the wrong reason.
NO_MATCH_PAT = "zz-fq-advrev-no-such-process-zz"

TRAP_PRELUDE = (
    'say() { echo "SAY $*"; }\n'
    f'SERVE_PID={DEAD_PID}\n'
    'SCRAPE_PID=""\n'
)


def _cleanup_body(pattern_file: str) -> str:
    """cleanup(), with the leftover-worker pattern read from a file."""
    body = cleanup_block().replace('pgrep -f "VLLM::"', 'pgrep -f "$PAT"')
    return f'PAT=$(cat {pattern_file})\n' + body


def test_cleanup_does_not_turn_a_successful_run_into_a_failure(tmp_path):
    """The EXIT trap runs under `set -e` too.

    On the happy path the serve has already drained, so `kill -KILL -- -PGID`
    returns ESRCH and `pgrep -f VLLM::` matches nothing (pipefail promotes
    pgrep's 1 to the whole pipeline). Either one aborts the handler, skipping
    `return $rc`, and the shell exits with 1 — so a demo that printed DONE and
    wrote every artifact reports failure to anything that checks $?.
    """
    pat = tmp_path / "pat"
    pat.write_text(NO_MATCH_PAT + "\n")
    out = run_bash(TRAP_PRELUDE
                   + f'SERVE_PGID={DEAD_PID}\n'
                   + _cleanup_body(str(pat))
                   + '\nsay "DONE — artifacts in OUT"\n')
    assert out.returncode == 0, (
        f"a successful run exited {out.returncode}\n"
        f"stdout={out.stdout}\nstderr={out.stderr}")
    # The handler must run to the END, not just happen to exit 0.
    assert "shutting down serve" in out.stdout


def test_cleanup_preserves_a_real_failure_code(tmp_path):
    """...and it must not swallow one either."""
    pat = tmp_path / "pat"
    pat.write_text(NO_MATCH_PAT + "\n")
    out = run_bash(TRAP_PRELUDE
                   + 'SERVE_PGID=""\n'
                   + _cleanup_body(str(pat))
                   + '\nexit 7\n')
    assert out.returncode == 7, out.stdout


def test_cleanup_still_warns_about_leftover_workers(tmp_path):
    """The `|| left=0` fallback must not disarm the warning it guards."""
    pat = tmp_path / "pat"
    pat.write_text("sleep\n")          # matches the sleeper started below
    body = (TRAP_PRELUDE + 'SERVE_PGID=""\n' + _cleanup_body(str(pat))
            + '\nsleep 7 & SLEEPER=$!\n'
            + 'say "DONE"\n')
    out = run_bash(body)
    subprocess.run(["pkill", "-f", "sleep 7"], capture_output=True)
    assert out.returncode == 0, out.stderr
    assert "still alive" in out.stdout


# ------------------------------------------ step 2: the checkpoint tier gate

def ckpt_gate_block() -> str:
    """The step-2 python heredoc, extracted from the runner.

    The heredoc body starts after the *whole* logical command line, which is
    continued with a backslash onto the `2>&1 | tee` redirect — so walk the
    continuations rather than assuming the body starts on the next line.
    """
    i = next(k for k, ln in enumerate(LINES)
             if ln.startswith('$PY - "$CKPT" "$POLICY"'))
    assert "<<'PYEOF'" in LINES[i], LINES[i]
    while LINES[i].rstrip().endswith("\\"):
        i += 1
    body = i + 1
    end = next(k for k, ln in enumerate(LINES[body:], body) if ln == "PYEOF")
    return "\n".join(LINES[body:end])


def _run_ckpt_gate(tmp_path, policy_bits, ckpt_bits):
    gate = tmp_path / "gate.py"
    gate.write_text(ckpt_gate_block())
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / "tier_bitmap.json").write_text(json.dumps(
        {str(l): {"bits_per_expert": b} for l, b in ckpt_bits.items()}))
    policy = tmp_path / "policy.json"
    policy.write_text(json.dumps(
        {"bits_per_expert": {str(l): b for l, b in policy_bits.items()}}))
    return subprocess.run(
        [sys.executable, str(gate), str(ckpt), str(policy),
         str(tmp_path / "match.json")],
        capture_output=True, text=True, timeout=60)


def test_checkpoint_gate_accepts_an_exact_match(tmp_path):
    bits = {4: [4, 4, 3, 3], 5: [3, 3, 3, 3]}
    out = _run_ckpt_gate(tmp_path, bits, {k: list(v) for k, v in bits.items()})
    assert out.returncode == 0, out.stdout + out.stderr
    assert "matches the policy on all 2 layers" in out.stdout


def test_checkpoint_gate_rejects_a_k2_or_k5_slab_where_the_policy_says_k3(tmp_path):
    """A K4-set diff cannot see a K2/K5 expert.

    The published segment family (`malaiwah/GLM-5.2-EXL3-FQ-segments`) is
    K2 x 64 + K3 x 76 + K5 x 24 and zero K4, so a checkpoint assembled from
    it declares K2/K5 exactly where this policy declares K3. Comparing only
    `{e | b == 4}` calls that a match, and the run boots to discover that
    MixedLayerState takes tier_bits == (3, 4) only — and that a K5 mixed
    tier is a hard SM120 shared-memory failure.
    """
    policy = {4: [4, 4, 3, 3], 5: [3, 3, 3, 3]}
    ckpt = {4: [4, 4, 3, 3], 5: [2, 3, 5, 3]}
    out = _run_ckpt_gate(tmp_path, policy, ckpt)
    assert out.returncode != 0, (
        "a K2/K5 checkpoint passed the gate that exists to catch it:\n"
        + out.stdout)
    assert "K[2, 5]" in out.stdout
    assert "two-tier" in out.stdout


def test_checkpoint_gate_rejects_a_wrong_width_bitmap(tmp_path):
    out = _run_ckpt_gate(tmp_path, {4: [4, 4, 3, 3]}, {4: [4, 4, 3]})
    assert out.returncode != 0
    assert "3 experts, the policy has 4" in out.stdout


def test_checkpoint_gate_still_rejects_a_missing_k4_slab(tmp_path):
    """The pre-existing check must survive the new ones."""
    out = _run_ckpt_gate(tmp_path, {4: [4, 4, 3, 3]}, {4: [3, 3, 3, 3]})
    assert out.returncode != 0
    assert "declared-but-absent" in out.stdout


# ------------------------------------------------ the memory-envelope default

def test_default_max_gib_buys_exactly_the_reference_headroom():
    """`FQ_MAX_GIB` is documented as "the reference's own headroom".

    reference-coder-quant.json records 8042 promotions at 1,179,648 B/rank =
    8.835205... GiB. A default of 8.835 floors to 8041 — one promotion fewer
    than the human spent, which stops the run being the equal-budget
    comparison the whole experiment is premised on.
    """
    default = one_line("MAX_GIB=${FQ_MAX_GIB:-").split(":-")[1].rstrip("}")
    ref = json.loads((HERE / "reference-coder-quant.json").read_text())
    want = ref["memory"]["headroom_equals_n_promotions"]
    got = mk.promotions_for_gib(float(default), 4)
    assert got == want, (
        f"FQ_MAX_GIB default {default} buys {got} promotions, "
        f"reference headroom is {want}")
    # and it must not overshoot into a budget the human did not spend
    assert mk.promotions_for_gib(float(default), 4) * mk.PROMOTION_BYTES // 4 \
        <= int(float(default) * (1 << 30))


# ------------------------------------------------- README numbers vs reality

def test_readme_quotes_the_ceiling_the_scorer_actually_uses():
    """`human_human_ceiling` is the MAX over siblings, not the 3.40bpw one.

    `fraction_of_human_agreement` divides by it, so a README quoting the
    3.40bpw number (0.657) mis-scales every headline the run prints by ~2%.
    """
    ref = json.loads((HERE / "reference-coder-quant.json").read_text())
    sibs = ref["sibling_human_builds_for_baseline"]
    ceiling = max(v["mean_per_layer_jaccard"] for v in sibs.values())
    readme = (HERE / "demo1-README.md").read_text()
    assert f"{ceiling:.3f}" in readme, (
        f"README does not quote the scorer's ceiling {ceiling:.3f}")


def test_readme_documents_the_max_gib_default_the_script_uses():
    default = one_line("MAX_GIB=${FQ_MAX_GIB:-").split(":-")[1].rstrip("}")
    readme = (HERE / "demo1-README.md").read_text()
    assert f"`FQ_MAX_GIB` | {default} |" in readme, (
        f"README knob table does not document FQ_MAX_GIB default {default}")


# ------------------------------------------------------------ sanity checks

def test_the_script_parses():
    assert subprocess.run(["bash", "-n", str(SCRIPT)],
                          capture_output=True, text=True).returncode == 0


@pytest.mark.parametrize("path", [
    "make_scenario1_policy.py", "score_convergence.py", "make_charts.py",
    "replay_mtp78.py", "swap_evidence.py", "reference-coder-quant.json",
    "harness/load_mtp78_corpus.py",
])
def test_preflight_names_files_that_exist(path):
    """Step 0 `have`s each of these; a rename here fails the run at boot."""
    assert f'"$RUN/{path}"' in SRC
    assert (HERE / path).is_file()
