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
