"""The lazy-import budget gate must measure the import and survive host load.

Two properties are pinned here, both of which the gate depends on and neither of
which any other test reaches:

* A dependency-policy leak and an exceeded budget are different failures. The
  gate decides whether a pull request may merge, so a leak reported as "slow"
  sends the reader to the wrong file.
* The gated statistic ignores a slow sample. The gate runs on shared CI runners
  and on developer machines, and every sample it takes passes through CPython
  interpreter startup, which machine load moves and no change to this repository
  can.
"""

import subprocess
import sys
from pathlib import Path

import pytest

from tools import check_import_time

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[2]


def run_probe(forbidden: tuple[str, ...]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        (sys.executable, "-c", check_import_time.probe_source(forbidden)),
        capture_output=True,
        text=True,
        cwd=ROOT,
    )


def test_probe_reports_a_leak_on_stderr_with_its_own_exit_code():
    """A forbidden module that is certainly loaded, so the leak path runs."""

    completed = run_probe(("sys",))

    assert completed.returncode == 2
    assert "loaded: sys" in completed.stderr
    assert completed.stdout.strip() == ""


def test_probe_reports_a_clean_import_as_a_float_on_stdout():
    completed = run_probe(())

    assert completed.returncode == 0
    assert completed.stderr == ""
    assert float(completed.stdout.strip()) >= 0.0


def test_sample_imports_refuses_a_leaking_probe(monkeypatch: pytest.MonkeyPatch):
    """The gate cannot report a duration for a run that tripped the policy."""

    monkeypatch.setattr(check_import_time, "forbidden_imports", lambda: ("sys",))

    with pytest.raises(RuntimeError, match="dependency policy forbids"):
        check_import_time.sample_imports(1)


def test_sample_imports_reports_what_the_child_measured(
    monkeypatch: pytest.MonkeyPatch,
):
    """The duration comes from the child's stdout, not from this process' clock."""

    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=[], returncode=0, stdout="0.5\n", stderr=""
        )

    monkeypatch.setattr(check_import_time.subprocess, "run", fake_run)

    assert check_import_time.sample_imports(3) == (0.5, 0.5, 0.5)


def test_gate_ignores_a_slow_sample():
    """One slow sample is load, not a regression, so it must not gate."""

    assert check_import_time.gated_statistic((0.9, 0.1, 0.9)) == 0.1


def test_main_accepts_a_run_whose_fastest_sample_is_inside_the_budget(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    """A loaded host slows most samples; that alone must not fail the gate."""

    monkeypatch.setattr(
        check_import_time, "sample_imports", lambda repetitions: (0.1, 0.9, 0.9)
    )
    monkeypatch.setattr(sys, "argv", ["check_import_time.py", "--max-seconds", "0.2"])

    assert check_import_time.main() == 0

    # A passing run still reports the whole distribution, so a reader can see how
    # much of it was the host rather than the import.
    printed = capsys.readouterr().out
    assert "fastest=0.100000s" in printed
    assert "slowest=0.900000s" in printed


def test_main_states_why_it_failed_when_the_budget_is_exceeded(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    monkeypatch.setattr(
        check_import_time, "sample_imports", lambda repetitions: (0.5, 0.6)
    )
    monkeypatch.setattr(sys, "argv", ["check_import_time.py", "--max-seconds", "0.2"])

    assert check_import_time.main() == 1
    assert "import budget exceeded" in capsys.readouterr().err
