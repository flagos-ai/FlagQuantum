"""Worker failures retain actual process evidence without changing pytest policy."""

import json
import os
import signal
import subprocess
import sys
from types import SimpleNamespace

import pytest

from tools.pytest_shard import pytest_testnodedown

pytestmark = pytest.mark.unit


def _node(process: subprocess.Popen[str] | None) -> SimpleNamespace:
    return SimpleNamespace(
        gateway=SimpleNamespace(id="gw3", _io=SimpleNamespace(popen=process))
    )


def _exited_process(script: str) -> subprocess.Popen[str]:
    process = subprocess.Popen([sys.executable, "-c", script], text=True)
    try:
        process.wait(timeout=10)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=10)
    return process


def test_failed_worker_records_actual_exit_code(
    capsys: pytest.CaptureFixture[str],
) -> None:
    process = _exited_process("raise SystemExit(17)")
    pytest_testnodedown(_node(process), "Not properly terminated")
    payload = json.loads(capsys.readouterr().err.removeprefix("CI worker exit: "))
    assert payload == {
        "worker": "gw3",
        "error": "Not properly terminated",
        "process_exit_status": 17,
        "termination_signal": None,
    }


@pytest.mark.skipif(os.name != "posix", reason="POSIX signal exit status")
def test_killed_worker_records_actual_signal(
    capsys: pytest.CaptureFixture[str],
) -> None:
    process = _exited_process("import os, signal; os.kill(os.getpid(), signal.SIGTERM)")
    pytest_testnodedown(_node(process), "Not properly terminated")
    payload = json.loads(capsys.readouterr().err.removeprefix("CI worker exit: "))
    assert payload["process_exit_status"] == -signal.SIGTERM
    assert payload["termination_signal"] == signal.SIGTERM


def test_missing_process_is_explicitly_unknown(
    capsys: pytest.CaptureFixture[str],
) -> None:
    pytest_testnodedown(_node(None), "transport unavailable")
    payload = json.loads(capsys.readouterr().err.removeprefix("CI worker exit: "))
    assert payload["process_exit_status"] is None
    assert payload["termination_signal"] is None
    assert payload["error"] == "transport unavailable"


def test_clean_worker_does_not_emit_failure(capsys: pytest.CaptureFixture[str]) -> None:
    pytest_testnodedown(_node(None), None)
    assert capsys.readouterr().err == ""
