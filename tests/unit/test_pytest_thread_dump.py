"""Timed diagnostics preserve test outcomes and release their worker threads."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("capture", ("fd", "sys", "no"))
@pytest.mark.parametrize("failure", (False, True))
@pytest.mark.parametrize("workers", (0, 2))
def test_long_tests_dump_stacks_and_finish_with_coverage(
    tmp_path: Path, capture: str, failure: bool, workers: int
) -> None:
    (tmp_path / "pytest.ini").write_text("[pytest]\n")
    source = tmp_path / "test_workflow.py"
    source.write_text(
        "import faulthandler\n"
        "import threading\n"
        "import time\n"
        "def test_long_workflow():\n"
        "    assert faulthandler.is_enabled()\n"
        "    time.sleep(0.15)\n"
        f"    assert not {failure!r}\n"
        "def test_next_workflow():\n"
        "    timers = [t for t in threading.enumerate() "
        "if isinstance(t, threading.Timer)]\n"
        "    assert len(timers) == 1\n"
        "    assert timers[0].name.endswith('::test_next_workflow')\n"
    )
    environment = dict(
        os.environ,
        PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
        PYTEST_ADDOPTS="",
        COVERAGE_CORE="sysmon",
        COVERAGE_FILE=str(tmp_path / ".coverage"),
    )
    environment.pop("FQ_CI_REPORT_DIR", None)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-c",
            str(tmp_path / "pytest.ini"),
            "-p",
            "tools.pytest_thread_dump",
            "-p",
            "pytest_cov.plugin",
            *(["-p", "xdist.plugin", "-n", str(workers)] if workers else []),
            "--fq-thread-dump-timeout=0.05",
            f"--capture={capture}",
            f"--cov={tmp_path}",
            "--cov-report=",
            str(source),
            "-q",
        ],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
    )
    output = result.stdout + result.stderr
    assert result.returncode == int(failure), output
    assert ("1 failed, 1 passed" if failure else "2 passed") in output
    assert "Test exceeded 0.05s:" in output
    assert "test_long_workflow" in output
    assert "Thread " in output
    assert (tmp_path / ".coverage").is_file()


@pytest.mark.parametrize(
    "arguments,message",
    (
        (["--fq-thread-dump-timeout=-1"], "finite and non-negative"),
        (["--fq-thread-dump-timeout=nan"], "finite and non-negative"),
        (["--fq-thread-dump-timeout=inf"], "finite and non-negative"),
        (
            ["--fq-thread-dump-timeout=1", "-o", "faulthandler_timeout=1"],
            "Use only the Python thread watchdog",
        ),
    ),
)
def test_invalid_timeout_is_rejected(
    tmp_path: Path, arguments: list[str], message: str
) -> None:
    (tmp_path / "pytest.ini").write_text("[pytest]\n")
    environment = dict(
        os.environ, PYTEST_DISABLE_PLUGIN_AUTOLOAD="1", PYTEST_ADDOPTS=""
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-c",
            str(tmp_path / "pytest.ini"),
            "-p",
            "tools.pytest_thread_dump",
            *arguments,
            "--collect-only",
            str(tmp_path),
        ],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == pytest.ExitCode.USAGE_ERROR
    assert message in result.stderr


def test_diagnostic_errors_fail_the_owning_test(tmp_path: Path) -> None:
    (tmp_path / "pytest.ini").write_text("[pytest]\n")
    source = tmp_path / "test_diagnostic_failure.py"
    source.write_text(
        "import time\n"
        "from tools import pytest_thread_dump\n"
        "def broken_report(*args):\n"
        "    raise OSError('diagnostic output unavailable')\n"
        "def test_workflow(monkeypatch):\n"
        "    monkeypatch.setattr(pytest_thread_dump, '_dump_thread_stacks', "
        "broken_report)\n"
        "    time.sleep(0.15)\n"
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-c",
            str(tmp_path / "pytest.ini"),
            "-p",
            "tools.pytest_thread_dump",
            "--fq-thread-dump-timeout=0.05",
            str(source),
            "-q",
        ],
        cwd=ROOT,
        env=dict(os.environ, PYTEST_DISABLE_PLUGIN_AUTOLOAD="1", PYTEST_ADDOPTS=""),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode != 0
    output = result.stdout + result.stderr
    assert "The test thread diagnostic failed" in output
    assert "diagnostic output unavailable" in output
