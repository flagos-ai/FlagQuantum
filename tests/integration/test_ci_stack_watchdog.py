"""The diagnostic watchdog preserves pytest outcomes and releases its thread."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[2]


def _run(
    directory: Path, source: str, *options: str
) -> subprocess.CompletedProcess[str]:
    (directory / "test_example.py").write_text(source, encoding="utf-8")
    # Run outside the repository's conftest and optional dependency collection.
    # Check for live watchdogs after pytest has completed in the same process.
    driver = (
        "import sys, threading, pytest\n"
        "status = pytest.main(sys.argv[1:])\n"
        "assert not any(t.name == 'fq-stack-watchdog' for t in threading.enumerate())\n"
        "print('Watchdog threads released', flush=True)\n"
        "raise SystemExit(status)\n"
    )
    return subprocess.run(
        [
            sys.executable,
            "-c",
            driver,
            "test_example.py",
            "--rootdir",
            str(directory),
            "-c",
            os.devnull,
            "-p",
            "tools.ci.pytest_stack_watchdog",
            "-q",
            *options,
        ],
        cwd=directory,
        env={
            **os.environ,
            "PYTHONPATH": os.pathsep.join(
                (str(ROOT), os.environ.get("PYTHONPATH", ""))
            ),
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
            # Parent CI worker/report options belong to the outer pytest run.
            "PYTEST_ADDOPTS": "",
        },
        capture_output=True,
        text=True,
        timeout=30,
    )


@pytest.mark.parametrize("failure", [False, True])
def test_slow_test_reports_its_stack_and_preserves_outcome(
    tmp_path: Path, failure: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent_report = tmp_path / "parent-report.xml"
    monkeypatch.setenv(
        "PYTEST_ADDOPTS", f"-n 12 --dist=worksteal --junitxml={parent_report}"
    )
    result = _run(
        tmp_path,
        "import time\ndef test_slow():\n    time.sleep(0.15)\n"
        f"    assert {not failure}\n",
        "--fq-stack-dump-after=0.02",
    )
    assert result.returncode == int(failure), result.stdout + result.stderr
    assert (
        "Python stack snapshot after 0.02s: test_example.py::test_slow" in result.stderr
    )
    assert "in test_slow" in result.stderr
    assert "Watchdog threads released" in result.stdout
    assert not parent_report.exists()


@pytest.mark.parametrize("seconds", ["0", "60"])
def test_fast_test_cancels_pending_diagnostics(tmp_path: Path, seconds: str) -> None:
    result = _run(
        tmp_path,
        "def test_fast():\n    assert sum(range(100)) == 4950\n",
        f"--fq-stack-dump-after={seconds}",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Python stack snapshot" not in result.stderr
    assert "Watchdog threads released" in result.stdout


def test_setup_failure_cancels_pending_diagnostics(tmp_path: Path) -> None:
    result = _run(
        tmp_path,
        "import pytest\n@pytest.fixture\ndef value():\n    raise ValueError('setup failure')\n"
        "def test_setup(value):\n    assert value\n",
        "--fq-stack-dump-after=60",
    )
    assert result.returncode == 1, result.stdout + result.stderr
    assert "setup failure" in result.stdout
    assert "Watchdog threads released" in result.stdout


def test_each_test_owns_its_watchdog(tmp_path: Path) -> None:
    result = _run(
        tmp_path,
        "import time\ndef test_first():\n    assert True\n"
        "def test_second():\n    time.sleep(0.6)\n",
        "--fq-stack-dump-after=0.3",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert (
        "Python stack snapshot after 0.3s: test_example.py::test_second"
        in result.stderr
    )
    assert (
        "Python stack snapshot after 0.3s: test_example.py::test_first"
        not in result.stderr
    )
    assert "Watchdog threads released" in result.stdout


@pytest.mark.parametrize("seconds", ["-1", "nan", "inf"])
def test_invalid_interval_fails_before_execution(tmp_path: Path, seconds: str) -> None:
    result = _run(
        tmp_path,
        "def test_unreached():\n    assert False\n",
        f"--fq-stack-dump-after={seconds}",
    )
    assert result.returncode == 4, result.stdout + result.stderr
    assert "must be finite and nonnegative" in result.stderr


def test_native_delayed_dump_cannot_be_enabled_with_the_watchdog(
    tmp_path: Path,
) -> None:
    result = _run(
        tmp_path,
        "def test_unreached():\n    assert False\n",
        "--fq-stack-dump-after=1",
        "-o",
        "faulthandler_timeout=1",
    )
    assert result.returncode == 4, result.stdout + result.stderr
    assert "requires faulthandler_timeout=0" in result.stderr


def test_xdist_workers_emit_diagnostics_and_complete(tmp_path: Path) -> None:
    result = _run(
        tmp_path,
        "import time\ndef test_left():\n    time.sleep(0.15)\n"
        "def test_right():\n    time.sleep(0.15)\n",
        "--fq-stack-dump-after=0.02",
        "-p",
        "xdist.plugin",
        "-n",
        "2",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert (
        "Python stack snapshot after 0.02s: test_example.py::test_left" in result.stderr
    )
    assert (
        "Python stack snapshot after 0.02s: test_example.py::test_right"
        in result.stderr
    )
    assert "2 passed" in result.stdout
