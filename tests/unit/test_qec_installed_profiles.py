"""Check source smoke logic and failure isolation; real installs run in package CI."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from tools.check_qec_installed_profiles import _run
from tools.qec_installed_smoke import PROFILES, _exercise

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("profile", PROFILES)
def test_public_workflow_in_the_available_source_profile(profile: str) -> None:
    available = tuple(
        importlib.util.find_spec(name) is not None for name in ("stim", "pymatching")
    )
    if available != PROFILES[profile]:
        pytest.skip("this source environment does not have that real SDK configuration")
    records = _exercise(profile)
    assert len(records) == (4 if available[0] else 0)
    for record in records:
        assert record["shots"] == 35
        assert record["observables"] == 1


def test_installer_child_cannot_import_through_caller_python_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in ("PYTHONPATH", "PYTHONHOME", "PYTHONUSERBASE"):
        monkeypatch.setenv(name, "/unrelated/source")
    log = tmp_path / "child.log"
    _run(
        [
            sys.executable,
            "-c",
            "import json,os; print(json.dumps([os.getenv(x) for x in "
            "('PYTHONPATH','PYTHONHOME','PYTHONUSERBASE')]))",
        ],
        cwd=tmp_path,
        log=log,
        seconds=10,
    )
    assert json.loads(log.read_text()) == [None, None, None]


def test_failed_install_is_not_reclassified_as_a_success(tmp_path: Path) -> None:
    log = tmp_path / "failed.log"
    with pytest.raises(subprocess.CalledProcessError) as error:
        _run(
            [sys.executable, "-c", "print('preserved failure'); raise SystemExit(13)"],
            cwd=tmp_path,
            log=log,
            seconds=10,
        )
    assert error.value.returncode == 13
    assert "preserved failure" in log.read_text()


def test_dependency_warning_does_not_corrupt_the_json_report(tmp_path: Path) -> None:
    report = tmp_path / "smoke.json"
    _run(
        [
            sys.executable,
            "-c",
            "import json,sys; print('dependency warning', file=sys.stderr); "
            "print(json.dumps({'profile': 'core'}))",
        ],
        cwd=tmp_path,
        log=report,
        seconds=10,
    )
    assert json.loads(report.read_text()) == {"profile": "core"}
    assert "dependency warning" in report.with_suffix(".stderr.log").read_text()


def test_hanging_child_is_bounded_and_its_log_is_retained(tmp_path: Path) -> None:
    log = tmp_path / "timed-out.log"
    with pytest.raises(subprocess.TimeoutExpired):
        _run(
            [
                sys.executable,
                "-c",
                "import time; print('started', flush=True); time.sleep(30)",
            ],
            cwd=tmp_path,
            log=log,
            seconds=1,
        )
    assert "started" in log.read_text()
