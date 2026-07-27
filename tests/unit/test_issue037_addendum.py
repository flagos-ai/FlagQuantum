import json
import sys
from pathlib import Path

import pytest

from tools.run_with_watchdog import run
from tools.validate_required_checks import main as validate_checks

pytestmark = pytest.mark.unit


def test_required_check_and_watchdog_policy_are_valid():
    assert validate_checks() == 0
    root = Path(__file__).resolve().parents[2]
    for workflow in ("local-gpu.yml", "scheduled-hardware.yml"):
        assert (
            "run_with_watchdog.py"
            in (root / ".github/workflows" / workflow).read_text()
        )


def test_required_gpu_workflows_fail_closed_and_bound_long_commands():
    root = Path(__file__).resolve().parents[2]
    local = (root / ".github/workflows/local-gpu.yml").read_text()
    scheduled = (root / ".github/workflows/scheduled-hardware.yml").read_text()
    trigger = local.split("pull_request:", 1)[1].split("jobs:", 1)[0]
    assert "paths:" not in trigger
    assert local.count("run_with_watchdog.py") >= 10
    assert scheduled.count("run_with_watchdog.py") >= 8
    scale_lane = scheduled.split("local-scale-scheduled:", 1)[1]
    assert "set -o pipefail" in scale_lane


def test_watchdog_preserves_diagnostics_and_terminates_stall(tmp_path):
    diagnostics = tmp_path / "diagnostics"
    code = run(
        [
            "--phase",
            "collective",
            "--stall-seconds",
            "0.2",
            "--timeout-seconds",
            "2",
            "--diagnostics",
            str(diagnostics),
            "--",
            sys.executable,
            "-c",
            "import time; time.sleep(5)",
        ]
    )
    assert code == 124
    payload = json.loads((diagnostics / "watchdog.json").read_text())
    assert payload["phase"] == "collective"
    assert payload["reason"] == "no_progress_timeout"
    assert payload["cleanup_required"] is True
    assert (diagnostics / "processes.log").exists()
    assert (diagnostics / "rank-stacks.log").exists()
