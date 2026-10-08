import json
import sys
from pathlib import Path

import pytest
import yaml

from tools.run_with_watchdog import run
from tools.validate_required_checks import _produced_checks, unproduced_checks
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


def test_every_required_check_is_one_a_workflow_job_produces():
    """A required name nobody reports is an `Expected` check that blocks `main`.

    All eight names are enforced on `main`, so a name no job produces never
    reports and nothing can merge. The reverse also matters: a name that no
    longer matches what a job reports stops gating silently.
    """

    root = Path(__file__).resolve().parents[2]
    assert unproduced_checks(root) == []


def test_a_required_check_no_job_produces_is_reported(tmp_path):
    """The gate names the offending check instead of merging nothing."""

    root = Path(__file__).resolve().parents[2]
    contract = json.loads((root / ".github/required-checks.json").read_text())
    workflows = tmp_path / ".github" / "workflows"
    workflows.mkdir(parents=True)
    (workflows / "ci.yml").write_text(
        "name: CI\non: push\njobs:\n  quality:\n"
        "    runs-on: ubuntu-latest\n    steps:\n      - run: true\n"
    )
    (tmp_path / ".github" / "required-checks.json").write_text(json.dumps(contract))
    assert unproduced_checks(tmp_path) == sorted(
        set(contract["checks"]) - {"CI / quality"}
    )


def test_matrix_check_names_are_derived_the_way_github_names_them():
    """A matrix check name comes from the matrix, and both matrix forms count.

    GitHub joins the values of the matrix keys with `, ` in declaration order,
    so `cpu-core` over one interpreter is `cpu-core (3.10)`, and an `include:`
    entry contributes its values in that entry's order. A renamed interpreter
    must change the check name rather than leave a required name unreported.
    """

    root = Path(__file__).resolve().parents[2]
    produced = _produced_checks(root)
    assert "CI / cpu-core (3.10)" in produced
    assert "CI / cpu-core (3.12)" in produced
    assert "CI / dependency-bounds (3.10, torch==2.5.*)" in produced
    assert "CI / dependency-bounds (3.12, torch>=2.5,<2.14)" in produced
    document = yaml.safe_load(
        (root / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    )
    assert "strategy" not in document["jobs"]["coverage"]


def test_manual_gpu_workflows_fail_closed_and_bound_long_commands():
    root = Path(__file__).resolve().parents[2]
    local = (root / ".github/workflows/local-gpu.yml").read_text()
    scheduled = (root / ".github/workflows/scheduled-hardware.yml").read_text()
    trigger = local.split("jobs:", 1)[0]
    assert "workflow_dispatch:" in trigger
    assert "pull_request" not in trigger
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
