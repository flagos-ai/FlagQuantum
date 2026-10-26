"""A proposed performance budget must not certify missing or invalid timings."""

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from benchmarks import module_execution_matrix as matrix
from benchmarks.module_execution_matrix import within_budget

pytestmark = pytest.mark.benchmark_contract


def test_public_overhead_budget_covers_relative_and_absolute_cost() -> None:
    # Permit five microseconds of fixed dispatch overhead on a tiny kernel.
    assert within_budget(6e-6, 1e-6, 1.15, 5e-6)
    assert not within_budget(7e-6, 1e-6, 1.15, 5e-6)
    # The same allowance must not hide a large relative loss on a longer call.
    assert within_budget(1.1, 1.0, 1.15, 5e-6)
    assert not within_budget(1.3, 1.0, 1.15, 5e-6)


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -1.0])
@pytest.mark.parametrize("position", [0, 1, 3])
def test_invalid_timing_cannot_pass_the_budget(invalid: float, position: int) -> None:
    values = [0.2, 0.1, 1.15, 5e-6]
    values[position] = invalid
    with pytest.raises(ValueError, match="finite and non-negative"):
        within_budget(*values)


@pytest.mark.parametrize("ratio", [0.0, float("nan"), float("inf")])
def test_invalid_ratio_cannot_certify_a_run(ratio: float) -> None:
    with pytest.raises(ValueError, match="ratio"):
        within_budget(0.2, 0.1, ratio, 5e-6)


@pytest.mark.parametrize("timeout", [False, True])
def test_worker_failures_are_retained_and_fail_without_enforcement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, timeout: bool
) -> None:
    output = tmp_path / "failed"
    monkeypatch.setattr(sys, "argv", ["matrix", "--output", str(output), "--smoke"])
    monkeypatch.setattr(matrix, "version", lambda name: "test")
    monkeypatch.setattr(
        matrix.subprocess, "check_output", lambda *args, **kwargs: "test"
    )

    def run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        if timeout:
            raise subprocess.TimeoutExpired(command, 120)
        return subprocess.CompletedProcess(command, 9)

    monkeypatch.setattr(matrix.subprocess, "run", run)
    assert matrix.main() == 1
    report = json.loads((output / "summary.json").read_text())
    assert report["cases"][0]["status"] == ("timeout" if timeout else "failed")
    assert "measurements" not in report["cases"][0]
    assert (output / "case-00.log").exists()


@pytest.mark.parametrize("enforce", [False, True])
def test_completed_measurements_cannot_hide_failed_budgets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, enforce: bool
) -> None:
    output = tmp_path / "completed"
    arguments = ["matrix", "--output", str(output), "--smoke"]
    monkeypatch.setattr(sys, "argv", arguments + (["--enforce"] if enforce else []))
    monkeypatch.setattr(matrix, "version", lambda name: "test")
    monkeypatch.setattr(
        matrix.subprocess, "check_output", lambda *args, **kwargs: "test"
    )

    def run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        rows = [
            {"route": route, "demand": demand, "median_seconds": seconds}
            for demand in ("parameters", "inputs", "both", "no_grad", "inference")
            for route, seconds in (
                ("default_module", 0.3),
                ("jax_module", 0.2),
                ("jax_kernel_diagnostic", 0.1),
            )
        ]
        Path(command[command.index("--output") + 1]).write_text(json.dumps(rows))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(matrix.subprocess, "run", run)
    assert matrix.main() == int(enforce)
    report = json.loads((output / "summary.json").read_text())
    case = report["cases"][0]
    assert case["status"] == "completed"
    assert len(case["measurements"]) == 15
    assert all(
        not row["public_overhead_passed"] and not row["default_route_passed"]
        for row in case["proposed_budgets"]
    )
