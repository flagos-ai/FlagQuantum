"""The documented split-FP32 workflow learns an analytic quantum objective."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from examples.ascend_910c_training import train_rotation

pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[1]


def test_explicit_parameter_shift_sgd_matches_analytic_solution() -> None:
    result = train_rotation("cpu")
    assert result.initial_expectation > 0.96
    assert result.final_expectation < -0.9999
    assert result.maximum_gradient_absolute_error < 2e-5
    assert result.state_device == result.gradient_device == "cpu"


@pytest.mark.gpu
@pytest.mark.distributed_accel
def test_documented_training_runs_on_ascend_910c() -> None:
    if os.environ.get("FLAGQUANTUM_TEST_ASCEND_910C") != "1":
        pytest.skip("set FLAGQUANTUM_TEST_ASCEND_910C=1 on the documented 910C stack")
    completed = subprocess.run(
        [sys.executable, "examples/ascend_910c_training.py", "--device", "flagos:0"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
        timeout=300,
    )
    result = json.loads(completed.stdout.strip().splitlines()[-1])
    assert result["final_expectation"] < -0.9999
    assert result["maximum_gradient_absolute_error"] < 2e-5
    assert result["state_device"] == result["gradient_device"] == "flagos:0"
