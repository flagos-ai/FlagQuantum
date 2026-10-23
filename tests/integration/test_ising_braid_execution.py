"""Real source, sealed plan, Simulation and terminal observation composition."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from examples.ising_braid_execution import main
from flagquantum._api import plan, run
from flagquantum.core.ising import IsingBraidProgram

pytestmark = pytest.mark.integration


def test_native_user_example_runs_the_complete_path(
    capsys: pytest.CaptureFixture[str],
) -> None:
    main()
    assert capsys.readouterr().out == (
        "Plan schema: 1.1\nQubit count: None\nState shape: (1, 2, 2)\n"
        "P(psi): 0.5\nExact replay: True\nTerminal shots: 2000\n"
        "Caller RNG unchanged: True\n"
    )


def test_sealed_execution_does_not_replan_or_compile() -> None:
    source = IsingBraidProgram().exchange(1)
    selected = plan(source)
    expected = run(selected)
    with (
        patch(
            "flagquantum.runtime._ising_plan.plan_ising",
            side_effect=AssertionError("replanned"),
        ),
        patch(
            "flagquantum.runtime.planner.compile_program",
            side_effect=AssertionError("compiled"),
        ),
    ):
        repeated = run(selected)
    assert expected.state is not None and repeated.state is not None
    assert torch.equal(expected.state, repeated.state)
    assert repeated.plan is selected


def test_native_json_replay_in_another_process_matches_captured_density() -> None:
    selected = plan(IsingBraidProgram().exchange(1).exchange(0, clockwise=True))
    result = run(selected)
    assert result.state is not None
    root = Path(__file__).parents[2]
    code = """import json, sys
import flagquantum as fq
selected = fq.ExecutionPlan.from_json(sys.stdin.read())
result = fq.run(selected)
assert result.state is not None
print(json.dumps({'real': result.state.real.tolist(), 'imag': result.state.imag.tolist(), 'identity': selected.identity}))
"""
    process = subprocess.run(
        [sys.executable, "-c", code],
        input=selected.to_json(),
        capture_output=True,
        text=True,
        timeout=60,
        cwd=root,
        env={
            **os.environ,
            "PYTHONPATH": str(root),
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        },
        check=True,
    )
    captured = json.loads(process.stdout)
    replay = torch.complex(
        torch.tensor(captured["real"], dtype=torch.float64),
        torch.tensor(captured["imag"], dtype=torch.float64),
    )
    assert captured["identity"] == selected.identity
    assert torch.equal(replay, result.state)
