"""PR coverage retains quantum checks and separates full optimizer acceptance."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[2]
FILES = (
    "tests/integration/test_synthesis_induced_color_notebook.py",
    "tests/integration/test_synthesis_min_coloring_notebook.py",
    "tests/integration/test_synthesis_coloring_pr_workflows.py",
)
FULL_NAMES = (
    "test_actual_fifty_iteration_optimization_and_sampling",
    "test_actual_hundred_iteration_optimization_and_sampling",
)


@pytest.mark.parametrize("full", (False, True))
def test_real_collection_separates_full_acceptance_and_keeps_quantum_checks(full):
    selection = "integration" if full else "integration and not slow"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            *FILES,
            "--collect-only",
            "-q",
            "-o",
            "addopts=",
            "-m",
            selection,
        ],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT)},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    nodes = [line for line in result.stdout.splitlines() if "::test_" in line]
    for name in FULL_NAMES:
        assert any(name in node for node in nodes) is full
    assert (
        sum("test_complete_complex_states_and_named_readout[" in n for n in nodes) == 4
    )
    assert sum("test_resource_budget_refusals[" in n for n in nodes) == 8
    assert (
        sum("test_two_iteration_trace_sampling_and_feasibility[" in n for n in nodes)
        == 2
    )
    assert any("test_all4096_absolute_cost_layer_amplitudes" in n for n in nodes)
    assert any("test_all32768_absolute_cost_layer_amplitudes" in n for n in nodes)
