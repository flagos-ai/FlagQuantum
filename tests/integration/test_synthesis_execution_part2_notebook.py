"""Verify the full source sweep, persistent job readback and Bell objective."""

import json
import math
from pathlib import Path

import pytest

from examples.synthesis.tutorials.basics.quantum_programming.execution_tutorial_part2.overview import (
    jobs,
    workflow,
)

pytestmark = pytest.mark.integration


def test_runtime_binding_full_original_sweep() -> None:
    circuit = workflow.template()
    rows = workflow.sweep(circuit)
    assert circuit.parameter_names == ("angle",)
    assert len(rows) == 50
    assert rows[0].angle == 0 and rows[-1].angle == math.tau
    assert all(sum(row.counts.values()) == 2048 for row in rows)
    assert all(set(row.counts) <= {"00", "11"} for row in rows)
    assert all(row.projector_shots == 2048 for row in rows)
    assert max(row.state_error for row in rows) < 3e-12
    with pytest.raises(ValueError, match="finite"):
        workflow.bind(circuit, math.nan)


def test_actual_jobs_and_durable_readback(tmp_path: Path) -> None:
    created = jobs.submit_both(workflow.template(), math.pi / 5, tmp_path)
    assert len(set(created)) == 2
    measured, observed = (jobs.lookup(tmp_path, key) for key in created)
    assert measured.counts is not None and sum(measured.counts.values()) == 2048
    assert observed.expectation is not None and observed.exact_expectation is not None
    assert (observed.expectation * 2048).is_integer()
    assert math.isclose(
        observed.exact_expectation, (1 - math.sin(math.pi / 5)) / 2, abs_tol=3e-12
    )
    (tmp_path / f"{created[0]}.json").write_text("{}")
    with pytest.raises(ValueError, match="differs from the job ID"):
        jobs.lookup(tmp_path, created[0])
    with pytest.raises(ValueError, match="SHA-256"):
        jobs.lookup(tmp_path, "../")


def test_original_bell_query_budget_and_observed_fidelity() -> None:
    result = workflow.optimize()
    assert 0 < len(result.trials) <= 200
    assert result.trials[0].angles == (0.0, 0.0)
    assert max(trial.fidelity for trial in result.trials) > 0.999999
    assert result.state_error < 3e-12


def test_all_original_code_stages_unexecuted() -> None:
    path = (
        Path(__file__).resolve().parents[2]
        / "examples/synthesis/tutorials/basics/quantum_programming/execution_tutorial_part2/overview/walkthrough.ipynb"
    )
    data = json.loads(path.read_text())
    cells = [cell for cell in data["cells"] if cell["cell_type"] == "code"]
    assert len(cells) == 16
    assert all(
        cell["execution_count"] is None and cell["outputs"] == [] for cell in cells
    )
