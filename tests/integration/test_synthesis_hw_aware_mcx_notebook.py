"""Verify the original dimensions and independent hardware-aware MCX checks."""

import json
from pathlib import Path

import pytest

from examples.synthesis.community.basic_examples.hw_aware_synthesis.hw_aware_synthesis.overview import (
    workflow,
)

pytestmark = pytest.mark.integration


def test_original_depth_constrained_mcx() -> None:
    program = workflow.build_program(10)
    logical = (*program.registers["controls"], *program.registers["target"])
    assert len(logical) == 11
    assert program.circuit.n_wires <= 20
    checked = workflow.verify(program.circuit, logical)
    assert checked.state_error < 2e-11
    assert checked.workspace_probability < 1e-22


def test_exact_grid_and_star_edges_and_minimum_emitted_cx() -> None:
    assert set(workflow.GRID_EDGES) == (
        {(6 * row + col, 6 * row + col + 1) for row in range(3) for col in range(5)}
        | {(6 * row + col, 6 * (row + 1) + col) for row in range(2) for col in range(6)}
    )
    assert tuple((0, i) for i in range(1, 18)) == workflow.STAR_EDGES
    program = workflow.build_program(12)
    for match in (workflow.route(program, "grid"), workflow.route(program, "star")):
        coupling = workflow.coupling_map(match.topology)
        assert len(match.candidates) == 6
        assert match.selected.resources.n_qubits == 18
        assert len(match.selected.logical_wires) == 13
        for row in match.candidates:
            assert set(row.resources.operation_counts) == {"cx", "u3"}
            assert all(
                coupling.has_edge(*op.wires)
                for op in row.circuit.instructions
                if op.name == "cx"
            )
        assert match.selected.resources.operation_counts["cx"] == min(
            row.resources.operation_counts["cx"] for row in match.candidates
        )
        result = workflow.verify(match.selected.circuit, match.selected.logical_wires)
        assert result.state_error < 2e-11
        assert result.workspace_probability < 1e-22


def test_walkthrough_preserves_original_stages_unexecuted() -> None:
    path = (
        Path(__file__).resolve().parents[2]
        / "examples/synthesis/community/basic_examples/hw_aware_synthesis/hw_aware_synthesis/overview/walkthrough.ipynb"
    )
    cells = json.loads(path.read_text())["cells"]
    code = [cell for cell in cells if cell["cell_type"] == "code"]
    assert len(code) == 18
    assert all(
        cell["execution_count"] is None and cell["outputs"] == [] for cell in code
    )
