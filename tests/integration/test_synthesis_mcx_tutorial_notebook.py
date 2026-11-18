"""Check the native original-size multi-control X workflows."""

import json
from pathlib import Path

import pytest

from examples.synthesis.tutorials.basics.mcx.overview import workflow

pytestmark = pytest.mark.integration


def test_original_depth_and_cx_constrained_circuits() -> None:
    depth_program = workflow.program_14(20, 5)
    assert workflow.resources(depth_program).n_qubits <= 20
    assert len(depth_program.registers["controls"]) == 14
    assert workflow.verify_program(depth_program).error < 3e-11
    candidates, selected = workflow.cx_search()
    assert len(candidates) == 2
    for candidate in candidates:
        assert candidate.resources.n_qubits <= 17
        assert set(candidate.resources.operation_counts) == {"cx", "u3"}
        checked = workflow.verify(candidate.circuit, candidate.logical)
        assert checked.error < 3e-11 and checked.workspace_probability < 1e-21
    assert selected.resources.operation_counts["cx"] == min(
        row.resources.operation_counts["cx"] for row in candidates
    )
    assert selected.resources.operation_counts["cx"] == 384


def test_pinned_boston_device_basis_and_routed_state() -> None:
    snapshot = workflow.load_boston()
    assert snapshot.n_qubits == 156 and len(snapshot.edges) == 176
    case = workflow.boston_circuit()
    assert len(case.physical) == case.resources.n_qubits <= 21
    assert len(case.logical) == 15
    assert len(case.candidate_depths) == 6
    assert case.resources.depth == min(depth for _, depth in case.candidate_depths)
    assert case.resources.operation_counts.keys() <= set(snapshot.basis)
    assert all(
        tuple(sorted(case.physical[i] for i in op.wires)) in snapshot.edges
        for op in case.circuit.instructions
        if op.name == "cz"
    )
    outcome = workflow.verify(case.circuit, case.logical, projective=True)
    assert outcome.error < 3e-11 and outcome.workspace_probability < 1e-21


def test_fifty_controls_full_width_xor_and_clean_trace() -> None:
    for enabled, target in ((0, 0), (49, 0), (50, 0), (50, 1)):
        resources, observed = workflow.check_50(enabled, target)
        assert resources.n_qubits == 64
        assert observed == {(target ^ (enabled == 50),): 1.0}


def test_all_original_workflow_stages_unexecuted() -> None:
    path = (
        Path(__file__).resolve().parents[2]
        / "examples/synthesis/tutorials/basics/mcx/overview/walkthrough.ipynb"
    )
    cells = json.loads(path.read_text())["cells"]
    code = [cell for cell in cells if cell["cell_type"] == "code"]
    assert len(code) == 8
    assert all(
        cell["execution_count"] is None and cell["outputs"] == [] for cell in code
    )
