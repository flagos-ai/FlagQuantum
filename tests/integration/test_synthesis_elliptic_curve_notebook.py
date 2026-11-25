"""Verify the original-size local elliptic-curve discrete logarithm notebook."""

import json
from pathlib import Path

import pytest

from examples.synthesis.algorithms.number_theory_and_cryptography.elliptic_curves.elliptic_curve_discrete_log.overview import (
    workflow,
)
from examples.synthesis.algorithms.number_theory_and_cryptography.elliptic_curves.elliptic_curve_discrete_log.overview.arithmetic import (
    circuit_from_gates,
)
from examples.synthesis.algorithms.number_theory_and_cryptography.elliptic_curves.elliptic_curve_discrete_log.overview.kaliski import (
    InverseWorkspace,
    inverse,
)

pytestmark = pytest.mark.integration


def test_reversible_inverse_is_xor_and_cleans_every_workspace_bit() -> None:
    source, target = (0, 1, 2), (3, 4, 5)
    workspace = InverseWorkspace.starting_at(6)
    gates = inverse(source, target, workspace).lower((37, 38, 39, 40))
    circuit = circuit_from_gates(41, gates)
    outputs = workflow.trace_basis(circuit.to_ir().instructions, 41, tuple(range(64)))
    expected = tuple(
        value | ((prior ^ (pow(value, -1, 7) if value % 7 else 0)) << 3)
        for prior in range(8)
        for value in range(8)
    )
    assert outputs == expected


def test_two_sampled_paths_and_independent_complex_reference() -> None:
    point, point_cost = workflow.sampled_point_add()
    rows, period_cost, error = workflow.sampled_period_finding()
    assert point == (4, 5)
    assert (point_cost.n_qubits, period_cost.n_qubits) == (9, 15)
    assert sum(row.count for row in rows) == 2048
    assert [row.logarithm() for row in rows if row.logarithm() is not None][:10] == [
        4
    ] * 10
    assert error < 3e-12


def test_arithmetic_period_finding_all_branches_and_fourier_suffix() -> None:
    point = workflow.point_add_circuit()
    assert workflow.checked_point_add(point) == (4, 5)
    assert len(point) > 100_000
    del point
    period = workflow.period_finding_circuit()
    assert period.n_qubits == 54
    assert len(period) == 796_852
    assert workflow.checked_period_finding(period) < 3e-12


def test_walkthrough_retains_all_source_stages_unexecuted() -> None:
    path = (
        Path(__file__).resolve().parents[2]
        / "examples/synthesis/algorithms/number_theory_and_cryptography/elliptic_curves/elliptic_curve_discrete_log/overview/walkthrough.ipynb"
    )
    cells = json.loads(path.read_text())["cells"]
    code = [cell for cell in cells if cell["cell_type"] == "code"]
    assert len(code) == 21
    assert all(
        cell["execution_count"] is None and cell["outputs"] == [] for cell in code
    )
