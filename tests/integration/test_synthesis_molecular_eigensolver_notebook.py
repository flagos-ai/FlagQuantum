"""Check both native H2 ansatz paths and actual finite-shot estimation."""

import json
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from examples.synthesis.applications.chemistry.molecule_eigensolver.overview import (
    workflow,
)

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("scale", [0.07, -0.031])
def test_hardware_ansatz_matches_independent_circuit(scale: float) -> None:
    angles = tuple(scale * (i + 1) for i in range(12))
    circuit = fq.Circuit(4, dtype=torch.complex128).x(3).x(1)
    for layer in range(3):
        for wire in range(4):
            circuit.x(3 - wire).ry(3 - wire, theta=angles[layer * 4 + wire])
        for wire in range(3):
            circuit.cx(3 - wire, 2 - wire)
    actual = fq.run(workflow.build_hardware(angles)).statevector().flatten()
    expected = fq.run(circuit).statevector().flatten()
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=0)


def test_original_choice_and_finite_shot_result() -> None:
    result = workflow.run_workflow()
    assert len(workflow.GEOMETRIES) == 5
    assert workflow.DISTANCE == 0.735
    assert result.hardware.n_qubits == 4
    assert result.tapered.n_qubits == 1
    assert result.converged and 1 <= result.iterations <= 200
    assert result.fci == pytest.approx(-1.13730603575, abs=1e-8)
    assert result.exact_expectation == pytest.approx(result.fci, abs=1e-4, rel=0)
    assert abs(result.sampled - result.exact_expectation) < 0.003
    assert sum(abs(t.coefficient) > 1e-12 for t in result.full.terms) == 15
    circuit = fq.Circuit(1, dtype=torch.complex128).x(0).ry(0, theta=-result.angle)
    a, b = fq.run(circuit).statevector().flatten()
    offset, x, z = result.reduced
    energy = (
        offset
        + 2 * x * float((a.conj() * b).real)
        + z * float(a.abs().square() - b.abs().square())
    )
    assert energy == pytest.approx(result.exact_expectation, abs=1e-12, rel=0)


def test_all_source_stages() -> None:
    directory = Path(workflow.__file__).parent
    notebook = json.loads((directory / "walkthrough.ipynb").read_text())
    provenance = json.loads((directory / "provenance.json").read_text())
    stages = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert len(stages) == len(provenance["reviewed_code_cells"]) == 9
    assert all(cell["execution_count"] is None for cell in stages)
