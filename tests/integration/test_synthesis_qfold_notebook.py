"""Verify the original alanylalanine data, complete walk, and finite counts."""

import json
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from examples.synthesis.applications.chemistry.protein_folding.protein_folding_with_quantum_walk.qfold.overview import (
    reference,
    workflow,
)

pytestmark = pytest.mark.integration


def test_original_input_and_angles() -> None:
    data = workflow.dataset()
    assert data.protein == "alanylalanine"
    assert len(data.energy_differences) == 16
    assert data.move_bits == data.rotation_bits == 1
    assert data.encoding_width == 4
    assert workflow.coin_angles(data) == pytest.approx(
        (4.9944226978996045e-6, 1.3721747807752471)
    )
    assert min(workflow.energy_lookup(data).values()) == -25.86237460092684
    with pytest.raises(ValueError, match="beta"):
        workflow.coin_angles(data, float("nan"))


@pytest.mark.parametrize("beta", [1.0, 0.6])
def test_complete_walk_and_clean_workspace(beta: float) -> None:
    angles = workflow.coin_angles(workflow.dataset(), beta)
    program = workflow.build(angles)
    result = fq.run(program)
    actual = result.statevector().reshape(16, -1)
    expected = reference.complete_state(angles, program.circuit.n_wires)
    assert dict(program.registers) == {"phi": (0,), "psi": (1,)}
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=0)
    assert float(actual[:, 1:].abs().square().sum()) < 1e-22
    assert float(actual.reshape(4, 4, -1)[:, 1:].abs().square().sum()) > 0.01
    marginal = reference.walk_state(angles).abs().square().reshape(2, 2, 4).sum(dim=2)
    assert dict(
        program.probabilities(result, registers=("phi", "psi"))
    ) == pytest.approx(
        {(phi, psi): float(marginal[phi, psi]) for phi in range(2) for psi in range(2)}
    )


def test_finite_counts_and_stage_map() -> None:
    angles = workflow.coin_angles(workflow.dataset())
    counts = workflow.sample(workflow.build(angles))
    assert sum(counts.values()) == 2048
    assert counts.keys() == {(0, 0), (0, 1), (1, 0), (1, 1)}
    ideal = reference.walk_state(angles).abs().square().reshape(2, 2, 4).sum(dim=2)
    assert all(
        abs(count / 2048 - float(ideal[key])) < 0.04 for key, count in counts.items()
    )
    book = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "examples/synthesis/applications/chemistry/protein_folding/protein_folding_with_quantum_walk/qfold/overview/walkthrough.ipynb"
        ).read_text()
    )
    cells = [c for c in book["cells"] if c["cell_type"] == "code"]
    assert [c["metadata"]["source_stage"] for c in cells] == [
        6,
        8,
        9,
        10,
        11,
        12,
        16,
        17,
        19,
        20,
        21,
        22,
        23,
    ]
    assert all(c["execution_count"] is None and not c["outputs"] for c in cells)
