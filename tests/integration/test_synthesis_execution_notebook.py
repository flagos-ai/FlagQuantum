"""Verify finite counts, coherent state, and calibration-derived noise."""

import json
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from examples.synthesis.tutorials.basics.quantum_programming.execution_tutorial.overview import (
    reference,
    workflow,
)

pytestmark = pytest.mark.integration


def test_square_complete_state_and_finite_counts() -> None:
    program = workflow.build_square()
    state = fq.run(program)
    assert program.circuit.n_wires == 10
    assert tuple(map(len, (program.registers["x"], program.registers["y"]))) == (3, 6)
    torch.testing.assert_close(
        state.statevector().reshape(-1), reference.square_state(), atol=1e-12, rtol=0
    )
    expected = {(x, x * x + 1) for x in range(8)}
    probabilities = program.probabilities(state, registers=("x", "y"))
    assert probabilities.keys() == expected
    assert all(p == pytest.approx(0.125) for p in probabilities.values())
    for shots in (2048, 10_000):
        counts = workflow.square_counts(program, shots)
        assert sum(counts.values()) == shots
        assert counts.keys() == expected
        assert all(abs(count / shots - 0.125) < 0.04 for count in counts.values())


def test_ghz_exact_and_mps() -> None:
    ghz = workflow.build_ghz()
    assert ghz.circuit.n_wires == 3
    torch.testing.assert_close(
        workflow.ideal_state(ghz), reference.ghz_state(), atol=1e-12, rtol=0
    )
    counts = workflow.mps_counts(ghz)
    assert counts.keys() == {"000", "111"}
    assert sum(counts.values()) == 2048
    assert "pytorch" in workflow.backend_names()


def test_noisy_density_and_asymmetric_readout() -> None:
    profile = workflow.calibration()
    assert profile.device == "ibm_pittsburgh"
    assert profile.release == "0.49.0"
    circuit = workflow.native_basis(workflow.build_ghz(), profile)
    assert {op.name for op in circuit.instructions} == {"rz", "sx", "cz"}
    assert [
        tuple(sorted(op.wires)) for op in circuit.instructions if op.name == "cz"
    ] == [(1, 2), (0, 1)]
    counts, probabilities, density = workflow.noisy_ghz(circuit, profile)
    expected_density, expected_probabilities = reference.noisy_state(circuit, profile)
    torch.testing.assert_close(density, expected_density, atol=1e-12, rtol=0)
    torch.testing.assert_close(
        probabilities, expected_probabilities, atol=1e-12, rtol=0
    )
    assert probabilities[4] != probabilities[1]
    assert sum(counts.values()) == 2048
    assert 0 < 2048 - counts["000"] - counts["111"] < 150


def test_original_stage_scale() -> None:
    book = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "examples/synthesis/tutorials/basics/quantum_programming/execution_tutorial/overview/walkthrough.ipynb"
        ).read_text()
    )
    stages = [c for c in book["cells"] if c["cell_type"] == "code"]
    assert [c["metadata"]["source_stage"] for c in stages] == [
        2,
        4,
        6,
        9,
        11,
        13,
        15,
        18,
        21,
        23,
        24,
        25,
    ]
    assert all(c["execution_count"] is None and not c["outputs"] for c in stages)
