"""Verify absolute states, source discretization, and controlled rotations."""

import json
import math
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from examples.synthesis.community.hackathons.qinnovision_2025.advection_equation.second_place.overview import (
    physics,
    quantum,
    workflow,
)

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("time", [0.5, 1.0])
def test_all_complex_amplitudes_at_original_nonzero_times(time: float) -> None:
    actual = quantum.state(quantum.build(time=time))
    expected = physics.reference(7, time, 1 / 32)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=0)
    assert float(actual.abs().square().sum()) == pytest.approx(1, abs=1e-12)


def test_controlled_rotation_complete_basis_and_named_resources() -> None:
    for width in range(2, 8):
        circuit = fq.Circuit(width, dtype=torch.complex128)
        quantum.controlled_ry(circuit, tuple(range(width - 1)), width - 1, math.pi / 4)
        for word in range(1 << width):
            initial = torch.zeros(1 << width, dtype=torch.complex128)
            initial[word] = 1
            actual = (
                fq.Circuit(width, inputs=initial, dtype=torch.complex128)
                .compose(circuit)
                .state()
                .reshape(-1)
            )
            expected = initial.clone()
            if word >> 1 == (1 << (width - 1)) - 1:
                c, s = math.cos(math.pi / 8), math.sin(math.pi / 8)
                expected[word] = c
                expected[word ^ 1] = -s if word & 1 else s
            torch.testing.assert_close(actual, expected, atol=1e-12, rtol=0)
        assert quantum.metrics(circuit).operation_counts["cx"] == 1 << (width - 1)


def test_classical_boundary_and_original_scale() -> None:
    initial = (physics.grid() > 1) & (physics.grid() < 2)
    assert int(initial.sum()) == 32
    values = physics.classical()
    assert values.numel() == 128
    assert values[0] == values[-1] == 0
    assert float(values.abs().max()) == pytest.approx(1.32968199685, abs=1e-10)
    assert workflow.theoretical_cx(7, 1) == 7808
    metrics = quantum.metrics(quantum.build())
    assert (
        metrics.n_qubits,
        metrics.n_operations,
        metrics.operation_counts["cx"],
        metrics.depth,
    ) == (7, 9510, 5376, 9505)


def test_original_stage_map_and_static_refusals() -> None:
    book = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "examples/synthesis/community/hackathons/qinnovision_2025/advection_equation/second_place/overview/walkthrough.ipynb"
        ).read_text()
    )
    cells = [c for c in book["cells"] if c["cell_type"] == "code"]
    assert [c["metadata"]["source_stage"] for c in cells] == [
        2,
        4,
        7,
        8,
        10,
        11,
        25,
        27,
        29,
        31,
        33,
        35,
        37,
        39,
        41,
        43,
        44,
        45,
        47,
        49,
    ]
    assert all(c["execution_count"] is None and not c["outputs"] for c in cells)
    with pytest.raises(ValueError, match="seven"):
        quantum.build(width=8)
    with pytest.raises(ValueError, match="sixty-four"):
        quantum.build(time=3)
