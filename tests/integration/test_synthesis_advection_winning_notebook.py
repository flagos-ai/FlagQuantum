"""Verify fresh operators, original discretization and complete native quantum states."""

import json
import math
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from examples.synthesis.community.hackathons.qinnovision_2025.advection_equation.winning.overview import (
    physics,
    quantum,
    reference,
)
from flagquantum import synth

pytestmark = pytest.mark.integration


def test_original_grid_and_precision_preserving_initial_state() -> None:
    p = physics.create()
    assert p.initial.numel() == 128
    assert int(p.initial.count_nonzero()) == 26
    assert p.norm == pytest.approx(math.sqrt(26))
    assert physics.rk4(p).shape == (128, 28)
    assert int(1 / 0.038) == 26
    program = synth.synthesize(
        quantum.initial_state,
        classical_inputs={"bits": 7, "weights": p.initial.tolist()},
        max_qubits=7,
        max_workspace=0,
        memory_limit_bytes=1024**2,
    )
    state = fq.run(program).statevector().reshape(-1)
    torch.testing.assert_close(
        state, p.initial.to(torch.complex128), atol=1e-12, rtol=0
    )


@pytest.mark.parametrize("variant", ["proposed", "custom", "trotter"])
def test_complete_original_complex_state(variant: quantum.Variant) -> None:
    p = physics.create()
    terms = quantum.paulis(p)
    assert len(terms) == 127
    torch.testing.assert_close(
        reference.reassemble(p, terms), -1j * p.derivative, atol=1e-12, rtol=0
    )
    circuit = quantum.build(p, variant)
    actual = quantum.state(circuit)
    expected = (
        reference.trotter(p, terms) if variant == "trotter" else reference.proposed(p)
    )
    assert circuit.to_ir().n_wires == 7
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=0)
    assert abs(float(actual.norm()) - 1) < 3e-12


def test_custom_controlled_rz_all_basis_columns() -> None:
    for width in range(2, 8):
        controlled = fq.Circuit(width, dtype=torch.complex128)
        quantum.custom_rz(controlled, tuple(range(width - 1)), width - 1, 0.73)
        for word in range(1 << width):
            initial = torch.zeros(1 << width, dtype=torch.complex128)
            initial[word] = 1
            actual = (
                fq.Circuit(width, inputs=initial, dtype=torch.complex128)
                .compose(controlled)
                .state()
                .reshape(-1)
            )
            expected = initial.clone()
            if word >> 1 == (1 << (width - 1)) - 1:
                expected[word] *= torch.exp(
                    torch.tensor(
                        (0.5j if word & 1 else -0.5j) * 0.73, dtype=torch.complex128
                    )
                )
            torch.testing.assert_close(actual, expected, atol=1e-12, rtol=0)


def test_original_stage_map_and_disabled_setup() -> None:
    book = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "examples/synthesis/community/hackathons/qinnovision_2025/advection_equation/winning/overview/walkthrough.ipynb"
        ).read_text()
    )
    stages = [c for c in book["cells"] if c["cell_type"] == "code"]
    assert len(stages) == 51
    assert [c["metadata"]["source_stage"] for c in stages][:3] == [5, 6, 7]
    assert stages[-1]["metadata"]["source_stage"] == 110
    assert all(c["execution_count"] is None and not c["outputs"] for c in stages)
