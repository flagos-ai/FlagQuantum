"""Check the complete plasma projection, phase, bit order and work semantics."""

import json
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from examples.synthesis.applications.plasma.vlasov_ampere.vlasov_ampere_qiskit.overview import (
    physics,
    quantum,
    workflow,
)

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("seed", [17, 29])
def test_original_full_grid_complex_projection(seed: int) -> None:
    result = workflow.run(seed)
    assert len(result.expected) == len(result.projected) == 128
    assert result.l2_error < 1e-12
    assert result.work_leak < 1e-24
    assert (result.block_resources.n_qubits, result.qsvt_resources.n_qubits) == (17, 18)
    assert result.block_resources.n_operations == 51
    assert result.qsvt_resources.n_operations == 108
    assert result.block_resources.basis == "static_instruction_sequence"
    assert result.qsvt_resources.max_operation_width == 9


def test_independent_shift_columns_real_preparations_and_rejections() -> None:
    for amount in (-1, 1):
        permutation = quantum.shift(4, amount)
        for column in range(16):
            expected = torch.zeros(16, dtype=torch.complex128)
            expected[(column + amount) % 16] = 1
            torch.testing.assert_close(permutation[:, column], expected, atol=0, rtol=0)
    for vector in (
        quantum.BC / quantum.BC_NORM,
        -quantum.FORCE / quantum.FORCE_NORM,
        physics.initial_state(31),
    ):
        unitary = quantum.prepare(vector)
        torch.testing.assert_close(
            unitary[:, 0], vector.to(torch.complex128), atol=1e-12, rtol=0
        )
        torch.testing.assert_close(
            unitary.mH @ unitary,
            torch.eye(vector.numel(), dtype=torch.complex128),
            atol=1e-12,
            rtol=0,
        )
    with pytest.raises(ValueError, match="normalized"):
        quantum.prepare(torch.zeros(4))
    with pytest.raises(ValueError, match="real preparation"):
        quantum.prepare(torch.tensor([0.0, 1j], dtype=torch.complex128))
    with pytest.raises(ValueError, match="disjoint"):
        quantum.Builder(fq.Circuit(2)).flip(0, ((0, 1),))
    with pytest.raises(ValueError, match="finite"):
        quantum.qsvt_step(float("nan"), 0.2)


def test_full_demonstration_pair_unitarity_and_zero_angle_roundtrip() -> None:
    actual = fq.run(quantum.qsvt_step()).statevector().reshape(2, -1)
    assert float(actual.norm()) == pytest.approx(1, abs=1e-12)
    assert float(actual[1].abs().square().sum()) < 1e-24
    restored = fq.run(quantum.qsvt_step(0.0, 0.0)).statevector().reshape(-1)
    expected = torch.zeros_like(restored)
    expected[0] = 1
    torch.testing.assert_close(restored, expected, atol=1e-12, rtol=0)


def test_all_original_stages_and_local_randomness() -> None:
    before = torch.random.get_rng_state().clone()
    first = physics.initial_state(17)
    torch.testing.assert_close(first, physics.initial_state(17), atol=0, rtol=0)
    assert torch.equal(before, torch.random.get_rng_state())
    assert not torch.equal(first, physics.initial_state(29))
    book = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "examples/synthesis/applications/plasma/vlasov_ampere/vlasov_ampere_qiskit/overview/walkthrough.ipynb"
        ).read_text()
    )
    cells = [cell for cell in book["cells"] if cell["cell_type"] == "code"]
    assert [cell["metadata"]["source_stage"] for cell in cells] == [
        1,
        2,
        4,
        5,
        7,
        10,
        11,
        13,
        15,
        16,
        17,
        18,
        21,
        22,
        23,
        24,
        25,
        26,
        27,
        29,
        31,
        33,
        34,
        35,
        36,
    ]
    assert all(
        cell["execution_count"] is None and not cell["outputs"] for cell in cells
    )
