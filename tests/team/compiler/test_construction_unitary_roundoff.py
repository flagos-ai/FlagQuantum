"""Givens elimination does not amplify unit-scale floating-point residuals."""

import math

import pytest
import torch

import flagquantum as fq
from flagquantum.compiler._construction.semantics.unitary import unitary_instructions
from flagquantum.compiler._hybrid.model import Operation
from flagquantum.core.ir import CircuitIR, Instruction

pytestmark = pytest.mark.unit


def _near_identity(factor: float) -> torch.Tensor:
    matrix = torch.eye(8, dtype=torch.complex128)
    angle = factor * math.ulp(1.0)
    rotation = torch.tensor(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]],
        dtype=torch.complex128,
    )
    for row in (6, 7):
        matrix[[0, row]] = rotation @ matrix[[0, row]]
    return matrix


def test_sub_epsilon_residuals_do_not_create_finite_givens_rotations() -> None:
    matrix = _near_identity(0.75)
    gates = unitary_instructions(matrix, (0, 1, 2), Operation("qreg.unitary"))
    assert gates == ()
    torch.testing.assert_close(
        matrix, torch.eye(8, dtype=torch.complex128), atol=math.ulp(1.0), rtol=0
    )


@pytest.mark.parametrize("factor", [0.75, 1.5, 1_000_000.0])
@pytest.mark.parametrize("phase", [0.0, 0.37])
def test_roundoff_boundary_retains_columns_and_controlled_absolute_phase(
    factor: float, phase: float
) -> None:
    matrix = _near_identity(factor) * complex(math.cos(phase), math.sin(phase))
    gates = unitary_instructions(
        matrix, (1, 2, 3), Operation("qreg.unitary"), controls=(0,)
    )
    for basis in range(8):
        preparation = (
            Instruction("h", (0,)),
            *(Instruction("x", (3 - bit,)) for bit in range(3) if basis & (1 << bit)),
        )
        circuit = fq.Circuit.from_ir(
            CircuitIR(4, (*preparation, *gates)), dtype=torch.complex128
        )
        actual = fq.run(circuit).statevector().flatten()
        expected = torch.zeros(16, dtype=torch.complex128)
        expected[basis] = 1 / math.sqrt(2)
        expected[8:] = matrix[:, basis] / math.sqrt(2)
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=0)
    if factor == 1_000_000.0:
        assert abs(matrix[7, 0]) > 1e-10
        assert gates
