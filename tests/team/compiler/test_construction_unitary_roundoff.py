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


@pytest.mark.parametrize("control_count", [1, 2, 4])
def test_controlled_block_keeps_absolute_phase_with_two_clean_work_bits(
    control_count: int,
) -> None:
    """Phase compensation survives both full and borrowed-control predicates."""
    import cmath

    from flagquantum.compiler._construction.semantics.unitary import _controlled_block

    target = control_count
    gates = _controlled_block(
        (Instruction("u3", (target,), {"theta": 0.37, "phi": -0.42, "lbd": 0.73}),),
        tuple(range(control_count)),
        (target + 1, target + 2),
        global_phase=-0.31,
    )
    cosine, sine = math.cos(0.37 / 2), math.sin(0.37 / 2)
    matrix = torch.tensor(
        [
            [cosine, -cmath.exp(0.73j) * sine],
            [cmath.exp(-0.42j) * sine, cmath.exp(0.31j) * cosine],
        ],
        dtype=torch.complex128,
    ) * cmath.exp(-0.31j)
    n_qubits = control_count + 3
    logical_size = 2 ** (control_count + 1)
    for column in range(logical_size):
        preparation = tuple(
            Instruction("x", (bit,))
            for bit in range(target + 1)
            if column & (1 << (target - bit))
        )
        circuit = fq.Circuit.from_ir(
            CircuitIR(n_qubits, (*preparation, *gates)), dtype=torch.complex128
        )
        actual = fq.run(circuit).statevector().flatten()
        expected = torch.zeros(2**n_qubits, dtype=torch.complex128)
        if column // 2 == 2**control_count - 1:
            expected[(logical_size - 2) * 4] = matrix[0, column % 2]
            expected[(logical_size - 1) * 4] = matrix[1, column % 2]
        else:
            expected[column * 4] = 1
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=0)
