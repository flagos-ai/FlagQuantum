"""Matrix lowering retains absolute phase, controls and clean workspace."""

import cmath
import math

import pytest
import torch

import flagquantum as fq
from flagquantum.compiler._construction.semantics.unitary import unitary_instructions
from flagquantum.compiler.frontends._structured.model import Operation
from flagquantum.compiler.frontends._structured.verifier import HybridVerificationError
from flagquantum.core.ir import CircuitIR, Instruction

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "width, controlled, limited_workspace",
    [
        (3, False, False),
        (3, True, False),
        (4, False, False),
        (4, True, False),
        (4, True, True),
    ],
)
@pytest.mark.parametrize("family", ["fourier", "random"])
def test_complete_matrix_preserves_control_phase_and_clean_workspace(
    width: int, controlled: bool, family: str, limited_workspace: bool
) -> None:
    size = 1 << width
    if family == "fourier":
        indices = torch.arange(size, dtype=torch.float64)
        matrix = torch.exp(2j * math.pi * torch.outer(indices, indices) / size)
        matrix /= math.sqrt(size)
    else:
        generator = torch.Generator().manual_seed(719 + width)
        raw = torch.randn(size, size, dtype=torch.complex128, generator=generator)
        matrix, _ = torch.linalg.qr(raw)
    matrix *= cmath.exp(0.37j)
    original = matrix.clone()
    control = (width,) if controlled else ()
    scratch_size = width - 2 + len(control) if width >= 4 else 0
    if limited_workspace:
        scratch_size -= 1
    scratch = tuple(range(width + len(control), width + len(control) + scratch_size))
    total = width + len(control) + len(scratch)
    gates = unitary_instructions(
        matrix, tuple(range(width)), Operation("qreg.unitary"), control, scratch
    )
    control_values = (0, 1) if controlled else (0,)
    for enabled in control_values:
        for column in range(size):
            word = column << (total - width)
            if enabled:
                word |= 1 << (total - width - 1)
            preparation = tuple(
                Instruction("x", (wire,))
                for wire in range(total)
                if word & (1 << (total - wire - 1))
            )
            ir = CircuitIR(total, instructions=(*preparation, *gates))
            actual = fq.Circuit.from_ir(ir, dtype=torch.complex128).state().flatten()
            expected = torch.zeros(1 << total, dtype=torch.complex128)
            if controlled and not enabled:
                expected[word] = 1
            else:
                offset = (1 << (total - width - 1)) if enabled else 0
                for row in range(size):
                    expected[(row << (total - width)) | offset] = matrix[row, column]
            torch.testing.assert_close(actual, expected, atol=1e-12, rtol=0)
    torch.testing.assert_close(matrix, original, atol=0, rtol=0)


def test_matrix_reconstruction_rejects_a_nonunitary_input() -> None:
    matrix = torch.eye(8, dtype=torch.complex128)
    matrix[0, 0] = 1.0001
    with pytest.raises(HybridVerificationError, match="Givens reconstruction"):
        unitary_instructions(matrix, (0, 1, 2), Operation("qreg.unitary"))
