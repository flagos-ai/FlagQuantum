"""Named CX/U conversion retains absolute phase and trainable parameters."""

import pytest
import torch

from flagquantum.compiler.basis_conversion import BasisConversionError, convert_basis
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.simulation.unitary import get_unitary

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "opcode,params",
    [
        *(
            (name, {})
            for name in ("i", "x", "y", "z", "h", "s", "sdg", "t", "tdg", "sx", "sxdg")
        ),
        *(
            (name, {"theta": angle})
            for name in ("rx", "ry", "rz", "phase", "u1")
            for angle in (0.73, -1.19)
        ),
        ("u2", {"phi": 0.7, "lbd": -1.3}),
        ("u3", {"theta": -0.4, "phi": 0.7, "lbd": -1.3}),
    ],
)
def test_absolute_unitary_in_u_basis(opcode: str, params: dict[str, float]) -> None:
    source = CircuitIR(
        1, (Instruction(opcode, (0,), params=params),), dtype="complex128"
    )
    converted = convert_basis(source, gates=("u",))
    assert {op.name for op in converted.program.instructions} == {"u3"}
    torch.testing.assert_close(
        get_unitary(converted.program), get_unitary(source), atol=1e-12, rtol=0
    )


@pytest.mark.parametrize("name", ("ccx", "cswap"))
def test_recursive_three_wire_conversion_is_absolute(name: str) -> None:
    source = CircuitIR(3, (Instruction(name, (0, 1, 2)),), dtype="complex128")
    converted = convert_basis(source, gates=("cx", "u3"))
    assert {op.name for op in converted.program.instructions} <= {"u3", "cx"}
    torch.testing.assert_close(
        get_unitary(converted.program), get_unitary(source), atol=1e-12, rtol=0
    )
    with pytest.raises(BasisConversionError, match="max_added_operations"):
        convert_basis(source, gates=("cx", "u3"), max_added_operations=1)


def test_rz_scalar_compensation_preserves_gradient() -> None:
    angle = torch.tensor(0.73, dtype=torch.float64, requires_grad=True)
    source = CircuitIR(
        1, (Instruction("rz", (0,), params={"theta": angle}),), dtype="complex128"
    )
    converted = convert_basis(source, gates=("u3",))
    actual = get_unitary(converted.program)
    expected = get_unitary(source)
    weights = torch.tensor([[0.3, 0.7], [-0.4, 1.9]], dtype=torch.float64)
    actual_loss = (actual.real * weights + actual.imag * weights.T).sum()
    expected_loss = (expected.real * weights + expected.imag * weights.T).sum()
    actual_gradient = torch.autograd.grad(actual_loss, angle, retain_graph=True)[0]
    expected_gradient = torch.autograd.grad(expected_loss, angle)[0]
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=0)
    torch.testing.assert_close(actual_gradient, expected_gradient, atol=1e-12, rtol=0)


@pytest.mark.parametrize("mixed_vocabulary", [False, True])
def test_u3_vocabulary_preservation_follows_the_complete_program(
    mixed_vocabulary: bool,
) -> None:
    from flagquantum.compiler.optimization.one_qubit_optimization import (
        collapse_one_qubit_runs,
    )

    instructions = (
        Instruction("u3", (0,), params={"theta": 0.0, "phi": 0.0, "lbd": 0.3}),
        Instruction("u3", (0,), params={"theta": 0.0, "phi": 0.0, "lbd": 0.4}),
    )
    if mixed_vocabulary:
        instructions += (Instruction("h", (1,)),)
    instructions += (Instruction("cx", (0, 1)),)
    source = CircuitIR(2, instructions, dtype="complex128")
    folded = collapse_one_qubit_runs(source)
    assert len(folded.instructions) == len(instructions) - 1
    assert folded.instructions[0].name == ("phase" if mixed_vocabulary else "u3")
    torch.testing.assert_close(
        get_unitary(folded), get_unitary(source), atol=1e-12, rtol=0
    )
