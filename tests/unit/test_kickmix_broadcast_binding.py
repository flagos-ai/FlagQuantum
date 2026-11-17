"""Verify locked scalar, array and basis-aware broadcast admission."""

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.compiler.artifacts.operand_identity import OperandIdentity, OperandKind
from flagquantum.compiler.formats.kmx import encode_kmx
from flagquantum.compiler.lowering.reversible_phase import lower_reversible_phase
from flagquantum.compiler.registers.finite_field.coefficients import (
    ClassicalFieldCoefficient,
)
from flagquantum.compiler.registers.layout.operand_word import bind_operand_array
from flagquantum.compiler.registers.primitives.broadcast_binding import (
    broadcast_cx_ir,
    broadcast_cz_ir,
    broadcast_operand_words,
)
from flagquantum.core.classical_dataflow import ClassicalXor
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic


@pytest.mark.parametrize(
    "operands,physical",
    [
        ((0, 4), ((0, 4),)),
        ((0, (7, 6, 5, 4)), ((0, 7), (0, 6), (0, 5), (0, 4))),
        (((3, 2, 1, 0), 4), ((3, 4), (2, 4), (1, 4), (0, 4))),
        ((0, ()), ()),
        (((0, 2), (7, 5)), ((0, 7), (2, 5))),
    ],
)
def test_locked_cx_shapes_through_append(
    operands: tuple[int | tuple[int, ...], int | tuple[int, ...]],
    physical: tuple[tuple[int, int], ...],
) -> None:
    control, target = (
        bind_operand_array(port) if isinstance(port, tuple) else port
        for port in operands
    )
    gates = broadcast_cx_ir(control, target)
    labels = torch.arange(256)
    basis = labels.to(torch.float64)
    amplitude = torch.complex(basis + 1, 3 * basis - 2)
    amplitude /= amplitude.norm()
    mapped = labels.clone()
    for first, second in physical:
        mapped ^= ((mapped >> (7 - first)) & 1) << (7 - second)
    expected = torch.empty_like(amplitude)
    expected[mapped] = amplitude
    circuit = DynamicCircuit(8, inputs=amplitude, dtype=torch.complex128)
    circuit.append(FieldOperation(8, gates))
    program = lower_reversible_phase(circuit.to_ir())
    assert encode_kmx(program).splitlines()[1:] == [
        f"CX q{first} q{second}" for first, second in physical
    ]
    result = run_dynamic(circuit, shots=2, seed=19, strategy="batched")
    torch.testing.assert_close(
        result.final_states.reshape(2, -1), expected.expand(2, -1), rtol=0, atol=1e-12
    )


def test_sequence_width_and_first_nonscalar_selection() -> None:
    assert broadcast_operand_words(99, (8, 6), False) == (
        (99, 99),
        (8, 6),
        (False, False),
    )
    assert broadcast_operand_words(1, (), False) == ((), (), ())
    assert broadcast_operand_words(1, (2,), 3) == ((1,), (2,), (3,))
    for left, right in (((0,), (4, 5, 6, 7)), ((), (4,))):
        with pytest.raises(ValueError, match="Inconsistent broadcast sizes"):
            broadcast_cx_ir(left, right)


@pytest.mark.parametrize("enabled", [False, True])
def test_locked_classical_and_x_basis_broadcast(enabled: bool) -> None:
    bit = ClassicalFieldCoefficient(0)
    xbit = OperandIdentity(OperandKind.X_CLASSICAL, bit.slot)

    gates = (
        *broadcast_cz_ir(bit, bind_operand_array(range(4))[::-2]),
        *broadcast_cx_ir(0, (xbit, "|->", "|+>")),
    )
    labels = torch.arange(16)
    basis = labels.to(torch.float64)
    amplitude = torch.complex(basis + 1, 2 * basis - 3)
    amplitude /= amplitude.norm()
    signs = (labels & 1) ^ ((labels >> 2) & 1) if enabled else labels >> 3
    expected = torch.where(signs.bool(), -amplitude, amplitude)
    prefix = ClassicalXor((), 0, constant=enabled).instruction()
    circuit = DynamicCircuit(4, inputs=amplitude, dtype=torch.complex128)
    circuit.append(FieldOperation(4, (prefix, *gates)))
    program = lower_reversible_phase(circuit.to_ir())
    assert encode_kmx(program).splitlines()[2 + int(enabled) :] == [
        "Z q3 if b0",
        "Z q1 if b0",
        "Z q0 if b0",
        "Z q0",
    ]
    result = run_dynamic(circuit, shots=2, seed=19, strategy="batched")
    torch.testing.assert_close(
        result.final_states.reshape(2, -1), expected.expand(2, -1), rtol=0, atol=1e-12
    )
    assert torch.all(result.classical_bits[:, 0] == enabled)
