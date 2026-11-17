"""Verify locked typed arrays, strided views and their append consumer."""

import pytest
import torch

from examples.algorithms.register_bit_rotation import RegisterBitRotation
from flagquantum.compiler.artifacts.operand_array import OperandArray
from flagquantum.compiler.artifacts.operand_identity import OperandIdentity, OperandKind
from flagquantum.compiler.formats.kmx import encode_kmx
from flagquantum.compiler.lowering.reversible_phase import lower_reversible_phase
from flagquantum.compiler.registers.finite_field.coefficients import (
    ClassicalFieldCoefficient as Bit,
)
from flagquantum.compiler.registers.layout.operand_word import bind_operand_array
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic


def test_typed_array_creation_and_projection() -> None:
    data = [0, Bit(0), False, True, "|+>", "|->"]
    array = bind_operand_array(data)
    data.clear()
    assert [(op.kind, op.value) for op in array] == [
        (OperandKind.QUANTUM, 0),
        (OperandKind.CLASSICAL, 0),
        (OperandKind.BOOLEAN, 0),
        (OperandKind.BOOLEAN, 1),
        (OperandKind.X_BOOLEAN, 0),
        (OperandKind.X_BOOLEAN, 1),
    ]
    assert len(set(array)) == 6
    namespace = {
        "OperandArray": OperandArray,
        "OperandIdentity": OperandIdentity,
        "OperandKind": OperandKind,
    }
    assert eval(repr(array), {"__builtins__": {}, **namespace}) == array
    assert array[-1] == array[5]
    with pytest.raises(IndexError):
        _ = array[6]
    with pytest.raises(ValueError, match="sequence"):
        bind_operand_array("|+>")
    with pytest.raises(ValueError, match="basis literal"):
        bind_operand_array([None])
    assert bind_operand_array() == bind_operand_array([])
    assert bind_operand_array().common_type == 2


def test_source_slices_owner_and_concatenation() -> None:
    source = bind_operand_array((*range(10), False, *(Bit(i) for i in range(12)), True))
    for item, offset, count, stride in (
        (source[5:], 5, 19, 1),
        (source[14::2], 14, 5, 2),
        (source[::-1], 23, 24, -1),
        (source[5000:], 24, 0, 1),
        (source[1::2][2::3], 5, 4, 6),
    ):
        assert item._owner is source._owner
        assert (item.offset, len(item), item.stride, item.common_type) == (
            offset,
            count,
            stride,
            128,
        )
        assert list(item) == [
            source[i] for i in range(offset, offset + count * stride, stride)
        ]
    selected = source[4:8]
    assert selected.common_type == 128
    compact = selected + bind_operand_array()
    assert compact == selected
    assert compact.common_type == 2
    assert compact._owner is not source._owner
    assert bind_operand_array([0]) != bind_operand_array([False])
    assert bind_operand_array([0, "|+>"]).common_type == 64
    assert bind_operand_array([True, "|+>"]).common_type == 192


def test_reversed_array_view_through_append() -> None:
    wires = bind_operand_array(range(4))[::-1]
    labels = torch.arange(16)
    basis = labels.to(torch.float64)
    amplitudes = torch.complex(basis + 1, 3 * basis - 1)
    amplitudes /= amplitudes.norm()
    circuit = DynamicCircuit(4, inputs=amplitudes, dtype=torch.complex128)
    circuit.append(RegisterBitRotation(wires))
    assert encode_kmx(lower_reversible_phase(circuit.to_ir())).splitlines()[1:] == [
        "SWAP q3 q0",
        "SWAP q2 q1",
        "SWAP q3 q1",
    ]
    bit_reversed = sum(((labels >> bit) & 1) << (3 - bit) for bit in range(4))
    word = ((bit_reversed << 1) | (bit_reversed >> 3)) & 15
    mapped = sum(((word >> bit) & 1) << (3 - bit) for bit in range(4))
    expected = torch.empty_like(amplitudes)
    expected[mapped] = amplitudes
    result = run_dynamic(circuit, shots=2, seed=1, strategy="batched")
    torch.testing.assert_close(
        result.final_states.reshape(2, -1), expected.expand(2, -1), atol=1e-12, rtol=0
    )
