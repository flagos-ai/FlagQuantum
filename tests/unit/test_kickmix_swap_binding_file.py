"""Verify the locked matching-basis swap binding and native append."""

from itertools import product

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.compiler.artifacts.operand_identity import OperandIdentity, OperandKind
from flagquantum.compiler.registers.finite_field.coefficients import (
    ClassicalFieldCoefficient as Bit,
)
from flagquantum.compiler.registers.primitives.broadcast_swap import broadcast_swap_ir
from flagquantum.core.classical_dataflow import ClassicalXor
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic
from tests.unit.fixtures.kickmix_binary_gate_matrix import (
    Port,
    source_matrix,
    verify_stream,
)


@pytest.mark.parametrize("left,right,expected", tuple(source_matrix("swap")))
def test_source_swap_streams(
    left: Port, right: Port, expected: tuple[int, str]
) -> None:
    verify_stream(broadcast_swap_ir, left, right, expected)


def test_mixed_basis_swap_appends_and_preserves_complex_state() -> None:
    gates = broadcast_swap_ir(
        (0, Bit(0), OperandIdentity(OperandKind.X_CLASSICAL, 2)),
        (1, Bit(1), OperandIdentity(OperandKind.X_CLASSICAL, 3)),
    )
    basis = torch.arange(4, dtype=torch.float64)
    amplitudes = torch.complex(basis + 1, 3 * basis - 7)
    amplitudes /= amplitudes.norm()
    expected = amplitudes[[0, 2, 1, 3]]
    for bits in product((False, True), repeat=4):
        prefix = tuple(
            ClassicalXor((), slot, constant=value).instruction()
            for slot, value in enumerate(bits)
        )
        circuit = DynamicCircuit(2, inputs=amplitudes, dtype=torch.complex128)
        circuit.append(FieldOperation(2, (*prefix, *gates)))
        result = run_dynamic(circuit, shots=2, seed=19, strategy="batched")
        torch.testing.assert_close(
            result.final_states.reshape(2, -1),
            expected.expand(2, -1),
            atol=1e-12,
            rtol=0,
        )
        assert (
            result.classical_bits.tolist() == [[bits[1], bits[0], bits[3], bits[2]]] * 2
        )


def test_swap_refusal_is_atomic_and_empty_columns_emit_nothing() -> None:
    circuit = DynamicCircuit(3)
    before = circuit.to_ir()
    with pytest.raises(ValueError, match="matching writable types"):
        circuit.append(FieldOperation(3, broadcast_swap_ir((0, 1), (2, Bit(0)))))
    assert circuit.to_ir() == before
    with pytest.raises(ValueError):
        broadcast_swap_ir(0, 0)
    assert broadcast_swap_ir("|->", ()) == ()
