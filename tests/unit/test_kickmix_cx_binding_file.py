"""Whole-file acceptance for locked CX bindings and generation."""

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.compiler.formats.kmx import encode_kmx
from flagquantum.compiler.lowering.reversible_phase import lower_reversible_phase
from flagquantum.compiler.registers.finite_field.coefficients import (
    ClassicalFieldCoefficient as Bit,
)
from flagquantum.compiler.registers.primitives.broadcast_binding import broadcast_cx_ir
from flagquantum.core.classical_dataflow import ClassicalXor
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic
from tests.unit.fixtures.kickmix_binary_gate_matrix import (
    Port,
    source_matrix,
    verify_stream,
)


@pytest.mark.parametrize("left,right,expected", tuple(source_matrix("cx")))
def test_cx_source_matrix(left: Port, right: Port, expected: tuple[int, str]) -> None:
    verify_stream(broadcast_cx_ir, left, right, expected)


def test_cx_classical_target_sequences_and_invalid_lanes() -> None:
    assert broadcast_cx_ir("|->", ()) == broadcast_cx_ir((), True) == ()
    for left, right in (
        (0, Bit(1)),
        ("|->", 1),
        (0, True),
        ((0, 1), (2, Bit(3))),
        (0, 0),
    ):
        with pytest.raises(ValueError):
            broadcast_cx_ir(left, right)


@pytest.mark.parametrize("value", [False, True])
def test_cx_classical_self_alias_clears_slot(value: bool) -> None:
    amplitude = torch.tensor([1 + 2j, -3 + 4j], dtype=torch.complex128)
    amplitude /= amplitude.norm()
    setup = ClassicalXor((), 0, constant=value).instruction()
    circuit = DynamicCircuit(1, inputs=amplitude, dtype=torch.complex128)
    circuit.append(FieldOperation(1, (setup, *broadcast_cx_ir(Bit(0), Bit(0)))))
    assert (
        encode_kmx(lower_reversible_phase(circuit.to_ir())).splitlines()[-1]
        == "BIT_INVERT b0 if b0"
    )
    result = run_dynamic(circuit, shots=2, seed=13, strategy="batched")
    assert not torch.any(result.classical_bits[:, 0])
    torch.testing.assert_close(
        result.final_states.reshape(2, -1), amplitude.expand(2, -1)
    )


def test_locked_bit_target_sequence() -> None:
    setup = tuple(ClassicalXor((), slot).instruction() for slot in range(8))
    gates = (
        *broadcast_cx_ir(Bit(0), Bit(1)),
        *broadcast_cx_ir(True, Bit(2)),
        *broadcast_cx_ir(False, Bit(3)),
        *broadcast_cx_ir(True, (Bit(4), Bit(5))),
        *broadcast_cx_ir((Bit(0), True), (Bit(6), Bit(7))),
    )
    circuit = DynamicCircuit(1)
    circuit.append(FieldOperation(1, (*setup, *gates)))
    assert encode_kmx(lower_reversible_phase(circuit.to_ir())).splitlines()[9:] == [
        "BIT_INVERT b1 if b0",
        "BIT_INVERT b2",
        "BIT_INVERT b4",
        "BIT_INVERT b5",
        "BIT_INVERT b6 if b0",
        "BIT_INVERT b7",
    ]
