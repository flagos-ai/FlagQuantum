"""Whole-file acceptance for locked CZ bindings and generation."""

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.compiler.formats.kmx import encode_kmx
from flagquantum.compiler.lowering.reversible_phase import lower_reversible_phase
from flagquantum.compiler.registers.finite_field.coefficients import (
    ClassicalFieldCoefficient as Bit,
)
from flagquantum.compiler.registers.primitives.broadcast_binding import broadcast_cz_ir
from flagquantum.core.classical_dataflow import ClassicalXor
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic
from tests.unit.fixtures.kickmix_binary_gate_matrix import (
    Port,
    source_matrix,
    verify_stream,
)


@pytest.mark.parametrize("left,right,expected", tuple(source_matrix("cz")))
def test_cz_source_matrix(left: Port, right: Port, expected: tuple[int, str]) -> None:
    verify_stream(broadcast_cz_ir, left, right, expected)


def test_cz_basis_validation_and_empty_lanes() -> None:
    assert broadcast_cz_ir("|->", ()) == broadcast_cz_ir((), "|+>") == ()
    for left, right in (("|->", 1), (0, "|+>"), ((0, 1), (2, "|->")), (0, 0)):
        with pytest.raises(ValueError):
            broadcast_cz_ir(left, right)


@pytest.mark.parametrize("value", [False, True])
def test_cz_duplicate_classical_read_keeps_one_effective_guard(value: bool) -> None:
    amplitude = torch.tensor([1 + 2j, -3 + 4j], dtype=torch.complex128)
    amplitude /= amplitude.norm()
    setup = ClassicalXor((), 0, constant=value).instruction()
    circuit = DynamicCircuit(1, inputs=amplitude, dtype=torch.complex128)
    circuit.append(FieldOperation(1, (setup, *broadcast_cz_ir(Bit(0), Bit(0)))))
    assert (
        encode_kmx(lower_reversible_phase(circuit.to_ir())).splitlines()[-1]
        == "NEG if b0"
    )
    result = run_dynamic(circuit, shots=2, seed=13, strategy="batched")
    assert torch.all(result.classical_bits[:, 0] == value)
    expected = -amplitude if value else amplitude
    torch.testing.assert_close(
        result.final_states.reshape(2, -1), expected.expand(2, -1)
    )
