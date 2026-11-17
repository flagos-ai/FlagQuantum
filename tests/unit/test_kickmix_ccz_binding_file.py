"""Whole-file source verification for the locked CCZ binding."""

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.compiler.registers.finite_field.coefficients import (
    ClassicalFieldCoefficient as Bit,
)
from flagquantum.compiler.registers.primitives.broadcast_ccz import broadcast_ccz_ir
from flagquantum.core.classical_dataflow import ClassicalXor
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic
from tests.unit.fixtures.kickmix_binary_gate_matrix import (
    Port,
    ternary_matrix,
    verify_gate_stream,
)


@pytest.mark.parametrize("columns,expected", tuple(ternary_matrix("ccz")))
def test_ccz_source_matrix(
    columns: tuple[Port, ...], expected: tuple[int, str]
) -> None:
    verify_gate_stream(lambda ports, _: broadcast_ccz_ir(*ports), columns, expected)


@pytest.mark.parametrize("bit", [False, True])
def test_ccz_preserves_full_complex_phase_for_mixed_lanes(bit: bool) -> None:
    labels = torch.arange(16)
    basis = labels.to(torch.float64)
    amplitudes = torch.complex(basis + 1, 3 * basis - 2)
    amplitudes /= amplitudes.norm()
    phase = ((labels >> 3) & (labels >> 2) & (labels >> 1) & 1) ^ (labels & int(bit))
    expected = torch.where(phase.bool(), -amplitudes, amplitudes)
    gates = broadcast_ccz_ir((0, Bit(0)), (1, True), (2, 3))
    prefix = ClassicalXor((), 0, constant=bit).instruction()
    circuit = DynamicCircuit(4, inputs=amplitudes, dtype=torch.complex128)
    circuit.append(FieldOperation(4, (prefix, *gates)))
    result = run_dynamic(circuit, shots=2, seed=19, strategy="batched")
    torch.testing.assert_close(
        result.final_states.reshape(2, -1), expected.expand(2, -1), rtol=0, atol=1e-12
    )
    assert torch.all(result.classical_bits[:, 0] == bit)


def test_ccz_basis_and_empty_admission() -> None:
    assert broadcast_ccz_ir("|->", (), 2) == ()
    with pytest.raises(ValueError, match="Z-basis"):
        broadcast_ccz_ir((0, 1), ("|->", True), (2, 3))
