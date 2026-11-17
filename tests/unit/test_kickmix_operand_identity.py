"""Verify source operand families, ranges, and typed append dispatch."""

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.compiler.artifacts.operand_identity import OperandIdentity, OperandKind
from flagquantum.compiler.formats.kmx import decode_kmx, encode_kmx
from flagquantum.compiler.formats.kmx_registers import (
    KmxRegister,
    decode_kmx_registers,
    encode_kmx_registers,
)
from flagquantum.compiler.lowering.reversible_phase import lower_reversible_phase
from flagquantum.compiler.registers.comparison.equality_binding import (
    bound_quantum_equal_ir,
)
from flagquantum.compiler.registers.finite_field.coefficients import (
    ClassicalFieldCoefficient,
)
from flagquantum.compiler.registers.layout.operand_word import (
    bind_operand_word_or_integer,
    bind_quantum_word,
    bind_z_word,
)
from flagquantum.compiler.registers.primitives.control_binding import (
    bind_x_output,
    bind_z_enable,
)
from flagquantum.compiler.registers.primitives.operand_binding import bind_operand
from flagquantum.core.classical_dataflow import ClassicalXor
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic


@pytest.mark.parametrize(
    "kind,bits",
    [
        (OperandKind.BOOLEAN, 1),
        (OperandKind.CLASSICAL, 29),
        (OperandKind.QUANTUM, 29),
        (OperandKind.X_BOOLEAN, 1),
        (OperandKind.X_CLASSICAL, 29),
        (OperandKind.REGISTER, 32),
    ],
)
def test_locked_identity_ranges(kind: OperandKind, bits: int) -> None:
    limit = 1 << bits
    high = OperandIdentity(kind, limit - 1)
    assert high.value == limit - 1 and high.kind is kind
    assert hash(high) == hash(OperandIdentity(kind, limit - 1))
    for bound in (-1, limit):
        with pytest.raises(ValueError, match="family range"):
            OperandIdentity(kind, bound)


def test_source_tags_remain_distinct_and_ordered() -> None:
    zero = tuple(OperandIdentity(kind, 0) for kind in OperandKind)
    assert len(set(zero)) == 6
    ordered = (
        zero[OperandKind.BOOLEAN],
        OperandIdentity(OperandKind.CLASSICAL, 5),
        zero[OperandKind.QUANTUM],
        zero[OperandKind.X_BOOLEAN],
        OperandIdentity(OperandKind.X_BOOLEAN, 1),
        zero[OperandKind.X_CLASSICAL],
    )
    assert sorted(reversed(ordered)) == list(ordered)
    register = zero[OperandKind.REGISTER]
    assert register < OperandIdentity(OperandKind.REGISTER, 1)
    with pytest.raises(TypeError):
        _ = register < zero[OperandKind.QUANTUM]
    assert not zero[OperandKind.X_BOOLEAN]
    assert OperandIdentity(OperandKind.X_BOOLEAN, 1)
    for invalid in (register, "q0", "x+"):
        with pytest.raises(ValueError):
            bind_operand(invalid)


def test_basis_admission_preserves_the_underlying_classical_index() -> None:
    b = OperandIdentity(OperandKind.CLASSICAL, 2)
    xb = OperandIdentity(OperandKind.X_CLASSICAL, 2)
    q = OperandIdentity(OperandKind.QUANTUM, 2)
    assert bind_operand(b) == ClassicalFieldCoefficient(2)
    assert bind_z_enable(b) == (True, (2,))
    assert bind_x_output(xb) == (None, False, (2,))
    assert bind_x_output("|+>") == (None, True, ())
    assert bind_x_output("|->") == (None, False, ())
    assert bind_quantum_word(q) == (2,)
    assert bind_operand_word_or_integer(q, 3) == (2,)
    assert bind_operand_word_or_integer(2, 3) == (False, True, False)
    for invalid in (b, xb, True):
        with pytest.raises(ValueError, match="quantum-only"):
            bind_quantum_word(invalid)
    with pytest.raises(ValueError, match="X-basis"):
        bind_x_output(b)
    with pytest.raises(ValueError, match="classical slot"):
        bind_z_enable(xb)
    with pytest.raises(ValueError, match="Z-basis"):
        bind_z_word((bind_operand(xb),))


@pytest.mark.parametrize("sign", [False, True])
def test_tagged_phase_target_through_append_and_kmx(sign: bool) -> None:
    q = OperandIdentity(OperandKind.QUANTUM, 0)
    xb = OperandIdentity(OperandKind.X_CLASSICAL, 0)
    gates = bound_quantum_equal_ir(q, 1, xb)
    amplitude = torch.tensor([1 + 2j, 3 - 4j], dtype=torch.complex128)
    amplitude /= amplitude.norm()
    expected = torch.stack((amplitude[0], -amplitude[1] if sign else amplitude[1]))
    prefix = ClassicalXor((), 0, constant=sign).instruction()
    circuit = DynamicCircuit(1, inputs=amplitude, dtype=torch.complex128)
    circuit.append(FieldOperation(1, (prefix, *gates)))

    program = lower_reversible_phase(circuit.to_ir())
    assert encode_kmx(program).splitlines()[2 + int(sign) :] == [
        "X q0 if b0",
        "X q0 if b0",
        "Z q0 if b0",
        "X q0 if b0",
        "X q0 if b0",
    ]
    assert decode_kmx(encode_kmx(program)) == program
    registers = (KmxRegister("target", (("q", 0),)), KmxRegister("sign", (("b", 0),)))
    assert decode_kmx_registers(encode_kmx_registers(program, registers))[:2] == (
        program,
        registers,
    )
    result = run_dynamic(circuit, shots=2, seed=13, strategy="batched")
    torch.testing.assert_close(
        result.final_states.reshape(2, -1), expected.expand(2, -1), rtol=0, atol=1e-12
    )
    assert torch.all(result.classical_bits[:, 0] == sign)
