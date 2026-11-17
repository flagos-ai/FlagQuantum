"""Check locked operand admission and its native register consumers."""

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.compiler.lowering.reversible_phase import lower_reversible_phase
from flagquantum.compiler.registers.arithmetic.mixed_addition import mixed_addition_ir
from flagquantum.compiler.registers.bits.bit_rotation_binding import (
    bound_bit_rotation_ir,
)
from flagquantum.compiler.registers.comparison.equality_binding import (
    bound_quantum_equal_ir,
)
from flagquantum.compiler.registers.finite_field.coefficients import (
    ClassicalFieldCoefficient as Bit,
)
from flagquantum.compiler.registers.layout.operand_word import (
    bind_operand_word,
    bind_operand_word_or_integer,
    bind_quantum_word,
    bind_z_word,
)
from flagquantum.compiler.registers.layout.register_bits import join_integer
from flagquantum.compiler.registers.primitives._mixed_gates import _XBasisTarget
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic
from tests.unit.test_kickmix_bit_rotation import _certify


@pytest.mark.parametrize("value", [0, 7, Bit(1), _XBasisTarget(Bit(1)), True, False])
def test_typed_operand_singletons(value: object) -> None:
    assert bind_operand_word(value) == (value,)
    with pytest.raises(ValueError, match="sequence"):
        bind_operand_word(value, sequence_only=True)


def test_snapshot_retains_order_basis_tags_and_view_width() -> None:
    values = [3, Bit(1), "|->", "|+>"]
    port = bind_operand_word(values, sequence_only=True)
    values.clear()
    assert port == (3, Bit(1), _XBasisTarget(True), _XBasisTarget(False))
    assert bind_quantum_word(range(6, 1, -2)) == (6, 4, 2)
    assert bind_operand_word(()) == ()
    assert bind_operand_word_or_integer(True, 65) == (True,)
    with pytest.raises(ValueError, match="Z-basis"):
        bind_z_word(port)
    with pytest.raises(ValueError, match="quantum-only"):
        bind_quantum_word((2, True))
    with pytest.raises(ValueError, match="sequence"):
        bind_operand_word(b"q0", sequence_only=True)


@pytest.mark.parametrize("actual", [1, 3])
def test_reject_inconsistent_sequence_without_binding_partial_word(actual: int) -> None:
    class Sequence:
        def __len__(self) -> int:
            return 2

        def __getitem__(self, index: int) -> int:
            if index < actual:
                return index
            raise IndexError(index)

    with pytest.raises(ValueError, match="length differs"):
        bind_operand_word(Sequence())


@pytest.mark.parametrize("width", [0, 4, 64, 65])
def test_literal_expansion_matches_source_width_range_and_signed_residue(
    width: int,
) -> None:
    limit = 1 << width
    values = {-limit, -limit + 1, -1, limit - 1}
    if width:
        values.add((1 << (width - 1)) | 1)
    for value in values:
        word = bind_operand_word_or_integer(value, width, allow_negative=True)
        assert len(word) == width and all(type(bit) is bool for bit in word)
        assert (
            join_integer(tuple(int(bit is True) for bit in word)) if word else 0
        ) == value % limit
    for larger in (-limit - 1, limit):
        with pytest.raises(ValueError, match="outside"):
            bind_operand_word_or_integer(larger, width, allow_negative=True)
    with pytest.raises(ValueError, match="outside"):
        bind_operand_word_or_integer(-1, width)


def test_sequences_and_literal_expansion_reach_existing_register_bindings() -> None:
    rotation = bound_bit_rotation_ir([3, 2, 1, 0], [True, False])
    assert rotation == bound_bit_rotation_ir((3, 2, 1, 0), (True, False))
    assert bound_quantum_equal_ir([1, 0], 2, 2, (3, 4), measurement_bit=0) == (
        bound_quantum_equal_ir((1, 0), (True, False), 2, (3, 4), measurement_bit=0)
    )
    assert bound_quantum_equal_ir(0, 1, 1) == bound_quantum_equal_ir((0,), (True,), 1)


def test_signed_literal_addition_restores_clean_work_and_complex_amplitudes() -> None:
    offset = bind_z_word(bind_operand_word_or_integer(-1, 4, allow_negative=True))
    gates = mixed_addition_ir(
        (0, 1, 2, 3), offset, clean=(4, 5, 6, 7), classical_bits=(0, 1, 2, 3)
    )
    labels = torch.arange(16)
    mapped = (labels - 1) % 16
    basis = labels.to(torch.float64)
    amplitude = torch.complex(2 * basis + 1, 3 * basis - 2)
    amplitude /= amplitude.norm()
    initial = torch.zeros(256, dtype=torch.complex128)
    observed = torch.zeros_like(initial)
    initial[labels << 4] = amplitude
    observed[mapped << 4] = amplitude
    circuit = DynamicCircuit(8, inputs=initial, dtype=torch.complex128)
    circuit.append(FieldOperation(8, gates))
    program = lower_reversible_phase(circuit.to_ir())
    _certify(program, 4, mapped)
    result = run_dynamic(circuit, shots=16, seed=13, strategy="batched")
    torch.testing.assert_close(
        result.final_states.reshape(16, -1), observed.expand(16, -1), rtol=0, atol=1e-12
    )
