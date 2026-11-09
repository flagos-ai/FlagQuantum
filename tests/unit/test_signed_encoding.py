"""Signed encoding permutations and their measured carry cleanup."""

from collections import Counter
from itertools import product

import pytest
import torch

from flagquantum.compiler.registers.bits.signed_encoding import (
    ones_complement_to_twos_ir,
    sign_magnitude_to_twos_ir,
    signed_encoding_workspace_size,
)
from flagquantum.core.ir import Instruction
from flagquantum.simulation.integer_encoding import integer_to_bits
from flagquantum.simulation.ones_complement_encoding import (
    ones_complement_domain,
    ones_complement_from_bits,
    ones_complement_to_bits,
)
from tests.unit.measurement_checks import certified_and_channel_map, forced_branch
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_register_subtraction import assert_source

pytestmark = pytest.mark.unit
CONVERSIONS = (ones_complement_to_twos_ir, sign_magnitude_to_twos_ir)


def layout(n):
    return tuple(range(n)), tuple(range(n, n + signed_encoding_workspace_size(n)))


def to_bits(word, n):
    return tuple((word >> i) & 1 for i in reversed(range(n)))


def raw_output(word, n, sign_magnitude):
    sign = 1 << (n - 1)
    if word < sign:
        return word
    payload = word ^ sign
    if sign_magnitude:
        payload ^= sign - 1
    return sign | ((payload + 1) % sign)


@pytest.mark.parametrize("n", (2, 3, 4, 5, 8))
@pytest.mark.parametrize("convert", CONVERSIONS)
def test_all_words_all_histories_and_resources(n, convert):
    word, work = layout(n)
    clean = (0,) * len(work)
    initial = tuple(to_bits(i, n) + clean for i in range(1 << n))
    signed = convert is sign_magnitude_to_twos_ir
    expected = tuple(
        to_bits(raw_output(i, n, signed), n) + clean for i in range(1 << n)
    )
    for adjoint, a, b in ((False, initial, expected), (True, expected, initial)):
        gates = convert(word, work, adjoint=adjoint, measurement_bit=7)
        assert certified_and_channel_map(gates, a) == (b, n - 2)
        counts = Counter(g.name for g in gates)
        assert counts["t"] + counts["tdg"] == 4 * (n - 2)
        assert counts["measure"] == counts["reset"] == counts["cz"] == n - 2
        assert counts["cx"] == 7 * (n - 2) + 1 + signed * (n - 1)
        assert counts["x"] == adjoint * 4 * (n - 2)


@pytest.mark.parametrize("n", (2, 3, 5))
@pytest.mark.parametrize("convert", CONVERSIONS)
def test_append_complex_spectator_and_all_raw_branches(n, convert):
    from examples.algorithms.binary_field_arithmetic import FieldOperation
    from flagquantum.dynamic import DynamicCircuit

    word, work = layout(n)
    size = n + len(work) + 1
    initial = torch.zeros(1 << size, dtype=torch.complex128)
    expected = torch.zeros_like(initial)
    coeff = torch.arange(1, 2 ** (n + 1) + 1, dtype=torch.float64)
    amps = torch.complex(coeff, 1 - coeff / 2)
    amps /= amps.norm()
    for i, spectator in product(range(1 << n), (0, 1)):
        output = raw_output(i, n, convert is sign_magnitude_to_twos_ir)
        amplitude = amps[2 * i + spectator]
        initial[(i << (len(work) + 1)) | spectator] = amplitude
        expected[(output << (len(work) + 1)) | spectator] = amplitude
    for adjoint, a, b in ((False, initial, expected), (True, expected, initial)):
        gates = convert(word, work, adjoint=adjoint)
        circuit = DynamicCircuit(size).append(FieldOperation(size, gates)).to_ir()
        for history in product((0, 1), repeat=len(work)):
            torch.testing.assert_close(
                forced_branch(a, tuple(circuit.instructions), history),
                b / 2 ** (len(work) / 2),
                atol=2e-12,
                rtol=0,
            )


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("n", (2, 4, 5, 8))
def test_locked_word_encoding_and_model(n):
    pytest.importorskip("qualtran")
    from qualtran import DecomposeNotImplementedError, QInt, QIntOnesComp
    from qualtran.bloqs.arithmetic.conversions.ones_complement_to_twos_complement import (
        SignedIntegerToTwosComplement,
    )
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    reference = SignedIntegerToTwosComplement(n)
    assert reference.signature.get_left("x").dtype == QIntOnesComp(n)
    assert reference.signature.get_right("y").dtype == QInt(n)
    with pytest.raises(DecomposeNotImplementedError):
        reference.decompose_bloq()
    assert get_cost_value(reference, QECGatesCost()).toffoli == n - 2

    word, work = layout(n)
    gates = ones_complement_to_twos_ir(word, work)
    domain = tuple(ones_complement_domain(n))
    assert domain == tuple(QIntOnesComp(n).get_classical_domain())
    initial = tuple(
        tuple(ones_complement_to_bits(i, n)) + (0,) * len(work) for i in domain
    )
    expected = tuple(
        tuple(integer_to_bits(i, n, signed=True)) + (0,) * len(work) for i in domain
    )
    assert initial == tuple(
        tuple(QIntOnesComp(n).to_bits(i)) + (0,) * len(work) for i in domain
    )
    assert expected == tuple(
        tuple(QInt(n).to_bits(i)) + (0,) * len(work) for i in domain
    )
    assert ones_complement_from_bits([1] * n, n) == 0
    assert certified_and_channel_map(gates, initial) == (expected, n - 2)
    assert QIntOnesComp(n).from_bits([1] * n) == 0
    assert QInt(n).from_bits([1] + [0] * (n - 1)) == -(1 << (n - 1))
    if n == 4:
        assert QIntOnesComp(n).to_bits(-1) == [1, 1, 1, 0]
        assert QInt(n).to_bits(-1) == [1, 1, 1, 1]
        assert QInt(n).from_bits([1, 0, 1, 0]) == -6  # CNOT then increment on 1110.


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("n", (3, 5, 8))
@pytest.mark.parametrize("adjoint", (False, True))
def test_locked_carry_gates_controls_and_feedback(n, adjoint):
    pytest.importorskip("qualtran")
    from qualtran.bloqs.mcmt.and_bloq import And

    word, work = layout(n)
    sign, *payload = word
    lsb = payload[::-1]
    spec = list(zip((sign, *work[:-1]), lsb[:-1], work, strict=True))
    expected = []
    bit = 9
    source_and = And(1, int(not adjoint))
    for control, digit, carry in spec:
        expected.extend(
            reference_arithmetic_gates(
                source_and,
                {"ctrl": (control, digit), "target": (carry,)},
                (),
                {},
                cirq_style=True,
            )
        )
    for i in reversed(range(len(spec))):
        control, digit, carry = spec[i]
        expected.append(Instruction("cx", (carry, lsb[i + 1])))
        expected.extend(
            reference_arithmetic_gates(
                source_and.adjoint(),
                {"ctrl": (control, digit), "target": (carry,)},
                (),
                {carry: bit},
                cirq_style=True,
            )
        )
    expected.append(Instruction("cx", (sign, lsb[0])))
    assert_source(
        ones_complement_to_twos_ir(word, work, adjoint=adjoint, measurement_bit=bit),
        tuple(expected),
    )


@pytest.mark.parametrize(
    "word,work,slot,match",
    [
        ((0,), (), 0, "at least two bits"),
        ((0, 1, 2), (), 0, "clean carry"),
        ((0, 1, 2), (0,), 0, "distinct nonnegative"),
        ((0, 1), (), -1, "nonnegative measurement slot"),
    ],
)
def test_invalid_bindings(word, work, slot, match):
    with pytest.raises(ValueError, match=match):
        ones_complement_to_twos_ir(word, work, measurement_bit=slot)
