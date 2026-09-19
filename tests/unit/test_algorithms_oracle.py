from __future__ import annotations

import pytest

from flagquantum.algorithms.primitives.oracle import (
    append_bit_oracle,
    append_comparator,
    append_multi_controlled_x,
    append_phase_oracle,
    bit_oracle,
    marked_states,
    phase_oracle,
)
from flagquantum.circuit import Circuit

pytestmark = pytest.mark.unit


def _basis_index(circuit: Circuit, n_wires: int) -> int:
    """Return the computational basis state a classical circuit lands on.

    The purity guard matters: reading ``argmax`` of a state that is *not* a basis state would
    return a plausible index for a superposition, which is exactly the wrong answer this
    module's tests exist to catch.
    """
    probabilities = circuit.state().reshape(-1).abs() ** 2
    assert float(probabilities.max().item()) == pytest.approx(1.0, abs=1e-6)
    return int(probabilities.argmax().item())


def _wire(circuit: Circuit, n_wires: int, wire: int) -> int:
    """Return one wire's value, read by its bit position. Wire 0 is the most significant bit."""
    return (_basis_index(circuit, n_wires) >> (n_wires - 1 - wire)) & 1


def test_multi_controlled_x_flips_only_the_saturated_pattern() -> None:
    """With two controls the target flips exactly when both are set, and nothing else moves."""
    for pattern in range(8):
        circuit = Circuit(3)
        for position, wire in enumerate((0, 1, 2)):
            if (pattern >> (2 - position)) & 1:
                circuit.gate("x", wire)
        append_multi_controlled_x(circuit, [0, 1], 2)
        # The gate XORs its target, so a target that enters as |1> leaves as |0> on the
        # saturated pattern. Asserting the XOR rather than a constant is what makes this
        # detect a gate that resets the target instead of flipping it.
        saturated = int(pattern & 0b110 == 0b110)
        assert _wire(circuit, 3, 2) == ((pattern & 0b1) ^ saturated), pattern
        assert _wire(circuit, 3, 0) == (pattern >> 2) & 1, pattern
        assert _wire(circuit, 3, 1) == (pattern >> 1) & 1, pattern


def test_three_control_multi_controlled_x_is_exact_and_restores_its_ancilla() -> None:
    """Three controls need one ancilla; the target flips on saturation and the ancilla returns."""
    for pattern in range(8):
        circuit = Circuit(5)
        for position, wire in enumerate((0, 1, 2)):
            if (pattern >> (2 - position)) & 1:
                circuit.gate("x", wire)
        append_multi_controlled_x(circuit, [0, 1, 2], 3, ancillas=[4])
        assert _wire(circuit, 5, 3) == int(pattern == 0b111), pattern
        assert _wire(circuit, 5, 4) == 0, pattern
        for position, wire in enumerate((0, 1, 2)):
            assert _wire(circuit, 5, wire) == (pattern >> (2 - position)) & 1, pattern


def test_multi_controlled_x_scales_to_four_controls() -> None:
    """Four controls need two ancillas.

    This is the case that exercises the ladder's inner loop, which runs only above three
    controls; a wrong inner range leaves the three-control test passing.
    """
    for pattern in range(16):
        circuit = Circuit(7)
        for position, wire in enumerate((0, 1, 2, 3)):
            if (pattern >> (3 - position)) & 1:
                circuit.gate("x", wire)
        append_multi_controlled_x(circuit, [0, 1, 2, 3], 4, ancillas=[5, 6])
        assert _wire(circuit, 7, 4) == int(pattern == 0b1111), pattern
        assert _wire(circuit, 7, 5) == 0, pattern
        assert _wire(circuit, 7, 6) == 0, pattern


def test_multi_controlled_x_refuses_three_controls_without_an_ancilla() -> None:
    """There is no native three-control gate, so the caller must supply the ancilla."""
    with pytest.raises(ValueError):
        append_multi_controlled_x(Circuit(4), [0, 1, 2], 3)


def test_multi_controlled_x_refuses_the_wrong_number_of_ancillas() -> None:
    """Four controls need exactly two ancillas."""
    with pytest.raises(ValueError):
        append_multi_controlled_x(Circuit(7), [0, 1, 2, 3], 4, ancillas=[5])


def test_multi_controlled_x_refuses_an_empty_control_list() -> None:
    """A gate with no controls is not this module's business."""
    with pytest.raises(ValueError):
        append_multi_controlled_x(Circuit(2), [], 1)


def test_multi_controlled_x_refuses_an_overlapping_target() -> None:
    """A target that is also a control would flip itself."""
    with pytest.raises(ValueError):
        append_multi_controlled_x(Circuit(3), [0, 1], 1)


def test_comparator_marks_lhs_greater_than_rhs() -> None:
    """Over every two-bit pair, the target is set exactly when lhs > rhs."""
    for left in range(4):
        for right in range(4):
            circuit = Circuit(9)
            for position, wire in enumerate((0, 1)):
                if (left >> (1 - position)) & 1:
                    circuit.gate("x", wire)
            for position, wire in enumerate((2, 3)):
                if (right >> (1 - position)) & 1:
                    circuit.gate("x", wire)
            append_comparator(
                circuit, lhs=[0, 1], rhs=[2, 3], target=4, equality=[5, 6, 7], scratch=8
            )
            assert _wire(circuit, 9, 4) == int(left > right), (left, right)


def test_comparator_leaves_every_other_wire_alone() -> None:
    """The clean-exit contract: only the target moves.

    One equality compares the whole resulting index, so this single assertion covers the
    operands, the target, all ``n + 1`` equality flags and the scratch wire at once.
    """
    for left in range(4):
        for right in range(4):
            circuit = Circuit(9)
            for position, wire in enumerate((0, 1)):
                if (left >> (1 - position)) & 1:
                    circuit.gate("x", wire)
            for position, wire in enumerate((2, 3)):
                if (right >> (1 - position)) & 1:
                    circuit.gate("x", wire)
            append_comparator(
                circuit, lhs=[0, 1], rhs=[2, 3], target=4, equality=[5, 6, 7], scratch=8
            )
            expected = (left << 7) | (right << 5) | (int(left > right) << 4)
            assert _basis_index(circuit, 9) == expected, (left, right)


def test_comparator_is_exact_and_clean_at_three_bits() -> None:
    """The three-bit case is where the comparator's internal multi-controlled X needs its ancilla."""
    for left in range(8):
        for right in range(8):
            circuit = Circuit(12)
            for position, wire in enumerate((0, 1, 2)):
                if (left >> (2 - position)) & 1:
                    circuit.gate("x", wire)
            for position, wire in enumerate((3, 4, 5)):
                if (right >> (2 - position)) & 1:
                    circuit.gate("x", wire)
            append_comparator(
                circuit,
                lhs=[0, 1, 2],
                rhs=[3, 4, 5],
                target=6,
                equality=[7, 8, 9, 10],
                scratch=11,
            )
            expected = (left << 9) | (right << 6) | (int(left > right) << 5)
            assert _basis_index(circuit, 12) == expected, (left, right)


def test_comparator_refuses_mismatched_operand_lengths() -> None:
    """Comparing two different-width bit strings is not defined here."""
    with pytest.raises(ValueError):
        append_comparator(
            Circuit(6), lhs=[0, 1], rhs=[2], target=3, equality=[4, 5], scratch=6
        )


def test_comparator_refuses_a_wrong_equality_width() -> None:
    """The equality register is n + 1 wires: one flag per bit, plus the constant input."""
    with pytest.raises(ValueError):
        append_comparator(
            Circuit(8), lhs=[0, 1], rhs=[2, 3], target=4, equality=[5, 6], scratch=7
        )


def test_comparator_refuses_overlapping_wires() -> None:
    """A scratch wire that is also an operand would corrupt the comparison."""
    with pytest.raises(ValueError):
        append_comparator(
            Circuit(8), lhs=[0, 1], rhs=[2, 3], target=4, equality=[5, 6, 7], scratch=3
        )


def test_marked_states_enumerates_the_predicate() -> None:
    """The marked set is exactly the satisfying assignment, in ascending order."""
    assert marked_states(lambda value: value in (3, 5), 3) == (3, 5)
    assert marked_states(lambda value: False, 2) == ()
    assert marked_states(lambda value: True, 2) == (0, 1, 2, 3)


def test_marked_states_rejects_a_non_positive_wire_count() -> None:
    """An empty register has no states to mark."""
    with pytest.raises(ValueError):
        marked_states(lambda value: True, 0)


def test_phase_oracle_flips_the_sign_of_marked_states_only() -> None:
    """Applied to each basis state, the sign is negated on exactly the marked set.

    Reading the amplitude at the prepared index is the decisive check here: a phase oracle
    that does nothing, or that flips every state, both fail it.
    """
    for n_wires in (1, 2, 3):

        def predicate(value: int, n: int = n_wires) -> bool:
            return value in (1, 2**n - 1)

        marked = set(marked_states(predicate, n_wires))
        for basis in range(2**n_wires):
            circuit = Circuit(n_wires)
            for position in range(n_wires):
                if (basis >> (n_wires - 1 - position)) & 1:
                    circuit.gate("x", position)
            append_phase_oracle(circuit, predicate, list(range(n_wires)))
            amplitude = complex(circuit.state().reshape(-1)[basis])
            expected = -1.0 if basis in marked else 1.0
            assert amplitude == pytest.approx(expected, abs=1e-5), (n_wires, basis)


def test_phase_oracle_on_a_superposition() -> None:
    """In a uniform superposition only the marked amplitude is negated."""
    n_wires = 2
    circuit = Circuit(n_wires)
    for wire in range(n_wires):
        circuit.gate("h", wire)
    before = circuit.state().reshape(-1).clone()
    append_phase_oracle(circuit, lambda value: value == 2, list(range(n_wires)))
    after = circuit.state().reshape(-1)
    for index in range(2**n_wires):
        expected = -before[index] if index == 2 else before[index]
        # ``complex(...)`` on both sides: ``pytest.approx`` only understands a tensor
        # when numpy is importable, and this repository does not declare numpy, so a
        # bare tensor operand compares with ``==`` and never equals an ``approx``.
        assert complex(after[index]) == pytest.approx(
            complex(expected), abs=1e-6
        ), index


def test_phase_oracle_refuses_more_than_three_wires() -> None:
    """A four-wire phase oracle has no wire to spare for the multi-controlled X's ancilla."""
    with pytest.raises(ValueError):
        phase_oracle(lambda value: value == 1, 4)


def test_bit_oracle_wire_budget() -> None:
    """The output wire plus one ladder ancilla per control above two."""
    for n_wires, expected in ((1, 2), (2, 3), (3, 5), (4, 7)):
        assert (
            bit_oracle(lambda value: value == 0, n_wires).n_qubits == expected
        ), n_wires


def test_bit_oracle_xors_the_predicate_onto_the_output() -> None:
    """The output carries the predicate value, and every ladder ancilla returns to zero."""
    for n_wires in (1, 2, 3):
        n_ancillas = max(0, n_wires - 2)

        def predicate(value: int, n: int = n_wires) -> bool:
            return value == 2**n - 1

        for value in range(2**n_wires):
            circuit = Circuit(n_wires + 1 + n_ancillas)
            for position in range(n_wires):
                if (value >> (n_wires - 1 - position)) & 1:
                    circuit.gate("x", position)
            append_bit_oracle(
                circuit,
                predicate,
                list(range(n_wires)),
                target=n_wires,
                ancillas=list(range(n_wires + 1, n_wires + 1 + n_ancillas)),
            )
            probabilities = circuit.state().reshape(-1).abs() ** 2
            assert float(probabilities.max().item()) == pytest.approx(
                1.0, abs=1e-6
            ), value
            index = int(probabilities.argmax().item())
            expected = (value << (n_ancillas + 1)) | (
                int(predicate(value)) << n_ancillas
            )
            assert index == expected, (n_wires, value)


def test_bit_oracle_is_its_own_inverse() -> None:
    """Applying the oracle twice leaves the output where it started."""
    n_wires = 2
    circuit = Circuit(n_wires + 1)
    circuit.gate("x", 0)
    append_bit_oracle(
        circuit, lambda value: value == 2, list(range(n_wires)), target=n_wires
    )
    once = int((circuit.state().reshape(-1).abs() ** 2).argmax().item())
    append_bit_oracle(
        circuit, lambda value: value == 2, list(range(n_wires)), target=n_wires
    )
    twice = int((circuit.state().reshape(-1).abs() ** 2).argmax().item())
    assert once != twice
    assert twice == (1 << (n_wires + 1 - 1 - 0))
