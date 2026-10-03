"""Unit coverage for the Steane seven-qubit code record.

As with the rotated surface code, the record is verified against its own algebra
rather than against a transcribed table: the checks must commute pairwise, must
generate a rank-six group whose non-identity elements all have weight four, and
must admit a weight-three logical observable that is independent of them. Two
properties are specific to this code and are asserted directly. It is a CSS code
whose X-type and Z-type checks measure the same three supports, and it is
perfect, which forces some data qubit into all three Z-type checks in every basis
of the group -- the shape that makes its phenomenological detector error model
non-graphlike.
"""

from __future__ import annotations

import itertools

import pytest

from flagquantum.qec.circuit import build_memory_circuit
from flagquantum.qec.codes import StabilizerCode, SteaneCode
from flagquantum.qec.pauli import Pauli

pytestmark = pytest.mark.unit

_DATA_WIRES = 7


def _gf2_rank(rows: list[list[int]]) -> int:
    """Return the rank of a binary matrix over GF(2)."""

    matrix = [list(row) for row in rows]
    rank = 0
    columns = len(matrix[0]) if matrix else 0
    for column in range(columns):
        pivot = next(
            (row for row in range(rank, len(matrix)) if matrix[row][column]), None
        )
        if pivot is None:
            continue
        matrix[rank], matrix[pivot] = matrix[pivot], matrix[rank]
        for row in range(len(matrix)):
            if row != rank and matrix[row][column]:
                matrix[row] = [
                    left ^ right
                    for left, right in zip(matrix[row], matrix[rank], strict=True)
                ]
        rank += 1
    return rank


def _symplectic(pauli: Pauli, wires: int) -> list[int]:
    return [1 if wire in pauli.x_wires else 0 for wire in range(wires)] + [
        1 if wire in pauli.z_wires else 0 for wire in range(wires)
    ]


def _check_supports(*, x_type: bool) -> tuple[tuple[int, ...], ...]:
    """Return the declared supports of one check class, in declaration order."""

    return tuple(
        check.stabilizer.support
        for check in SteaneCode().checks
        if bool(check.stabilizer.x_wires) is x_type
    )


def _symmetric_difference(
    supports: tuple[tuple[int, ...], ...],
) -> tuple[int, ...]:
    combined: set[int] = set()
    for support in supports:
        combined ^= set(support)
    return tuple(sorted(combined))


def _group_elements(*, x_type: bool) -> tuple[tuple[int, ...], ...]:
    """Return the non-identity elements the declared checks of one class generate."""

    generators = _check_supports(x_type=x_type)
    elements = {
        _symmetric_difference(combination)
        for size in range(1, len(generators) + 1)
        for combination in itertools.combinations(generators, size)
    }
    elements.discard(())
    return tuple(sorted(elements, key=lambda item: (len(item), item)))


def _kernel_vectors(*, x_type: bool) -> tuple[tuple[int, ...], ...]:
    """Return every non-zero binary vector with even overlap on each support.

    The enumeration walks sizes in ascending order, so the first entry is a
    minimum-weight element of the kernel and every element of that weight
    follows it before any heavier one.
    """

    rows = _check_supports(x_type=x_type)
    return tuple(
        support
        for size in range(1, _DATA_WIRES + 1)
        for support in itertools.combinations(range(_DATA_WIRES), size)
        if all(len(set(support) & set(row)) % 2 == 0 for row in rows)
    )


def test_wire_layout_is_seven_data_wires_and_six_ancillas() -> None:
    code = SteaneCode()

    assert code.distance == 3
    assert code.num_data_qubits == 7
    assert code.num_ancilla_qubits == 6
    assert code.data_wires == (0, 1, 2, 3, 4, 5, 6)
    assert code.ancilla_wires == (7, 8, 9, 10, 11, 12)


def test_checks_are_three_z_type_then_three_x_type_all_of_weight_four() -> None:
    checks = SteaneCode().checks

    assert [check.index for check in checks] == [0, 1, 2, 3, 4, 5]
    assert [check.ancilla_wire for check in checks] == [7, 8, 9, 10, 11, 12]
    assert [bool(check.stabilizer.x_wires) for check in checks] == [
        False,
        False,
        False,
        True,
        True,
        True,
    ]
    assert {check.stabilizer.weight for check in checks} == {4}


def test_both_check_classes_measure_the_same_supports() -> None:
    """The code is CSS on one parity check, so the two classes are not independent."""

    assert _check_supports(x_type=True) == _check_supports(x_type=False)


def test_the_group_has_seven_non_identity_elements_all_of_weight_four() -> None:
    """A perfect distance-three code: the other seven elements are the weight fours."""

    for x_type in (True, False):
        elements = _group_elements(x_type=x_type)
        assert len(elements) == 7
        assert {len(element) for element in elements} == {4}


def test_checks_pairwise_commute_and_generate_a_rank_six_group() -> None:
    code = SteaneCode()
    checks = code.checks

    for left in checks:
        for right in checks:
            assert left.stabilizer.commutes_with(right.stabilizer)

    rows = [_symplectic(check.stabilizer, _DATA_WIRES) for check in checks]
    assert _gf2_rank(rows) == 6


@pytest.mark.parametrize("x_type", (True, False))
def test_every_check_class_has_distance_three(x_type: bool) -> None:
    """The kernel's lightest vectors outside the group are the distance witnesses.

    Every non-identity group element has weight four, so no kernel vector of
    weight three lies in the group, and the lightest kernel vector is therefore a
    genuine logical operator of that class.
    """

    kernel = _kernel_vectors(x_type=x_type)
    group = set(_group_elements(x_type=x_type))
    lightest = [vector for vector in kernel if len(vector) == 3]

    assert len(lightest) == 7
    assert not group.intersection(lightest)
    assert min(len(vector) for vector in kernel if vector not in group) == 3


def test_the_declared_logical_observable_is_the_lightest_z_representative() -> None:
    code = SteaneCode()
    z_type = tuple(item for item in code.logical_observables if not item.x_wires)
    (observable,) = z_type

    assert observable.z_wires
    assert not observable.x_wires
    assert observable.weight == 3
    # Seven weight-three representatives are equivalent up to a stabilizer. The
    # record declares the lexicographically first of them, which is what the
    # kernel enumeration below returns first.
    assert observable.z_wires == _kernel_vectors(x_type=False)[0]
    assert observable.z_wires == (0, 1, 2)
    assert all(observable.commutes_with(check.stabilizer) for check in code.checks)
    assert observable.support not in set(_group_elements(x_type=False))

    rows = [_symplectic(check.stabilizer, _DATA_WIRES) for check in code.checks]
    assert _gf2_rank(rows + [_symplectic(observable, _DATA_WIRES)]) == 7

    # The record declares the X-type partner as well, on the same support, and
    # the two anticommute because their overlap has odd size. That partner is a
    # real logical operator of the code; it is simply not one a Z-basis readout
    # can measure, so a memory circuit built from this record carries the Z-type
    # operator alone.
    x_type = tuple(item for item in code.logical_observables if item.x_wires)
    (partner,) = x_type
    assert partner.x_wires == _kernel_vectors(x_type=True)[0]
    assert partner.x_wires == observable.z_wires
    assert not observable.commutes_with(partner)


def test_an_x_type_check_couples_the_ancilla_into_the_data() -> None:
    """The X gadget's CNOT direction is the mirror of the Z gadget's."""

    for check in SteaneCode().checks:
        controls = tuple(control for control, _ in check.cnot_wires)
        targets = tuple(target for _, target in check.cnot_wires)
        if check.stabilizer.x_wires:
            assert controls == (check.ancilla_wire,) * len(check.cnot_wires)
            assert targets == check.stabilizer.support
        else:
            assert targets == (check.ancilla_wire,) * len(check.cnot_wires)
            assert controls == check.stabilizer.support


def test_no_basis_of_the_group_produces_a_graphlike_check_matrix() -> None:
    """Some data qubit lies in all three Z-type checks in every basis.

    A data qubit covered by all three checks flips three detectors when a single
    X error hits it, and a mechanism that flips three detectors is a hyperedge:
    the matching decoder requires every mechanism to flip one or two. The
    assertion is over every basis of the group rather than over the declared one,
    so it states a property of the code and not of this record's choice of
    generators. The best basis still reaches three, and the declared basis does
    too, with data wire 6 covered by all three Z-type checks.
    """

    elements = _group_elements(x_type=False)
    coverage_minimum: int | None = None
    for basis in itertools.combinations(elements, 3):
        if (
            basis[2] == _symmetric_difference(basis[:2])
            or basis[1] == _symmetric_difference((basis[0], basis[2]))
            or basis[0] == _symmetric_difference(basis[1:])
        ):
            continue
        coverage = max(
            sum(1 for support in basis if wire in support)
            for wire in range(_DATA_WIRES)
        )
        assert coverage >= 2
        coverage_minimum = (
            coverage if coverage_minimum is None else min(coverage_minimum, coverage)
        )

    assert coverage_minimum == 3

    declared = _check_supports(x_type=False)
    declared_coverage = [
        sum(1 for support in declared if wire in support) for wire in range(_DATA_WIRES)
    ]
    assert max(declared_coverage) == 3
    assert declared_coverage[6] == 3


@pytest.mark.parametrize("rounds", (1, 2, 3, 5))
def test_memory_detector_count_follows_the_two_check_classes(rounds: int) -> None:
    memory = build_memory_circuit(SteaneCode(), rounds=rounds)

    assert len(memory.detectors) == 3 * (rounds + 1) + 3 * (rounds - 1)
    assert len(memory.observables) == 1


def test_steane_code_satisfies_the_stabilizer_code_protocol() -> None:
    assert isinstance(SteaneCode(), StabilizerCode)


def test_steane_code_is_frozen_and_hashable() -> None:
    assert SteaneCode() == SteaneCode()
    assert len({SteaneCode(), SteaneCode()}) == 1


def test_the_registered_checks_and_stabilizers_agree() -> None:
    code = SteaneCode()

    assert code.stabilizers == tuple(check.stabilizer for check in code.checks)
    assert code.stabilizers == (
        Pauli(z_wires=(3, 4, 5, 6)),
        Pauli(z_wires=(1, 2, 5, 6)),
        Pauli(z_wires=(0, 2, 4, 6)),
        Pauli(x_wires=(3, 4, 5, 6)),
        Pauli(x_wires=(1, 2, 5, 6)),
        Pauli(x_wires=(0, 2, 4, 6)),
    )
