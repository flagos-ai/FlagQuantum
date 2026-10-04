"""Unit coverage for the rotated surface code record.

The record is verified against its own algebra rather than against a transcribed
coordinate table: the checks must commute with each other, must commute with the
declared logical observable, and must generate a group of rank ``distance**2 - 1``
that the observable is independent of. Those properties pin the code's
parameters without depending on how this module happens to number its wires.
"""

from __future__ import annotations

from collections import Counter

import pytest

from flagquantum.qec.circuit import build_memory_circuit
from flagquantum.qec.codes import (
    CodeCheck,
    RepetitionCode,
    RotatedSurfaceCode,
    ancilla_bands,
)
from flagquantum.qec.pauli import Pauli

pytestmark = pytest.mark.unit


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
    return [1 if wire in pauli.x_qubits else 0 for wire in range(wires)] + [
        1 if wire in pauli.z_qubits else 0 for wire in range(wires)
    ]


@pytest.mark.parametrize("distance", (3, 5, 7))
def test_check_counts_match_the_rotated_layout(distance: int) -> None:
    code = RotatedSurfaceCode(distance=distance)
    checks = code.checks
    x_checks = sum(1 for check in checks if check.stabilizer.x_qubits)
    z_checks = len(checks) - x_checks

    assert code.num_data_qubits == distance * distance
    assert code.num_ancilla_qubits == distance * distance - 1
    assert len(checks) == distance * distance - 1
    assert x_checks == z_checks
    assert code.data_qubits == tuple(range(distance * distance))
    assert code.ancilla_qubits == tuple(
        range(distance * distance, 2 * distance * distance - 1)
    )


@pytest.mark.parametrize("distance", (3, 5, 7))
def test_every_check_is_weight_two_or_four_and_the_boundary_ones_are_weight_two(
    distance: int,
) -> None:
    """Four boundary checks per side are the only truncated ones.

    A corner-adjacent ancilla reaches one data column or row instead of two, so
    it has two neighbours rather than four. There are ``2 * (distance - 1)`` such
    checks, and a code with a different number of them would be a different
    lattice rather than the rotated surface code.
    """

    checks = RotatedSurfaceCode(distance=distance).checks
    weights = Counter(check.stabilizer.weight for check in checks)

    assert set(weights) == {2, 4}
    assert weights[2] == 2 * (distance - 1)
    assert weights[4] == len(checks) - weights[2]


@pytest.mark.parametrize("distance", (3, 5, 7))
def test_checks_pairwise_commute_and_generate_the_full_stabilizer_group(
    distance: int,
) -> None:
    code = RotatedSurfaceCode(distance=distance)
    checks = code.checks
    wires = code.num_data_qubits

    for left in checks:
        for right in checks:
            assert left.stabilizer.commutes_with(right.stabilizer)

    rows = [_symplectic(check.stabilizer, wires) for check in checks]
    assert _gf2_rank(rows) == distance * distance - 1


@pytest.mark.parametrize("distance", (3, 5, 7))
def test_the_logical_observable_is_a_z_operator_outside_the_stabilizer_group(
    distance: int,
) -> None:
    code = RotatedSurfaceCode(distance=distance)
    checks = code.checks
    wires = code.num_data_qubits
    observable = code.logical_observables[0]

    assert observable.z_qubits
    assert not observable.x_qubits
    assert observable.weight == distance
    # The declared operator is the data row ``j == 0``. Which weight-``d``
    # operator a patch declares is a convention, and the assertion pins the one
    # this record states rather than any of the equivalent choices: a column
    # would satisfy every other assertion below, because every member of row
    # zero also lies in its own column.
    assert observable.z_qubits == tuple(range(distance))
    assert all(observable.commutes_with(check.stabilizer) for check in checks)

    rows = [_symplectic(check.stabilizer, wires) for check in checks]
    assert _gf2_rank(rows + [_symplectic(observable, wires)]) == distance * distance


@pytest.mark.parametrize("distance", (3, 5, 7))
def test_an_x_type_check_couples_the_ancilla_into_the_data(distance: int) -> None:
    """The X gadget's CNOT direction is the mirror of the Z gadget's.

    A Z-type check controls from each data wire into an ancilla prepared in
    ``|0>``; an X-type check controls from an ancilla prepared in ``|+>`` into
    each data wire. Getting the direction wrong in either place leaves a record
    that looks plausible but measures the wrong operator.
    """

    for check in RotatedSurfaceCode(distance=distance).checks:
        controls = tuple(control for control, _ in check.cnot_qubits)
        targets = tuple(target for _, target in check.cnot_qubits)
        if check.stabilizer.x_qubits:
            assert controls == (check.ancilla_qubit,) * len(check.cnot_qubits)
            assert targets == check.stabilizer.support
        else:
            assert targets == (check.ancilla_qubit,) * len(check.cnot_qubits)
            assert controls == check.stabilizer.support


@pytest.mark.parametrize("rounds", (1, 2, 3, 5))
@pytest.mark.parametrize("distance", (3, 5, 7))
def test_memory_detector_count_follows_the_two_check_classes(
    distance: int, rounds: int
) -> None:
    """Only the checks the readout can decide get round-zero and terminal detectors.

    The initial state is all-zero and the terminal readout is in the Z basis, so
    a Z-type check is deterministic in round zero and at the terminal readout,
    while an X-type check is deterministic only in comparison with the round
    before it.
    """

    code = RotatedSurfaceCode(distance=distance)
    memory = build_memory_circuit(code, rounds=rounds)
    z_checks = sum(1 for check in code.checks if not check.stabilizer.x_qubits)
    x_checks = len(code.checks) - z_checks

    assert len(memory.detectors) == z_checks * (rounds + 1) + x_checks * (rounds - 1)
    assert len(memory.observables) == 1


def test_repetition_code_detector_count_is_unchanged() -> None:
    """The general rule has to reproduce the frozen repetition profile exactly."""

    for distance in (2, 3, 5):
        for rounds in (1, 2, 4):
            memory = build_memory_circuit(RepetitionCode(distance), rounds=rounds)
            assert len(memory.detectors) == (distance - 1) * (rounds + 1)


def test_rotated_surface_code_rejects_a_non_integer_distance() -> None:
    with pytest.raises(TypeError, match="integer"):
        RotatedSurfaceCode(distance=3.0)  # type: ignore[arg-type]


def test_rotated_surface_code_rejects_a_distance_below_two() -> None:
    with pytest.raises(ValueError, match="at least two"):
        RotatedSurfaceCode(distance=1)


def test_rotated_surface_code_rejects_a_boolean_distance() -> None:
    with pytest.raises(TypeError, match="integer"):
        RotatedSurfaceCode(distance=True)


class _XMemoryCode:
    """A code whose declared logical observable is X-type."""

    distance = 3

    @property
    def num_data_qubits(self) -> int:
        return 2

    @property
    def num_ancilla_qubits(self) -> int:
        return 1

    @property
    def num_ancilla_x_qubits(self) -> int:
        return len(ancilla_bands(self.checks)[0])

    @property
    def num_ancilla_z_qubits(self) -> int:
        return len(ancilla_bands(self.checks)[1])

    @property
    def data_qubits(self) -> tuple[int, ...]:
        return (0, 1)

    @property
    def ancilla_qubits(self) -> tuple[int, ...]:
        return (2,)

    @property
    def checks(self) -> tuple[CodeCheck, ...]:
        return (
            CodeCheck(
                index=0,
                stabilizer=Pauli(z_qubits=(0, 1)),
                ancilla_qubit=2,
                cnot_qubits=((0, 2), (1, 2)),
            ),
        )

    @property
    def stabilizers(self) -> tuple[Pauli, ...]:
        return tuple(check.stabilizer for check in self.checks)

    @property
    def logical_observables(self) -> tuple[Pauli, ...]:
        return (Pauli(x_qubits=(0, 1)),)


def test_an_x_type_logical_observable_is_refused_rather_than_misread() -> None:
    """The terminal readout is in the Z basis, so an X memory is not expressible.

    Accepting it would produce a detector layout whose terminal detectors assert
    a determinism the readout cannot observe, so the code is refused instead.
    """

    with pytest.raises(ValueError, match="Z-type logical observable"):
        build_memory_circuit(_XMemoryCode(), rounds=2)
