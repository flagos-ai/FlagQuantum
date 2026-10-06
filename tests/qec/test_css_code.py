"""Unit coverage for the code record built from a parity-check matrix.

Six records in this package are written down as a family -- a repetition
lattice, a rotated surface lattice, the Steane code, the triangular colour patch,
the square-lattice torus, the ZXXZ surface patch -- and each states its own
checks, its own logical operators and its own distance, the colour patch and the
torus deriving them from a rule about their lattice rather than tabulating them.
This record is the sixth route: the matrices are the input,
so a code this package never wrote down can still be a
`~flagquantum.qec.StabilizerCode` and walk the rest of the path. The tests here
pin the algebra that route relies on and the refusals that keep a set of matrices
from being read as a code it is not.

The torus the hand-written cases here build is deliberately this file's own. It
is a caller's matrix set, and it is what the route exists to accept, so it is not
swapped for `~flagquantum.qec.toric_code`: a test of the matrix route that read
its matrices from the family route would stop telling an alternative route apart
from a competing source of truth. The two descriptions of that lattice are held
against each other in `examples/qec/css_code_from_matrices.py`, where a
disagreement is the point rather than a shared helper.

What this file proves
---------------------

1. ``test_the_declared_records_round_trip_through_their_own_matrices``: the
   matrices the package publishes for the repetition, rotated surface and Steane
   records describe those same records. The route is therefore not a second,
   disagreeing description of the codes that already exist -- which is the only
   way a test can tell an alternative route from a competing source of truth.
2. ``test_the_distance_is_a_property_of_the_code_and_not_of_the_representative``:
   the Steane code's logical operator declared at weight three and the same
   operator multiplied by a stabilizer of weight four give one pair of distances.
   A distance read off the declared representative would report seven and three;
   this one reports three and three, because it searches the code.
3. ``test_the_repetition_profile_separates_its_two_family_distances``: a
   repetition record's own ``distance`` is the bit-flip distance it is named for,
   while its Z-type distance is one. Both numbers are correct statements about
   different quantities, and this is the case that makes reporting a single
   distance ambiguous unless the record says which.
4. ``test_a_qldpc_family_this_package_does_not_declare_satisfies_the_protocol``:
   the toric code on an ``L``-by-``L`` torus, whose checks are local but whose
   logical operators are not, and which no record here declares. It satisfies the
   protocol with two logical qubits and a distance that grows with ``L``, which is
   the claim the route exists to make.
5. The refusal group: every invariant is checked against a code the caller
   supplied, so each is refused with the reason rather than carried into a memory
   circuit or a detector error model.

What this file does not prove
-----------------------------

It does not decode, and it makes no threshold, suppression or scaling claim. It
also does not prove that the route's distance search is cheap: the search tries
every wire subset up to ``distance_search_weight``, the default is three, and the
cost of a higher bound is the caller's to accept rather than something this file
measures. The execution half of the vertical slice lives beside this file in
``test_css_code_memory.py``, and the matrix-to-detector-error-model half in
``test_dem_code_matrices.py``.
"""

from __future__ import annotations

import itertools

import pytest

from flagquantum.qec.circuit import MemoryCircuit, build_memory_circuit
from flagquantum.qec.codes import (
    CssCode,
    RepetitionCode,
    StabilizerCode,
    SteaneCode,
)
from flagquantum.qec.dem_construction import css_code_matrices
from flagquantum.qec.surface import RotatedSurfaceCode

pytestmark = pytest.mark.unit

_STEANE_MATRICES = (
    (0, 0, 0, 1, 1, 1, 1),
    (0, 1, 1, 0, 0, 1, 1),
    (1, 0, 1, 0, 1, 0, 1),
)
_STEANE_LOGICAL = (1, 1, 1, 0, 0, 0, 0)
_TORUS_CHECKS = 4
# A toric code's star and face each touch four edges, which is what makes it a
# low-density parity-check code rather than a family whose checks grow with size.


def _gf2_rank(rows: list[list[int]]) -> int:
    """Return the rank of a binary matrix over GF(2), by elimination."""

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


def _in_span(vector: tuple[int, ...], rows: list[list[int]]) -> bool:
    """Whether ``vector`` is a GF(2) combination of ``rows``."""

    return _gf2_rank([*rows, list(vector)]) == _gf2_rank(rows)


def _odd_overlap(left: tuple[int, ...], right: tuple[int, ...]) -> bool:
    return sum(a * b for a, b in zip(left, right, strict=True)) % 2 == 1


def _toric_matrices(lattice: int) -> tuple[list[list[int]], ...]:
    """Return the checks and a logical basis of the toric code on a torus.

    One qubit sits on each edge: ``H(i, j)`` is the horizontal edge from vertex
    ``(i, j)`` to ``(i, j + 1)`` and ``V(i, j)`` the vertical edge from ``(i, j)``
    to ``(i + 1, j)``, wrapping at the boundary. A star is the four edges at one
    vertex and a face the four around one cell, so the Z-type checks are the
    faces and the X-type checks are the stars. Each family holds ``L * L``
    checks over ``2 * L * L`` edges, its rows have rank ``L * L - 1`` because the
    product of every star -- and of every face -- is the identity, so the code has
    ``2`` logical qubits, and the logical operators are the non-contractible
    cycles, of which the shortest has weight ``L``.
    """

    edges = 2 * lattice * lattice

    def horizontal(row: int, column: int) -> int:
        return 2 * ((row % lattice) * lattice + (column % lattice))

    def vertical(row: int, column: int) -> int:
        return horizontal(row, column) + 1

    def indicator(wires: list[int]) -> list[int]:
        return [1 if wire in wires else 0 for wire in range(edges)]

    faces = [
        indicator(
            [
                horizontal(row, column),
                horizontal(row + 1, column),
                vertical(row, column),
                vertical(row, column + 1),
            ]
        )
        for row in range(lattice)
        for column in range(lattice)
    ]
    stars = [
        indicator(
            [
                horizontal(row, column),
                horizontal(row, column - 1),
                vertical(row, column),
                vertical(row - 1, column),
            ]
        )
        for row in range(lattice)
        for column in range(lattice)
    ]
    horizontal_cycle = [horizontal(0, column) for column in range(lattice)]
    vertical_cycle = [vertical(row, 0) for row in range(lattice)]
    z_logicals = [indicator(horizontal_cycle), indicator(vertical_cycle)]
    x_logicals = [
        indicator([horizontal(row, 0) for row in range(lattice)]),
        indicator([vertical(0, column) for column in range(lattice)]),
    ]
    return faces, stars, z_logicals, x_logicals


def _toric(lattice: int) -> CssCode:
    faces, stars, z_logicals, x_logicals = _toric_matrices(lattice)
    return CssCode(
        hz=faces,
        hx=stars,
        lz=z_logicals,
        lx=x_logicals,
        distance_search_weight=lattice,
    )


@pytest.mark.parametrize(
    "record",
    (
        RepetitionCode(3),
        RepetitionCode(5),
        RotatedSurfaceCode(distance=3),
        SteaneCode(),
    ),
)
def test_the_declared_records_round_trip_through_their_own_matrices(
    record: StabilizerCode,
) -> None:
    """The published matrices and the declared record describe one code.

    The matrices are read out of the record and back into this route, and the
    resulting record must state the same stabilizer group and the same logical
    operators. Anything less would leave two descriptions of the same code free to
    drift apart.

    The check *order* and the ancilla each check owns are the route's own, because
    a matrix states which data qubits a check touches and not which ancillary wire
    measures it: the record's layout is a choice the declared families made, and
    the matrices do not carry it. What the matrices do determine is the code -- so
    the comparison is of the groups, not of the row order.
    """

    matrices = css_code_matrices(record)
    code = CssCode(
        hz=matrices.hz.tolist(),
        hx=matrices.hx.tolist(),
        lz=matrices.lz.tolist(),
        lx=matrices.lx.tolist(),
        distance_search_weight=record.distance,
    )

    assert code.num_data_qubits == record.num_data_qubits
    assert code.num_ancilla_qubits == record.num_ancilla_qubits
    assert code.data_wires == record.data_wires
    assert len(code.checks) == len(record.checks)
    assert sorted(code.stabilizers) == sorted(record.stabilizers)
    assert sorted(code.logical_observables) == sorted(record.logical_observables)


def test_the_distance_is_a_property_of_the_code_and_not_of_the_representative() -> None:
    """Two stabilizer-equivalent representatives report one pair of distances.

    Multiplying the declared logical operator by a weight-four stabilizer gives a
    weight-seven operator that is just as much a logical operator. A distance read
    off the declared representative would say seven; this record searches the code
    and says three, which is the number the code has.
    """

    light = CssCode(
        hz=_STEANE_MATRICES,
        hx=_STEANE_MATRICES,
        lz=[_STEANE_LOGICAL],
        lx=[_STEANE_LOGICAL],
    )
    heavy_wires = tuple(
        left ^ right
        for left, right in zip(_STEANE_LOGICAL, _STEANE_MATRICES[0], strict=True)
    )
    heavy = CssCode(
        hz=_STEANE_MATRICES,
        hx=_STEANE_MATRICES,
        lz=[heavy_wires],
        lx=[heavy_wires],
    )

    assert sum(heavy_wires) == 7
    assert (light.x_distance, light.z_distance) == (3, 3)
    assert (heavy.x_distance, heavy.z_distance) == (3, 3)
    assert heavy.logical_observables != light.logical_observables


def test_the_repetition_profile_separates_its_two_family_distances() -> None:
    """A repetition code's declared distance and its Z-type distance differ.

    The record's ``distance`` is the bit-flip distance the family is named for,
    which is the number of data qubits. Its Z-type distance is one, because a
    single phase flip commutes with every check and lies outside their span. Both
    statements are about the same code and neither is wrong, which is why this
    record reports the two families and then the smaller of them.
    """

    for qubits in (3, 5):
        record = RepetitionCode(qubits)
        matrices = css_code_matrices(record)
        code = CssCode(
            hz=matrices.hz.tolist(),
            hx=matrices.hx.tolist(),
            lz=matrices.lz.tolist(),
            lx=matrices.lx.tolist(),
            distance_search_weight=qubits,
        )

        assert code.x_distance == record.distance == qubits
        assert code.z_distance == 1
        assert code.distance == min(code.x_distance, code.z_distance) == 1


def test_a_qldpc_family_this_package_does_not_declare_satisfies_the_protocol() -> None:
    """The toric code is a code record this package never wrote down.

    Nothing here declares a toric family, and the record below is built from
    matrices alone. Its checks are local -- four data qubits each, however large
    the lattice -- while its logical operators wrap the torus, so it is the shape
    the old route could not admit: a family whose logical weight grows with the
    lattice rather than with a declared constant.
    """

    for lattice in (2, 3):
        code = _toric(lattice)

        assert isinstance(code, StabilizerCode)
        assert code.num_data_qubits == 2 * lattice * lattice
        assert code.num_ancilla_qubits == 2 * lattice * lattice
        assert len(code.checks) == 2 * lattice * lattice
        assert {check.stabilizer.weight for check in code.checks} == {_TORUS_CHECKS}
        assert len(code.logical_observables) == 4
        for check in code.checks:
            for other in code.checks:
                assert check.stabilizer.commutes_with(other.stabilizer)

    assert _toric(3).distance > _toric(2).distance


@pytest.mark.parametrize("lattice", (2, 3))
def test_the_reported_distance_is_the_lightest_logical_weight(lattice: int) -> None:
    """No operator lighter than the reported distance is a logical operator.

    Every wire subset strictly lighter than the reported distance is either
    excluded by the opposite family's checks or lies in its own family's span,
    which is the definition of not being a logical operator. The subsets are
    enumerated here rather than taken from the record, so the claim and its check
    do not share an implementation.
    """

    faces, stars, z_logicals, x_logicals = _toric_matrices(lattice)
    code = _toric(lattice)
    z_rows = [list(row) for row in faces]
    x_rows = [list(row) for row in stars]

    lighter = 0
    for weight in range(1, code.distance):
        for positions in itertools.combinations(range(code.num_data_qubits), weight):
            vector = tuple(
                1 if wire in positions else 0 for wire in range(code.num_data_qubits)
            )
            for rows, own_rows in ((x_rows, z_rows), (z_rows, x_rows)):
                if any(_odd_overlap(vector, row) for row in rows):
                    continue
                assert _in_span(vector, own_rows), (
                    f"a weight-{weight} operator outside its own family's span "
                    "commutes with every check, which would be lighter than the "
                    "reported distance"
                )
            lighter += 1

    assert lighter > 0
    assert code.x_distance == lattice
    assert code.z_distance == lattice
    assert code.distance == lattice
    # The declared logical operators are themselves at the reported weight, so the
    # distance is attained rather than merely bounded from below.
    assert {operator.weight for operator in code.logical_observables} == {lattice}
    assert sum(sum(row) for row in z_logicals) == 2 * lattice
    assert sum(sum(row) for row in x_logicals) == 2 * lattice


def test_the_z_type_checks_come_first_and_own_the_lower_ancillas() -> None:
    """The wire layout is a function of the matrices rather than of the caller.

    A caller supplies columns and rows; the record decides which ancilla belongs
    to which check. Z-type checks take the ancillas directly after the data and
    the X-type checks follow them, so a Z-type check couples data to ancilla and
    an X-type check is its mirror.
    """

    code = _toric(2)
    checks = code.checks
    z_checks = [check for check in checks if not check.stabilizer.x_wires]
    x_checks = [check for check in checks if check.stabilizer.x_wires]

    assert [check.index for check in checks] == list(range(len(checks)))
    assert [check.ancilla_wire for check in checks] == list(code.ancilla_wires)
    assert len(z_checks) == len(x_checks) == 4
    for check in z_checks:
        assert all(
            target == check.ancilla_wire and control in check.stabilizer.support
            for control, target in check.cnot_wires
        )
    for check in x_checks:
        assert all(
            control == check.ancilla_wire and target in check.stabilizer.support
            for control, target in check.cnot_wires
        )


def test_the_blocks_are_normalized_to_rows_of_integers() -> None:
    """The validated matrices are the ones the record keeps and states back."""

    code = CssCode(
        hz=[list(row) for row in _STEANE_MATRICES],
        hx=[list(row) for row in _STEANE_MATRICES],
        lz=[list(_STEANE_LOGICAL)],
        lx=[list(_STEANE_LOGICAL)],
    )

    assert code.hz == _STEANE_MATRICES
    assert code.hx == _STEANE_MATRICES
    assert code.lz == (_STEANE_LOGICAL,)
    assert code.lx == (_STEANE_LOGICAL,)
    assert all(type(entry) is int for row in code.hz for entry in row)


def test_a_check_ancilla_is_allocated_even_when_no_logical_is_declared() -> None:
    """A code may be read for its checks alone, and the memory path says so.

    The matrices of a code need not come with a logical operator: a detector error
    model is read from the checks. This record admits that, while the memory
    circuit still refuses it, because a memory experiment readout needs a logical
    operator and building one silently would state an experiment nobody asked for.
    """

    code = CssCode(hz=_STEANE_MATRICES, hx=_STEANE_MATRICES)

    assert code.logical_observables == ()
    assert code.num_ancilla_qubits == 6
    assert code.distance == 3
    with pytest.raises(ValueError, match="Z-type logical observable"):
        build_memory_circuit(code, rounds=1)


def test_the_record_is_frozen_and_hashable() -> None:
    code = _toric(2)

    assert code == _toric(2)
    assert len({code, _toric(2)}) == 1
    with pytest.raises(Exception):
        code.hz = ()  # type: ignore[misc]


def test_a_pair_of_check_families_that_does_not_commute_is_refused() -> None:
    """Two blocks that anticommute generate no stabilizer group."""

    with pytest.raises(ValueError, match="do not commute"):
        CssCode(hz=[[1, 1, 0, 0]], hx=[[1, 0, 1, 0]])


def test_checks_that_leave_no_logical_qubit_are_refused() -> None:
    """A full-rank pair of blocks describes a stabilizer state, not a code."""

    with pytest.raises(ValueError, match="no logical qubit"):
        CssCode(hz=[[1, 1]], hx=[[1, 1]])


def test_a_logical_family_of_the_wrong_length_is_refused() -> None:
    """A logical family has one operator per logical qubit, no more and no fewer."""

    with pytest.raises(ValueError, match="declares 2 logical operator"):
        CssCode(
            hz=_STEANE_MATRICES,
            hx=_STEANE_MATRICES,
            lz=[_STEANE_LOGICAL, _STEANE_MATRICES[0]],
            lx=[_STEANE_LOGICAL],
        )


def test_a_logical_operator_inside_its_own_family_span_is_refused() -> None:
    """A stabilizer is a fixed outcome, not information."""

    with pytest.raises(ValueError, match="combination of the hz rows"):
        CssCode(
            hz=_STEANE_MATRICES,
            hx=_STEANE_MATRICES,
            lz=[_STEANE_MATRICES[0]],
            lx=[_STEANE_LOGICAL],
        )


def test_a_logical_operator_that_anticommutes_with_a_check_is_refused() -> None:
    """An operator that anticommutes with a check leaves the code space."""

    with pytest.raises(ValueError, match="anticommutes with hx row"):
        CssCode(
            hz=_STEANE_MATRICES,
            hx=_STEANE_MATRICES,
            lz=[(1, 0, 0, 0, 0, 0, 0)],
            lx=[_STEANE_LOGICAL],
        )


def test_a_degenerate_logical_pairing_is_refused() -> None:
    """A singular pairing matrix generates a smaller logical group than k says.

    Both blocks below leave two logical qubits, and two Z-type operators are
    declared against two X-type operators whose pairing matrix has rank one, so
    the four rows generate the group of one logical qubit. The count of logical
    qubits is the rank of the pairing, and this record refuses the mismatch
    instead of reporting a code with more logical operators than it has.
    """

    with pytest.raises(ValueError, match="pair up\\s+degenerately"):
        CssCode(
            hz=[[1, 1, 0, 0], [0, 0, 1, 1]],
            hx=[],
            lz=[[1, 0, 0, 0], [0, 1, 0, 0]],
            lx=[[1, 1, 0, 0], [0, 0, 1, 1]],
        )


def test_a_search_that_never_reaches_the_distance_is_refused() -> None:
    """The record does not report a distance nothing checked."""

    with pytest.raises(ValueError, match="no X-type logical operator"):
        CssCode(hz=[[1, 1, 1, 1, 1]], hx=[], distance_search_weight=1)


@pytest.mark.parametrize("entry", (True, 2, -1, 1.0, "1", None))
def test_a_matrix_entry_that_is_not_a_zero_or_a_one_is_refused(entry: object) -> None:
    with pytest.raises((TypeError, ValueError), match="one 0-or-1 entry"):
        CssCode(hz=[[1, entry]], hx=[])


@pytest.mark.parametrize("block", (5, "0101", (1, 0, 1), [[1], 7]))
def test_a_block_that_is_not_a_matrix_is_refused(block: object) -> None:
    with pytest.raises(TypeError, match="sequence of sequences|sequence of 0s and 1s"):
        CssCode(hz=block, hx=[])


def test_a_tensor_block_names_the_bridge_that_reads_it() -> None:
    """A tensor is not a sequence, and the refusal says how to read one.

    This module holds no array dependency, so the refusal is the only place the
    bridge between a tensor-stated matrix and this record is stated.
    """

    torch = pytest.importorskip("torch")
    matrices = css_code_matrices(SteaneCode())

    with pytest.raises(TypeError, match=r"read a tensor block with \.tolist\(\) first"):
        CssCode(hz=matrices.hz, hx=matrices.hx, lz=matrices.lz, lx=matrices.lx)
    assert torch.__name__ == "torch"


@pytest.mark.parametrize(
    "kwargs, message",
    (
        ({"hz": [], "hx": []}, "do not state how many data qubits"),
        ({"hz": [[]], "hx": []}, "acts on no data qubit"),
        ({"hz": [[0, 0]], "hx": []}, "acts on no data qubit"),
        ({"hz": [[1, 1, 0]], "hx": [[1, 0]]}, "disagree about how many data qubits"),
    ),
)
def test_matrices_that_do_not_state_one_data_layout_are_refused(
    kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        CssCode(**kwargs)


@pytest.mark.parametrize("weight", (True, 2.0, "3", None))
def test_the_search_weight_must_be_an_integer(weight: object) -> None:
    with pytest.raises(TypeError, match="must be an integer"):
        CssCode(hz=_STEANE_MATRICES, hx=_STEANE_MATRICES, distance_search_weight=weight)


def test_the_search_weight_must_be_positive() -> None:
    with pytest.raises(ValueError, match="at least one"):
        CssCode(hz=_STEANE_MATRICES, hx=_STEANE_MATRICES, distance_search_weight=0)


def test_the_memory_circuit_detector_count_follows_the_two_check_classes() -> None:
    """The route reaches the memory path without a second layout definition.

    A Z-type check is compared across adjacent rounds and against the first and
    last round, while an X-type check carries no comparable raw bit at the round
    boundary, so the two classes contribute different detector counts. The toric
    code has two logical qubits, so its memory experiment carries two observables
    rather than the one a repetition or surface record carries.
    """

    code = _toric(2)
    faces, stars = 4, 4

    for rounds in (1, 2, 3):
        memory = build_memory_circuit(code, rounds=rounds)

        assert isinstance(memory, MemoryCircuit)
        assert len(memory.detectors) == faces * (rounds + 1) + stars * (rounds - 1)
        assert len(memory.observables) == 2
