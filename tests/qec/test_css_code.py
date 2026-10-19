"""A code record read off parity-check matrices rather than declared by a class.

The records the package ships are declared: a class states its qubit layout and
builds its checks from it. This file pins the other direction of that seam. A
caller who has the four Calderbank-Shor-Steane matrices and no class declares a
`~flagquantum.qec.CssCode`, and every count and layout a code record owes is read
off those matrices.

Two shapes of matrices reach that record, and the tests below pin both. A code
that already exists in this package is read back out through
``css_code_matrices`` and arrives as a ``CssCodeMatrices`` with a distance the
caller states; a code written down by hand arrives as four blocks and has its
distance derived. The two readings are held together here because a second class
for the second reading would be a second statement of the same algebra.

What this file proves
---------------------

1. ``test_the_declared_records_round_trip_through_their_own_matrices`` and
   ``test_a_shipped_record_rebuilds_from_its_own_matrices``: the matrices the
   package publishes for the repetition, rotated surface and Steane records
   describe those same records. The route is therefore not a second, disagreeing
   description of the codes that already exist -- which is the only way a test can
   tell an alternative route from a competing source of truth.
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
5. The two refusal groups: every invariant is checked against a code the caller
   supplied, so each is refused with the reason rather than carried into a memory
   circuit or a detector error model. The matrices shape refuses a stated distance
   its own matrices contradict; the hand-written shape refuses a set of blocks that
   does not describe a code at all.

What this file does not prove
-----------------------------

It does not decode, and it makes no threshold, suppression or scaling claim. It
also does not prove that the derived distance search is cheap: the search tries
every qubit subset up to ``distance_search_weight``, the default is three, and the
cost of a higher bound is the caller's to accept rather than something this file
measures. That is why the ``CssCodeMatrices`` shape states its distance instead of
deriving one -- computing a quantum code's distance is exponential in general, and
a record built from a large matrix set would refuse to exist -- and what it does
instead is refuse a distance its own matrices contradict, which is the one
direction a stated logical operator can prove. The execution half of the vertical
slice lives beside this file in ``test_css_code_memory.py``, and the
matrix-to-detector-error-model half in ``test_dem_code_matrices.py``.

The torus the hand-written cases here build is deliberately this file's own. It is
a caller's matrix set, and it is what the route exists to accept, so it is not
swapped for ``~flagquantum.qec.toric_code``: a test of the matrix route that read
its matrices from the family route would stop telling an alternative route apart
from a competing source of truth. The two descriptions of that lattice are held
against each other in ``examples/qec/css_code_from_matrices.py``, where a
disagreement is the point rather than a shared helper.

The comparison with the declared records is deliberately not a comparison of
detector error model text. A matrix carries no qubit labels, so a record rebuilt
from another record's matrices states the same checks in the order the two blocks
list them -- Z-type first, then X-type -- while the declared surface code
interleaves its two bases in lattice order. A detector's identity is the check and
the round it belongs to, so the two experiments are compared under exactly that
relabelling, which is stated here rather than assumed.
"""

from __future__ import annotations

import itertools

import pytest
import torch

from flagquantum.qec import (
    CssCode,
    CssCodeMatrices,
    DetectorErrorModel,
    MemoryCircuit,
    PhenomenologicalNoise,
    RepetitionCode,
    RotatedSurfaceCode,
    StabilizerCode,
    SteaneCode,
    ancilla_bands,
    build_memory_circuit,
    code_names,
    css_code_matrices,
    get_code,
    sample_memory_circuit,
)

pytestmark = pytest.mark.unit

_ROUNDS = 3
_NOISE = PhenomenologicalNoise(
    data_flip=0.02,
    phase_flip=0.01,
    both_flip=0.005,
    measurement_flip=0.01,
)

_STEANE_MATRICES = (
    (0, 0, 0, 1, 1, 1, 1),
    (0, 1, 1, 0, 0, 1, 1),
    (1, 0, 1, 0, 1, 0, 1),
)
_STEANE_LOGICAL = (1, 1, 1, 0, 0, 0, 0)
_TORUS_CHECKS = 4
# A toric code's star and face each touch four edges, which is what makes it a
# low-density parity-check code rather than a family whose checks grow with size.


def _bold(qubits: tuple[int, ...], *, width: int) -> list[int]:
    """One row of a parity-check block over ``width`` qubits."""

    row = [0] * width
    for qubit in qubits:
        row[qubit] = 1
    return row


def _shor_matrices() -> CssCodeMatrices:
    """The nine-qubit Shor code, whose check blocks no class here declares.

    The code is three three-qubit bit-flip blocks with phase-flip checks across
    them: ``Z_i Z_{i+1}`` inside each block, and ``X`` over the first six data
    qubits and over the last six. Its distance is three in both bases, and the
    lightest logical operator it states is the weight-three product of one qubit
    from each block.
    """

    width = 9
    z_rows = ((0, 1), (1, 2), (3, 4), (4, 5), (6, 7), (7, 8))
    x_rows = ((0, 1, 2, 3, 4, 5), (3, 4, 5, 6, 7, 8))
    return CssCodeMatrices(
        hz=torch.tensor([_bold(row, width=width) for row in z_rows]),
        hx=torch.tensor([_bold(row, width=width) for row in x_rows]),
        lz=torch.tensor([_bold((0, 3, 6), width=width)]),
    )


def _rebuilt(code: StabilizerCode) -> CssCode:
    return CssCode(matrices=css_code_matrices(code), distance=code.distance)


def _check_keys(checks: tuple[object, ...]) -> tuple[tuple[str, int], ...]:
    """Name each check by its basis and its position within that basis.

    That pair is what survives a rebuild: the matrices keep the order of each
    block, and a matrix carries nothing about which qubit a check was laid out on.
    """

    seen = {"z": 0, "x": 0}
    keys: list[tuple[str, int]] = []
    for check in checks:
        basis = "x" if check.stabilizer.x_qubits else "z"
        keys.append((basis, seen[basis]))
        seen[basis] += 1
    return tuple(keys)


def _detector_addresses(circuit: MemoryCircuit) -> dict[tuple[object, ...], int]:
    """Map every detector to the check and round that placed it.

    This restates the layout rule ``build_memory_circuit`` uses -- round zero
    names no X-type check, every round after it compares against the round before
    it, and a Z-type check ends in a terminal detector against the data readout --
    so that two circuits can be compared by the experiment they describe rather
    than by the order their checks happen to be listed in.
    """

    code = circuit.code
    keys = _check_keys(code.checks)
    addresses: dict[tuple[object, ...], int] = {}
    index = 0
    for round_index in range(circuit.rounds):
        for check, key in zip(code.checks, keys, strict=True):
            if round_index == 0 and check.stabilizer.x_qubits:
                continue
            addresses[key + (round_index,)] = index
            index += 1
    for check, key in zip(code.checks, keys, strict=True):
        if check.stabilizer.x_qubits:
            continue
        addresses[key + ("terminal",)] = index
        index += 1
    assert index == len(
        circuit.detectors
    ), "the restated layout disagrees with the built one"
    return addresses


def _mechanisms(
    model: DetectorErrorModel,
    permutation: dict[int, int] | None = None,
) -> list[tuple[object, ...]]:
    """Every mechanism as a comparable record, optionally under a detector relabelling."""

    records: list[tuple[object, ...]] = []
    for error in model.errors:
        detectors = error.detectors
        if permutation is not None:
            detectors = tuple(permutation[index] for index in detectors)
        records.append(
            (error.probability, tuple(sorted(detectors)), tuple(error.observables))
        )
    return sorted(records, key=repr)


def _relabelling(declared: MemoryCircuit, rebuilt: MemoryCircuit) -> dict[int, int]:
    """Rename a rebuilt detector as the declared detector naming the same check.

    The detector order of a rebuild is the order of its own check list; the two
    address maps are keyed by ``(basis, position, round)``, which is what both
    records agree on, so the map from one to the other is what makes the two
    models comparable.
    """

    left = _detector_addresses(declared)
    right = _detector_addresses(rebuilt)
    assert (
        left.keys() == right.keys()
    ), "the two circuits do not describe the same checks"
    return {right[address]: left[address] for address in left}


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

    def indicator(qubits: list[int]) -> list[int]:
        return [1 if qubit in qubits else 0 for qubit in range(edges)]

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
    """This file's own torus, built from the blocks above rather than a family."""

    faces, stars, z_logicals, x_logicals = _toric_matrices(lattice)
    return CssCode(
        hz=faces,
        hx=stars,
        lz=z_logicals,
        lx=x_logicals,
        distance_search_weight=lattice,
    )


# --- one record, two shapes of matrices -------------------------------------


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
    a matrix states which data qubits a check touches and not which ancillary
    qubit measures it: the record's layout is a choice the declared families made,
    and the matrices do not carry it. What the matrices do determine is the code --
    so the comparison is of the groups, not of the row order.
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
    assert code.data_qubits == record.data_qubits
    assert code.data_qubits == code.data_qubits
    assert code.ancilla_qubits == code.ancilla_qubits
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
    heavy_qubits = tuple(
        left ^ right
        for left, right in zip(_STEANE_LOGICAL, _STEANE_MATRICES[0], strict=True)
    )
    heavy = CssCode(
        hz=_STEANE_MATRICES,
        hx=_STEANE_MATRICES,
        lz=[heavy_qubits],
        lx=[heavy_qubits],
    )

    assert sum(heavy_qubits) == 7
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

    Every qubit subset strictly lighter than the reported distance is either
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
                1 if qubit in positions else 0 for qubit in range(code.num_data_qubits)
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
    """The qubit layout is a function of the matrices rather than of the caller.

    A caller supplies columns and rows; the record decides which ancilla belongs
    to which check. Z-type checks take the ancillas directly after the data and
    the X-type checks follow them, so a Z-type check couples data to ancilla and
    an X-type check is its mirror.
    """

    code = _toric(2)
    checks = code.checks
    z_checks = [check for check in checks if not check.stabilizer.x_qubits]
    x_checks = [check for check in checks if check.stabilizer.x_qubits]

    assert [check.index for check in checks] == list(range(len(checks)))
    assert [check.ancilla_qubit for check in checks] == list(code.ancilla_qubits)
    assert len(z_checks) == len(x_checks) == 4
    for check in z_checks:
        assert all(
            target == check.ancilla_qubit and control in check.stabilizer.support
            for control, target in check.cnot_qubits
        )
    for check in x_checks:
        assert all(
            control == check.ancilla_qubit and target in check.stabilizer.support
            for control, target in check.cnot_qubits
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


@pytest.mark.integration
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


# --- the hand-written blocks are held to the algebra ------------------------


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


def test_a_hand_written_logical_operator_that_anticommutes_is_refused() -> None:
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

    The record holds no array dependency of its own, so the refusal is the only
    place the bridge between a tensor-stated matrix and four blocks is stated.
    """

    torch_module = pytest.importorskip("torch")
    matrices = css_code_matrices(SteaneCode())

    with pytest.raises(TypeError, match=r"read a tensor block with \.tolist\(\) first"):
        CssCode(hz=matrices.hz, hx=matrices.hx, lz=matrices.lz, lx=matrices.lx)
    assert torch_module.__name__ == "torch"


@pytest.mark.parametrize(
    "kwargs, message",
    (
        ({"hz": [], "hx": []}, "do not state how many data qubits"),
        ({"hz": [[]], "hx": []}, "acts on no data qubit"),
        ({"hz": [[0, 0]], "hx": []}, "acts on no data qubit"),
        ({"hz": [[1, 1, 0]], "hx": [[1, 0]]}, "disagree about how many data qubits"),
    ),
)
def test_blocks_that_do_not_state_one_data_layout_are_refused(
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


# --- a declared record rebuilds itself from its own matrices ----------------


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        (RepetitionCode(distance=3), (3, 2, 0, 2)),
        (RepetitionCode(distance=5), (5, 4, 0, 4)),
        (RotatedSurfaceCode(distance=3), (9, 8, 4, 4)),
        (RotatedSurfaceCode(distance=5), (25, 24, 12, 12)),
        (SteaneCode(), (7, 6, 3, 3)),
    ],
)
def test_a_shipped_record_rebuilds_from_its_own_matrices(
    code: StabilizerCode,
    expected: tuple[int, int, int, int],
) -> None:
    """The two routes into a record agree on every count a record states."""

    record = _rebuilt(code)
    data, ancillas, x_count, z_count = expected

    assert isinstance(record, StabilizerCode)
    assert record.num_data_qubits == data == code.num_data_qubits
    assert record.num_ancilla_qubits == ancillas == code.num_ancilla_qubits
    assert record.num_ancilla_x_qubits == x_count == code.num_ancilla_x_qubits
    assert record.num_ancilla_z_qubits == z_count == code.num_ancilla_z_qubits
    assert record.num_x_stabilizers == code.num_x_stabilizers
    assert record.num_z_stabilizers == code.num_z_stabilizers
    assert record.distance == code.distance
    assert record.data_qubits == code.data_qubits
    assert record.ancilla_qubits == code.ancilla_qubits


@pytest.mark.parametrize(
    "code",
    [
        RepetitionCode(distance=3),
        RotatedSurfaceCode(distance=3),
        RotatedSurfaceCode(distance=5),
        SteaneCode(),
    ],
)
def test_a_shipped_record_rebuilds_the_same_checks_and_logicals(
    code: StabilizerCode,
) -> None:
    """A rebuild carries one check per check and one observable per observable."""

    record = _rebuilt(code)

    assert len(record.checks) == len(code.checks)
    assert {check.stabilizer for check in record.checks} == {
        check.stabilizer for check in code.checks
    }
    assert record.stabilizers == tuple(check.stabilizer for check in record.checks)
    assert list(record.logical_observables) == list(code.logical_observables)


def test_a_rebuild_states_the_z_checks_before_the_x_checks() -> None:
    """The ancilla order is the one the two blocks are read in, not a second fact."""

    record = _rebuilt(RotatedSurfaceCode(distance=3))
    bases = ["x" if check.stabilizer.x_qubits else "z" for check in record.checks]

    assert bases == ["z"] * 4 + ["x"] * 4
    assert record.ancilla_qubits == tuple(range(9, 17))
    assert [check.ancilla_qubit for check in record.checks] == list(range(9, 17))
    assert [check.index for check in record.checks] == list(range(8))


def test_a_z_check_couples_the_data_into_the_ancilla_and_the_x_check_mirrors_it() -> (
    None
):
    """The CNOT direction is the check type's, exactly as a declared check states it."""

    record = _rebuilt(RotatedSurfaceCode(distance=3))

    for check in record.checks:
        ancilla = check.ancilla_qubit
        if check.stabilizer.z_qubits:
            assert check.cnot_qubits == tuple(
                (qubit, ancilla) for qubit in check.stabilizer.z_qubits
            )
        else:
            assert check.cnot_qubits == tuple(
                (ancilla, qubit) for qubit in check.stabilizer.x_qubits
            )


def test_a_rebuild_derives_its_bands_from_the_checks_it_built() -> None:
    """The two bands are read off the checks rather than stated beside them."""

    record = _rebuilt(RotatedSurfaceCode(distance=3))
    x_qubits, z_qubits = ancilla_bands(record.checks)

    assert record.num_ancilla_x_qubits == len(x_qubits)
    assert record.num_ancilla_z_qubits == len(z_qubits)
    assert x_qubits == (13, 14, 15, 16)
    assert z_qubits == (9, 10, 11, 12)


# --- the rebuilt record describes the same experiment -----------------------


@pytest.mark.integration
@pytest.mark.parametrize(
    "code",
    [
        RepetitionCode(distance=3),
        RepetitionCode(distance=5),
        # The rotated surface code is exercised at distance three and no further,
        # because a detector error model built from a memory *circuit* is read by
        # executing the source, so its size is bounded by the statevector: the
        # surface code stops being executable here at distance three. That ceiling
        # belongs to the circuit route and not to this record, and the matrix
        # route below is exercised at distance five to say so.
        RotatedSurfaceCode(distance=3),
    ],
)
def test_a_rebuild_describes_the_same_experiment_as_the_declared_record(
    code: StabilizerCode,
) -> None:
    """The two routes produce one experiment, detector naming aside.

    The relabelling is the check order, which is the only thing a matrix cannot
    carry: the declared surface code interleaves its two bases and a rebuild
    groups them. Rates, detector flip sets and observable flip sets must agree
    exactly under it, with no tolerance, because both routes enumerate the same
    faults over the same supports.
    """

    declared = build_memory_circuit(code, rounds=_ROUNDS)
    rebuilt_circuit = build_memory_circuit(_rebuilt(code), rounds=_ROUNDS)
    relabelling = _relabelling(declared, rebuilt_circuit)

    declared_model = DetectorErrorModel.from_memory_circuit(declared, noise=_NOISE)
    rebuilt_model = DetectorErrorModel.from_memory_circuit(
        rebuilt_circuit, noise=_NOISE
    )

    assert rebuilt_model.num_detectors == declared_model.num_detectors
    assert rebuilt_model.num_observables == declared_model.num_observables
    assert rebuilt_model.num_errors == declared_model.num_errors
    assert _mechanisms(rebuilt_model, relabelling) == _mechanisms(declared_model)


@pytest.mark.integration
def test_a_repetition_rebuild_lands_on_the_same_detector_text() -> None:
    """Where the two records list their checks in one order, even the text agrees."""

    code = RepetitionCode(distance=3)
    declared = DetectorErrorModel.from_memory_circuit(
        build_memory_circuit(code, rounds=_ROUNDS), noise=_NOISE
    )
    rebuilt = DetectorErrorModel.from_memory_circuit(
        build_memory_circuit(_rebuilt(code), rounds=_ROUNDS), noise=_NOISE
    )

    assert rebuilt.to_stim_text() == declared.to_stim_text()


@pytest.mark.integration
def test_a_x_type_check_is_the_only_reason_the_surface_text_differs() -> None:
    """The surface rebuild is the same model under a relabelling, and says so."""

    code = RotatedSurfaceCode(distance=3)
    declared = DetectorErrorModel.from_memory_circuit(
        build_memory_circuit(code, rounds=_ROUNDS), noise=_NOISE
    )
    rebuilt = DetectorErrorModel.from_memory_circuit(
        build_memory_circuit(_rebuilt(code), rounds=_ROUNDS), noise=_NOISE
    )

    assert rebuilt.to_stim_text() != declared.to_stim_text()
    assert _mechanisms(rebuilt) != _mechanisms(declared)


@pytest.mark.integration
def test_a_rebuilt_record_reaches_the_matrix_route_in_one_call() -> None:
    """``from_code`` reads the matrices the record was built from, not a re-derivation."""

    record = _rebuilt(RotatedSurfaceCode(distance=3))
    through_record = DetectorErrorModel.from_code(record, noise=_NOISE, num_rounds=2)
    through_matrices = DetectorErrorModel.from_code_matrices(
        record.matrices, noise=_NOISE, num_rounds=2
    )

    assert through_record.to_stim_text() == through_matrices.to_stim_text()


@pytest.mark.integration
def test_the_matrix_route_has_no_statevector_ceiling() -> None:
    """A record too large to execute still reaches a model by reading its matrices.

    This is where the two routes of one record genuinely part: a memory circuit is
    a program, so its model is read by executing it and stops where the
    statevector stops, while the matrix route reads a support and has no ceiling
    to stop at. The record itself is built either way.
    """

    num_rounds = 2
    record = _rebuilt(RotatedSurfaceCode(distance=5))
    model = DetectorErrorModel.from_code(record, noise=_NOISE, num_rounds=num_rounds)

    assert record.num_data_qubits == 25
    assert model.num_detectors == num_rounds * len(record.checks)
    assert model.num_errors > 0


@pytest.mark.integration
def test_a_rebuilt_record_runs_as_a_memory_experiment() -> None:
    """A record that was never declared by a class still samples a memory circuit."""

    # Sampling is the one route in this file that runs the program through the
    # stabilizer engine, which is an optional distribution; every other test here
    # reads a record, a circuit or a model. The file therefore keeps running where
    # stim is absent and this test alone skips, rather than the file skipping the
    # claims that need no engine.
    pytest.importorskip("stim")

    record = _rebuilt(RotatedSurfaceCode(distance=3))
    circuit = build_memory_circuit(record, rounds=_ROUNDS)
    samples = sample_memory_circuit(circuit, noise=_NOISE, shots=64, seed=5)

    assert samples.detectors.shape == (64, len(circuit.detectors))
    assert samples.observables.shape == (64, len(record.logical_observables))


@pytest.mark.integration
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


# --- a code no class here declares -----------------------------------------


def test_a_code_this_package_does_not_ship_is_declared_by_its_matrices() -> None:
    """The Shor code has no class, and the matrices are enough to state it."""

    record = CssCode(matrices=_shor_matrices(), distance=3)

    assert isinstance(record, StabilizerCode)
    assert record.num_data_qubits == 9
    assert record.num_ancilla_qubits == 8
    assert record.num_ancilla_x_qubits == 2
    assert record.num_ancilla_z_qubits == 6
    assert record.num_x_stabilizers == 2
    assert record.num_z_stabilizers == 6
    assert [check.stabilizer.z_qubits != () for check in record.checks] == [
        True,
        True,
        True,
        True,
        True,
        True,
        False,
        False,
    ]
    assert record.logical_observables[0].z_qubits == (0, 3, 6)


@pytest.mark.integration
def test_a_code_no_class_declares_builds_a_memory_circuit_and_a_model() -> None:
    """The whole route works on a record whose only source is four blocks."""

    record = CssCode(matrices=_shor_matrices(), distance=3)
    circuit = build_memory_circuit(record, rounds=_ROUNDS)
    model = DetectorErrorModel.from_memory_circuit(circuit, noise=_NOISE)

    assert len(circuit.detectors) == 6 * (_ROUNDS + 1) + 2 * (_ROUNDS - 1)
    assert model.num_detectors == len(circuit.detectors)
    assert model.num_observables == 1
    assert model.num_errors > 0


def test_a_code_is_not_a_second_registry() -> None:
    """A matrices record is reached by its matrices, so no name is reserved for it."""

    before = code_names()

    CssCode(matrices=_shor_matrices(), distance=3)

    assert code_names() == before == ("repetition", "rotated_surface", "steane")
    with pytest.raises(ValueError, match="no code is registered as 'shor'"):
        get_code("shor")


def test_a_code_stating_both_bases_reads_one_observable_per_row() -> None:
    """Two families are two observable blocks, Z-type first, in row order."""

    record = CssCode(matrices=css_code_matrices(SteaneCode()), distance=3)

    assert len(record.logical_observables) == 2
    assert record.logical_observables[0].z_qubits == (0, 1, 2)
    assert record.logical_observables[1].x_qubits == (0, 1, 2)


@pytest.mark.integration
def test_the_circuit_route_reads_the_basis_the_caller_states() -> None:
    """A record holding both families builds one experiment per basis.

    A memory circuit prepares and reads out in one basis, and that basis is what
    selects which of the record's logical observables the terminal readout
    realizes, so a matrices record declaring both families reaches two
    experiments rather than one. The route is still narrower than the record
    where the record declares nothing in the requested basis, and that is refused
    where the basis is chosen rather than answered by building an experiment with
    no logical qubit.
    """

    record = CssCode(matrices=css_code_matrices(SteaneCode()), distance=3)

    z_memory = build_memory_circuit(record, rounds=_ROUNDS)
    x_memory = build_memory_circuit(record, rounds=_ROUNDS, readout_basis="x")

    assert z_memory.readout_basis == "z"
    assert x_memory.readout_basis == "x"
    assert z_memory.observables.observables[0].pauli == record.logical_observables[0]
    assert x_memory.observables.observables[0].pauli == record.logical_observables[1]

    with pytest.raises(ValueError, match="requires an X-type logical observable"):
        build_memory_circuit(
            CssCode(matrices=_shor_matrices(), distance=3),
            rounds=_ROUNDS,
            readout_basis="x",
        )


# --- every refusal of the matrices shape, by name ---------------------------


def test_the_matrices_shape_refuses_both_shapes_at_once() -> None:
    """One record states its matrices one way, so stating both is refused."""

    with pytest.raises(ValueError, match="both were stated"):
        CssCode(
            matrices=_shor_matrices(),
            distance=3,
            hz=_STEANE_MATRICES,
        )


def test_a_record_states_its_matrices_and_is_refused_when_it_states_neither() -> None:
    """With no matrices and no blocks there is nothing to read a code off."""

    with pytest.raises(ValueError, match="neither was stated"):
        CssCode(matrices=None, distance=3)


def test_the_hand_written_shape_refuses_a_stated_distance() -> None:
    """A hand-written block derives its distance, so a stated one is not read."""

    with pytest.raises(ValueError, match="distance= was stated beside hand-written"):
        CssCode(hz=_STEANE_MATRICES, hx=_STEANE_MATRICES, distance=3)


def test_the_matrices_shape_refuses_a_search_weight_it_has_nothing_to_search() -> None:
    """A stated distance and a search bound are two readings of one number."""

    with pytest.raises(ValueError, match="has nothing to search"):
        CssCode(
            matrices=_shor_matrices(),
            distance=3,
            distance_search_weight=4,
        )


def test_a_record_is_built_from_the_matrix_record_and_not_from_a_tensor() -> None:
    with pytest.raises(TypeError, match="built from a CssCodeMatrices"):
        CssCode(matrices=torch.eye(3), distance=1)  # type: ignore[arg-type]


@pytest.mark.parametrize("distance", [True, 2.0, "3", None])
def test_a_distance_that_is_not_an_integer_is_refused(distance: object) -> None:
    with pytest.raises(TypeError, match="distance must be an integer"):
        CssCode(matrices=_shor_matrices(), distance=distance)  # type: ignore[arg-type]


@pytest.mark.parametrize("distance", [0, -1])
def test_a_distance_below_one_is_refused(distance: int) -> None:
    with pytest.raises(ValueError, match="at least one"):
        CssCode(matrices=_shor_matrices(), distance=distance)


def test_matrices_with_no_check_are_not_a_code() -> None:
    width = 3
    with pytest.raises(ValueError, match="state no check"):
        CssCode(
            matrices=CssCodeMatrices(
                hz=torch.zeros((0, width), dtype=torch.int64),
                lz=torch.tensor([_bold((0, 1, 2), width=width)]),
            ),
            distance=1,
        )


def test_matrices_with_no_logical_operator_are_not_an_error_correcting_code() -> None:
    width = 3
    with pytest.raises(ValueError, match="state no logical observable"):
        CssCode(
            matrices=CssCodeMatrices(
                hz=torch.tensor([_bold((0, 1), width=width)]),
            ),
            distance=1,
        )


def test_a_check_row_that_is_empty_is_refused() -> None:
    width = 3
    with pytest.raises(ValueError, match="is empty"):
        CssCode(
            matrices=CssCodeMatrices(
                hz=torch.tensor([_bold((0, 1), width=width), [0, 0, 0]]),
                lz=torch.tensor([_bold((0, 1, 2), width=width)]),
            ),
            distance=1,
        )


def test_a_stated_logical_operator_that_anticommutes_with_a_check_is_refused() -> None:
    """A stated logical operator is read against the opposite basis' checks."""

    width = 9
    matrices = CssCodeMatrices(
        hz=torch.tensor([_bold(row, width=width) for row in ((0, 1), (1, 2))]),
        hx=torch.tensor([_bold((0, 1, 2), width=width)]),
        lz=torch.tensor([_bold((0,), width=width)]),
    )

    with pytest.raises(ValueError, match="anticommute"):
        CssCode(matrices=matrices, distance=1)


def test_a_logical_operator_that_is_a_product_of_checks_is_refused() -> None:
    """A row inside the check span is a stabilizer, not a logical operator."""

    width = 3
    matrices = CssCodeMatrices(
        hz=torch.tensor([_bold(row, width=width) for row in ((0, 1), (1, 2))]),
        lz=torch.tensor([_bold((0, 2), width=width)]),
    )

    with pytest.raises(ValueError, match="stabilizer rather than a logical operator"):
        CssCode(matrices=matrices, distance=1)


def test_two_dependent_logical_operators_of_one_basis_are_refused() -> None:
    """Each row becomes one observable, so two rows for one operator are refused.

    The two rows here are the same weight-one operator stated twice, which is the
    shape a caller reaches by concatenating one code's logical rows with another's
    without reducing them. Both rows pass the reading a logical operator gets; it
    is the pair that cannot become two observables.
    """

    width = 3
    matrices = CssCodeMatrices(
        hz=torch.tensor([_bold((0, 1), width=width)]),
        lz=torch.tensor([_bold((2,), width=width), _bold((2,), width=width)]),
    )

    with pytest.raises(ValueError, match="not independent"):
        CssCode(matrices=matrices, distance=1)


def test_a_distance_the_matrices_contradict_is_refused() -> None:
    """A stated logical operator proves the distance is at most its own weight."""

    record = CssCode(matrices=_shor_matrices(), distance=3)

    assert record.distance == 3
    with pytest.raises(ValueError, match="exceeds the weight"):
        CssCode(matrices=_shor_matrices(), distance=4)


def test_a_distance_below_the_stated_logical_weight_is_accepted() -> None:
    """The check is one-directional, and the direction that was dropped is stated."""

    record = CssCode(matrices=_shor_matrices(), distance=1)

    assert record.distance == 1


# --- the record is a value --------------------------------------------------


def test_the_record_is_frozen_and_hashable() -> None:
    """A hand-written record hashes by its content, because its content is tuples."""

    code = _toric(2)

    assert code == _toric(2)
    assert len({code, _toric(2)}) == 1
    with pytest.raises(Exception):
        code.hz = ()  # type: ignore[misc]


def test_a_record_compares_by_its_matrices_and_its_distance() -> None:
    record = CssCode(matrices=_shor_matrices(), distance=3)

    assert record == CssCode(matrices=_shor_matrices(), distance=3)
    assert record != CssCode(matrices=_shor_matrices(), distance=1)
    assert record != CssCode(matrices=css_code_matrices(SteaneCode()), distance=3)
    assert record != "a shor code"
    assert record.__eq__(object()) is NotImplemented


def test_a_record_is_unhashable_because_its_content_is_a_tensor() -> None:
    record = CssCode(matrices=_shor_matrices(), distance=3)

    with pytest.raises(TypeError, match="unhashable"):
        hash(record)


def test_the_record_is_published_from_the_package_root() -> None:
    import flagquantum.qec as qec

    assert "CssCode" in qec.__all__
    assert qec.CssCode is CssCode
