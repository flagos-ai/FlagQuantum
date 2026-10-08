"""Unit coverage for the subsystem-code record and the tesseract family.

``test_css_code.py`` pins the algebra of a code a caller writes down as matrices,
and ``test_colour_code.py`` pins a family whose matrices are derived from a
lattice. This file pins the record whose matrices are *not enough* to state the
code: a subsystem code, where four blocks describe the checks and the logical
operators and two more describe the gauge that decides which logical classes are
protected.

The record is worth a file of its own because the reading of the same matrices
changes with it. The tesseract's sixteen data qubits carry ten independent check
rows and two anticommuting gauge pairs, so the checks leave six classes and the
gauge claims two of them: the record reports four protected qubits, while the
Calderbank-Shor-Steane reading of the very same check matrices is a code with six
logical qubits. Neither reading is wrong and they are not interchangeable, which
is why the second one is a second record rather than two optional fields on the
first, and why the fail-closed property below is part of the claim rather than a
side effect.

What this file proves
---------------------

1. ``test_the_declared_parameters_are_the_ones_the_matrices_derive``: sixteen
   data qubits, ten stabilizer generators, two gauge qubits and four protected
   qubits, with the protected count equal to ``n - s - r`` rather than stated.
2. ``test_the_check_rank_difference_is_not_the_protected_count``: ``16 - 5 - 5``
   is six, so the number the record reports cannot be read off the check blocks
   alone; the gauge is what turns six into four.
3. ``test_every_check_is_weight_eight_and_every_gauge_generator_is_weight_four``:
   the check table is the hypercube's cell incidence and the gauge table its
   faces, which is what makes the two blocks the ones the family is published
   with rather than four matrices that happen to work.
4. ``test_the_checks_are_the_center_of_the_gauge_group``: every check commutes
   with every gauge generator, which is the condition that makes a check's
   outcome well defined when the gauge is measured alongside it.
5. ``test_the_gauge_generators_pair_up_nondegenerately``: the pairing matrix of
   the two gauge blocks is the two-by-two identity, and each gauge generator is
   independent of its own family's checks.
6. ``test_the_declared_logicals_are_dressed_operators``: every declared logical
   operator commutes with the opposite checks and lies outside its own family's
   checks *and* gauge, which is the only reading in which a subsystem code has a
   distance at all.
7. ``test_the_distance_is_four_under_both_readings``: both family distances are
   four, the same minimum appears when the operator is additionally required to
   commute with the X-type gauge generators, and the same matrices with a search
   bound of three are refused, so four is a number the record found rather than
   one it was told.
8. ``test_the_distance_is_dressed_and_not_the_check_only_reading``: the 3x3
   Bacon-Shor is the smallest family in which the two readings of one matrix set
   disagree, and there a search up to the checks alone finds a weight-two operator
   while the record reports the family's dressed distance of three, so "the
   distance is dressed" is a measured claim rather than one the tesseract's four
   happens to satisfy either way.
9. ``test_the_record_refuses_matrices_that_are_not_a_subsystem_code``: twelve
   malformed records are refused, each for a named reason -- a gauge generator
   that anticommutes with a check, a gauge pair whose members commute, a gauge
   generator inside its own check span, a logical operator inside the gauge span,
   a logical operator that does not commute with the opposite checks, and the
   shape and type errors of the blocks.
10. ``test_the_css_reading_of_the_same_matrices_is_a_different_code``: the CSS
    record refuses the four declared logicals of a code it counts six of, and the
    route that turns a record into CSS matrices refuses this record outright
    instead of building the matrices the gauge would be missing from.
11. ``test_the_column_swap_is_an_automorphism_of_the_gauge_group``: the
    sixteen-wire permutation the family publishes maps all 16384 elements of the
    gauge group onto the gauge group, which is the group-theoretic statement of
    "the automorphism preserves the gauge subsystem" that the family is used for.
12. ``test_the_column_swap_implements_two_cnots_on_the_declared_logical_order``:
    the same permutation acts on the four declared logical pairs exactly as the
    pairs the family publishes say ``CNOT(control -> target)`` does -- the
    control's X operator gains the target's and the target's Z operator gains the
    control's, with nothing else moving -- which is the free logical gate the
    family is published for.
13. ``test_the_published_permutation_is_the_layout_involution``: the permutation
    is read off the hypercube layout rather than transcribed, and the derivation
    is held against the tuple the framework this package replaces publishes, so
    "derived" is a measured claim and not a claim about how the code looks.
14. ``test_the_public_namespace_publishes_the_record``: ``flagquantum.qec``
    resolves the record, the family, the permutation and the declared action and
    names all four in its ``__all__``.

What this file does not prove
-----------------------------

It builds no memory experiment, measures no noise, and makes no decoding,
threshold or logical-suppression claim: a subsystem code is decoded by measuring
its gauge generators, and this package has no round structure, detector model or
decoder registered for that protocol, so the record is a parameter and generator
record rather than an executable one. It also does not prove that four is the
distance of the tesseract, or three that of the 3x3 Bacon-Shor, by an argument
independent of the search, and it does not exhibit the parent stabilizer code of
six logical qubits whose checks the tesseract's are. Point 12 pins the action of
one permutation on the declared operators, not the existence of a circuit that
realizes it on the data qubits.
"""

from __future__ import annotations

import itertools

import pytest

from flagquantum.qec import (
    CssCode,
    StabilizerCode,
    SubsystemCode,
    css_code_matrices,
    tesseract_code,
    tesseract_column_swap,
    tesseract_free_cnot_pairs,
)
from flagquantum.qec.gf2 import in_span, rank, reduce_rows, reduce_vector
from flagquantum.qec.pauli import Pauli

pytestmark = pytest.mark.unit

# The permutation the framework this package replaces publishes for the same
# family, transcribed from ``cudaq/logical/qec/reichardt.py`` where it is the
# ``column_swap`` constant of the free-CNOT gadget. It is an external statement,
# so it is repeated here rather than read out of the module under test: point 13
# holds the module's derivation against it, and a value read back from the same
# code that computed it would agree with that code whatever either said.
_CUDAQ_COLUMN_SWAP = (
    2,
    1,
    0,
    3,
    6,
    5,
    4,
    7,
    10,
    9,
    8,
    11,
    14,
    13,
    12,
    15,
)

# The hypercube's eight cells as the sixteen vertices they touch, and the two
# anticommuting gauge pairs. The tables are repeated here rather than read off
# the record so that point 3 can compare the record's generators against an
# independent statement of the lattice instead of against itself.
_CHECKS = (
    (0, 1, 2, 3, 4, 5, 6, 7),
    (0, 1, 2, 3, 8, 9, 10, 11),
    (0, 1, 4, 5, 8, 9, 12, 13),
    (0, 2, 4, 6, 8, 10, 12, 14),
    (8, 9, 10, 11, 12, 13, 14, 15),
)
_Z_GAUGES = ((1, 5, 9, 13), (0, 1, 2, 3))
_X_GAUGES = ((8, 9, 10, 11), (0, 4, 8, 12))

# The 3x3 Bacon-Shor code, which is the second fixture because it is the smallest
# subsystem family in which the dressed and the bare reading of the same matrices
# disagree. Wire ``3 * row + column``. The checks are the products of a full pair of
# rows or of columns; the gauge table is four representatives of the gauge span
# modulo the checks rather than the family's six raw neighbouring pairs, because a
# gauge *qubit* is a pair and six raw pairs of a 3x3 lattice pair up with rank four.
# The recorded tables were derived from the lattice and re-measured here; the one
# declared Z-type logical is a row and the one X-type logical is a column.
_BACONSHOR_HZ = ((0, 1, 2, 3, 4, 5), (0, 1, 2, 6, 7, 8))
_BACONSHOR_HX = ((0, 1, 3, 4, 6, 7), (0, 2, 3, 5, 6, 8))
_BACONSHOR_GZ = ((0, 3), (1, 4), (3, 6), (4, 7))
_BACONSHOR_GX = ((0, 1), (1, 2), (3, 4), (4, 5))
_BACONSHOR_LZ = ((0, 1, 2),)
_BACONSHOR_LX = ((0, 3, 6),)
_BACONSHOR_WIDTH = 9


def _mask(wires: tuple[int, ...]) -> int:
    """Return the bit vector of a wire collection, bit ``position`` for wire it."""

    vector = 0
    for wire in wires:
        vector |= 1 << wire
    return vector


def _permute(vector: int, mapping: tuple[int, ...]) -> int:
    """Return ``vector`` with each wire ``position`` moved to ``mapping[position]``."""

    image = 0
    for position in range(len(mapping)):
        if vector >> position & 1:
            image |= 1 << mapping[position]
    return image


def _row(wires: tuple[int, ...]) -> tuple[int, ...]:
    """Return a wire collection as a 0-or-1 row of the Bacon-Shor fixture's width."""

    return tuple(1 if wire in wires else 0 for wire in range(_BACONSHOR_WIDTH))


def _supports(pauli: Pauli) -> tuple[int, ...]:
    """Return the wires a Pauli acts on with a non-identity factor."""

    return tuple(sorted(set(pauli.x_wires) | set(pauli.z_wires)))


def _z_logicals(code: SubsystemCode) -> tuple[Pauli, ...]:
    """Return the Z-type half of the record's declared logical operators."""

    return code.logical_operators[: code.num_logical_qubits]


def _x_logicals(code: SubsystemCode) -> tuple[Pauli, ...]:
    """Return the X-type half of the record's declared logical operators."""

    return code.logical_operators[code.num_logical_qubits :]


def test_the_declared_parameters_are_the_ones_the_matrices_derive() -> None:
    tesseract = tesseract_code()

    assert tesseract.num_data_qubits == 16
    assert tesseract.num_stabilizer_generators == 10
    assert tesseract.num_gauge_qubits == 2
    assert tesseract.num_logical_qubits == 4
    assert (
        tesseract.num_logical_qubits
        == tesseract.num_data_qubits
        - tesseract.num_stabilizer_generators
        - tesseract.num_gauge_qubits
    )
    assert tesseract.distance == tesseract.x_distance == tesseract.z_distance == 4
    assert tesseract.data_wires == tuple(range(16))
    assert len(tesseract.stabilizer_generators) == 10
    assert len(tesseract.gauge_generators) == 4
    assert len(tesseract.logical_operators) == 8


def test_the_check_rank_difference_is_not_the_protected_count() -> None:
    tesseract = tesseract_code()
    z_checks = tuple(
        _mask(_supports(check)) for check in tesseract.stabilizer_generators[:5]
    )
    x_checks = tuple(
        _mask(_supports(check)) for check in tesseract.stabilizer_generators[5:]
    )

    assert rank(list(z_checks)) == 5
    assert rank(list(x_checks)) == 5
    # The CSS reading of the same check rows leaves six classes, and the record's
    # four is two fewer because the gauge claims the other two.
    assert tesseract.num_data_qubits - 5 - 5 == 6
    assert tesseract.num_logical_qubits == 6 - tesseract.num_gauge_qubits
    assert tesseract.num_logical_qubits != 6


def test_every_check_is_weight_eight_and_every_gauge_generator_is_weight_four() -> None:
    tesseract = tesseract_code()
    checks = tesseract.stabilizer_generators
    z_gauges = tesseract.gauge_generators[:2]
    x_gauges = tesseract.gauge_generators[2:]

    assert {frozenset(_supports(check)) for check in checks[:5]} == {
        frozenset(support) for support in _CHECKS
    }
    assert {frozenset(_supports(check)) for check in checks[5:]} == {
        frozenset(support) for support in _CHECKS
    }
    assert [check.weight for check in checks] == [8] * 10
    assert [frozenset(_supports(gauge)) for gauge in z_gauges] == [
        frozenset(support) for support in _Z_GAUGES
    ]
    assert [frozenset(_supports(gauge)) for gauge in x_gauges] == [
        frozenset(support) for support in _X_GAUGES
    ]
    assert [gauge.weight for gauge in tesseract.gauge_generators] == [4, 4, 4, 4]
    assert [logical.weight for logical in tesseract.logical_operators] == [4] * 8


def test_the_checks_are_the_center_of_the_gauge_group() -> None:
    tesseract = tesseract_code()

    for check in tesseract.stabilizer_generators:
        for gauge in tesseract.gauge_generators:
            assert check.commutes_with(gauge)


def test_the_gauge_generators_pair_up_nondegenerately() -> None:
    tesseract = tesseract_code()
    z_gauges = tesseract.gauge_generators[:2]
    x_gauges = tesseract.gauge_generators[2:]
    z_checks = tesseract.stabilizer_generators[:5]
    x_checks = tesseract.stabilizer_generators[5:]

    pairing = [
        [0 if z_gauge.commutes_with(x_gauge) else 1 for x_gauge in x_gauges]
        for z_gauge in z_gauges
    ]
    assert pairing == [[1, 0], [0, 1]]
    assert (
        rank([sum(bit << column for column, bit in enumerate(row)) for row in pairing])
        == tesseract.num_gauge_qubits
    )
    for gauge in z_gauges:
        assert not in_span(
            _mask(_supports(gauge)), [_mask(_supports(c)) for c in z_checks]
        )
    for gauge in x_gauges:
        assert not in_span(
            _mask(_supports(gauge)), [_mask(_supports(c)) for c in x_checks]
        )


def test_the_declared_logicals_are_dressed_operators() -> None:
    tesseract = tesseract_code()
    z_checks = tesseract.stabilizer_generators[:5]
    x_checks = tesseract.stabilizer_generators[5:]
    z_gauges = tesseract.gauge_generators[:2]
    x_gauges = tesseract.gauge_generators[2:]
    z_span = [_mask(_supports(operator)) for operator in z_checks + z_gauges]
    x_span = [_mask(_supports(operator)) for operator in x_checks + x_gauges]

    for logical in _z_logicals(tesseract):
        assert all(logical.commutes_with(check) for check in x_checks)
        assert not in_span(_mask(_supports(logical)), z_span)
    for logical in _x_logicals(tesseract):
        assert all(logical.commutes_with(check) for check in z_checks)
        assert not in_span(_mask(_supports(logical)), x_span)

    # The two families pair up non-degenerately, which is what makes them a
    # logical basis rather than two lists of operators with the right weights.
    pairing = [
        [
            0 if z_logical.commutes_with(x_logical) else 1
            for x_logical in _x_logicals(tesseract)
        ]
        for z_logical in _z_logicals(tesseract)
    ]
    assert pairing == [[int(row == column) for column in range(4)] for row in range(4)]


def test_the_distance_is_four_under_both_readings() -> None:
    tesseract = tesseract_code()
    z_checks = tesseract.stabilizer_generators[:5]
    x_checks = tesseract.stabilizer_generators[5:]
    z_gauges = tesseract.gauge_generators[:2]
    x_gauges = tesseract.gauge_generators[2:]
    z_span = [_mask(_supports(operator)) for operator in z_checks + z_gauges]
    x_span = [_mask(_supports(operator)) for operator in x_checks + x_gauges]
    pivots = reduce_rows(z_span)

    def least_weight(commuting_rows: list[int]) -> int | None:
        """Return the least weight of a Z-type operator outside the gauge span."""

        for weight in range(1, tesseract.num_data_qubits + 1):
            for positions in itertools.combinations(
                range(tesseract.num_data_qubits), weight
            ):
                word = _mask(positions)
                if any(bin(word & row).count("1") % 2 for row in commuting_rows):
                    continue
                if reduce_vector(word, pivots):
                    return weight
        return None

    assert tesseract.distance == 4
    # The record's reading: commute with the X-type checks and lie outside the
    # Z-type checks and gauge. The stricter reading of the same matrices, which
    # also asks the operator to commute with the X-type gauge generators, gives
    # the same four, so the declared distance is not an artifact of the reading.
    assert least_weight([_mask(_supports(check)) for check in x_checks]) == 4
    assert (
        least_weight([_mask(_supports(operator)) for operator in x_checks + x_gauges])
        == 4
    )
    assert all(logical.weight == 4 for logical in tesseract.logical_operators)
    assert len(x_span) == 7

    with pytest.raises(
        ValueError, match="no X-type logical operator of weight at most 3"
    ):
        SubsystemCode(
            hz=tesseract.hz,
            hx=tesseract.hx,
            lz=tesseract.lz,
            lx=tesseract.lx,
            gz=tesseract.gz,
            gx=tesseract.gx,
            distance_search_weight=3,
        )


def test_the_distance_is_dressed_and_not_the_check_only_reading() -> None:
    # The tesseract's four is the same number under both readings, so on that
    # family alone "the distance is dressed" is a claim the record would report
    # correctly even if it forgot the gauge. The 3x3 Bacon-Shor is the smallest
    # family where the two readings part: reading it up to the checks alone finds
    # a weight-two operator, and reading it up to the checks *and* the gauge finds
    # only three, which is the family's dressed distance.
    baconshor = SubsystemCode(
        hz=tuple(_row(row) for row in _BACONSHOR_HZ),
        hx=tuple(_row(row) for row in _BACONSHOR_HX),
        lz=tuple(_row(row) for row in _BACONSHOR_LZ),
        lx=tuple(_row(row) for row in _BACONSHOR_LX),
        gz=tuple(_row(row) for row in _BACONSHOR_GZ),
        gx=tuple(_row(row) for row in _BACONSHOR_GX),
        distance_search_weight=3,
    )

    assert baconshor.num_data_qubits == 9
    assert baconshor.num_stabilizer_generators == 4
    assert baconshor.num_gauge_qubits == 4
    assert baconshor.num_logical_qubits == 1

    z_checks = [_mask(row) for row in _BACONSHOR_HZ]
    x_checks = [_mask(row) for row in _BACONSHOR_HX]
    z_gauges = [_mask(row) for row in _BACONSHOR_GZ]
    x_gauges = [_mask(row) for row in _BACONSHOR_GX]

    def least_z_weight(commuting: list[int], excluded: list[int]) -> int | None:
        """Return the least weight of a Z-type operator outside ``excluded``.

        ``commuting`` is the set the operator has to commute with and ``excluded``
        the span it has to lie outside, so the same search states each reading of
        the matrices as a different pair of arguments.
        """

        pivots = reduce_rows(excluded)
        for weight in range(1, baconshor.num_data_qubits + 1):
            for positions in itertools.combinations(
                range(baconshor.num_data_qubits), weight
            ):
                word = _mask(positions)
                if any(bin(word & row).count("1") % 2 for row in commuting):
                    continue
                if reduce_vector(word, pivots):
                    return weight
        return None

    assert baconshor.distance == 3
    assert (baconshor.x_distance, baconshor.z_distance) == (3, 3)
    # The three readings of one matrix set. Commuting with the checks alone and
    # reading up to the checks alone is the bare reading and it finds two. The
    # record's reading reads up to the checks *and* the gauge, and so does the
    # reading that additionally asks the operator to commute with the X-type
    # gauge generators -- both find three, which is where the two disagree and
    # why the gauge is the difference rather than the search's bookkeeping.
    assert least_z_weight(x_checks, z_checks) == 2
    assert least_z_weight(x_checks, z_checks + z_gauges) == 3
    assert least_z_weight(x_checks + x_gauges, z_checks + z_gauges) == 3
    assert all(logical.weight == 3 for logical in baconshor.logical_operators)

    # The same matrices with the bound lowered to the bare reading's answer are
    # refused, so three is a number the search found against the gauge rather
    # than one the declared operators carried into the record. The X-type family
    # is the one the record searches first, so its message is the one that names
    # the bound.
    with pytest.raises(
        ValueError, match="no X-type logical operator of weight at most 2"
    ):
        SubsystemCode(
            hz=tuple(_row(row) for row in _BACONSHOR_HZ),
            hx=tuple(_row(row) for row in _BACONSHOR_HX),
            lz=tuple(_row(row) for row in _BACONSHOR_LZ),
            lx=tuple(_row(row) for row in _BACONSHOR_LX),
            gz=tuple(_row(row) for row in _BACONSHOR_GZ),
            gx=tuple(_row(row) for row in _BACONSHOR_GX),
            distance_search_weight=2,
        )


def test_the_record_refuses_matrices_that_are_not_a_subsystem_code() -> None:
    tesseract = tesseract_code()
    blocks = {
        "hz": tesseract.hz,
        "hx": tesseract.hx,
        "lz": tesseract.lz,
        "lx": tesseract.lx,
        "gz": tesseract.gz,
        "gx": tesseract.gx,
    }
    extra_check = tuple(
        1 if wire in _CHECKS[0] or wire == 8 else 0 for wire in range(16)
    )
    weight_one = tuple(1 if wire == 1 else 0 for wire in range(16))

    with pytest.raises(ValueError, match="hz row 0 and hx row 0 act on an odd number"):
        # One wire, one Z-type check and one X-type check: there is nothing else
        # for the record to refuse on, so this is the commutation condition alone.
        SubsystemCode(hz=((1,),), hx=((1,),))
    with pytest.raises(ValueError, match="gz row 0 anticommutes with hx row 1"):
        SubsystemCode(
            **{**blocks, "gz": (extra_check, tesseract.gz[1])}, distance_search_weight=4
        )
    with pytest.raises(ValueError, match="gz declares 1 gauge generator"):
        SubsystemCode(**{**blocks, "gz": tesseract.gz[:1]}, distance_search_weight=4)
    with pytest.raises(ValueError, match="gz is not independent of the checks"):
        SubsystemCode(
            **{**blocks, "gz": (tesseract.hz[0], tesseract.hz[4])},
            distance_search_weight=4,
        )
    with pytest.raises(ValueError, match="pair up degenerately"):
        # One gauge pair whose members commute is not a pair: the Z-type
        # generator is a protected operator the caller called gauge.
        SubsystemCode(
            **{**blocks, "gz": (tesseract.lz[0],), "gx": (tesseract.lx[1],)},
            distance_search_weight=4,
        )
    with pytest.raises(ValueError, match="lz declares 2 logical operator"):
        SubsystemCode(**{**blocks, "lz": tesseract.lz[:2]}, distance_search_weight=4)
    with pytest.raises(
        ValueError, match="lz row 1 is a combination of the hz and gz rows"
    ):
        SubsystemCode(
            **{
                **blocks,
                "lz": (tesseract.lz[0], tesseract.hz[0]) + tesseract.lz[2:],
            },
            distance_search_weight=4,
        )
    with pytest.raises(
        ValueError, match="lz row 0 is a combination of the hz and gz rows"
    ):
        SubsystemCode(
            **{**blocks, "lz": (tesseract.gz[0],) + tesseract.lz[1:]},
            distance_search_weight=4,
        )
    with pytest.raises(ValueError, match="lx row 1 anticommutes with hz row 0"):
        SubsystemCode(
            **{**blocks, "lx": (tesseract.lx[0], weight_one) + tesseract.lx[2:]},
            distance_search_weight=4,
        )
    with pytest.raises(
        ValueError, match="lx row 0 is a combination of the hx and gx rows"
    ):
        # A gauge generator offered as a logical operator: it commutes with the
        # opposite checks, so only the span it is read modulo catches it.
        SubsystemCode(
            **{**blocks, "lx": (tesseract.gx[1],) + tesseract.lx[1:]},
            distance_search_weight=4,
        )
    with pytest.raises(ValueError, match="pair up degenerately"):
        # One row offered twice as a logical operator: the X-type family spans
        # three classes rather than four, so the two families together generate a
        # smaller logical group than a code with four logical qubits has. Note
        # that re-using the Z-type supports as the X-type family is *not* this
        # case -- those four pair with the Z-type ones as the reverse permutation
        # of the identity, which is a basis.
        SubsystemCode(
            **{
                **blocks,
                "lx": (tesseract.lx[0], tesseract.lx[0]) + tesseract.lx[1:3],
            },
            distance_search_weight=4,
        )
    with pytest.raises(
        ValueError, match="the blocks disagree about how many data qubits"
    ):
        SubsystemCode(
            **{**blocks, "lz": tuple(row[:15] for row in tesseract.lz)},
            distance_search_weight=4,
        )
    with pytest.raises(ValueError, match="no block states a check"):
        SubsystemCode(hz=(), hx=(), lz=(), lx=(), gz=(), gx=())
    with pytest.raises(ValueError, match="hz row 1 acts on no data qubit"):
        SubsystemCode(
            **{**blocks, "hz": (tesseract.hz[0], (0,) * 16) + tesseract.hz[2:]},
            distance_search_weight=4,
        )
    with pytest.raises(TypeError, match="must be a matrix of 0s and 1s"):
        SubsystemCode(**{**blocks, "hz": "not a matrix"})
    with pytest.raises(TypeError, match="must be a sequence of 0s and 1s"):
        SubsystemCode(**{**blocks, "hz": (0, 1)})
    with pytest.raises(TypeError, match="hz row 0 carries True"):
        SubsystemCode(
            **{**blocks, "hz": ((True,) + tesseract.hz[0][1:],) + tesseract.hz[1:]},
            distance_search_weight=4,
        )
    with pytest.raises(TypeError, match="distance search weight must be an integer"):
        SubsystemCode(**blocks, distance_search_weight=4.0)
    with pytest.raises(ValueError, match="distance search weight must be at least one"):
        SubsystemCode(**blocks, distance_search_weight=0)


def test_the_css_reading_of_the_same_matrices_is_a_different_code() -> None:
    tesseract = tesseract_code()

    # The CSS record counts six logical qubits in these check rows, so it reads
    # the four declared logicals as a short family rather than as this code's.
    with pytest.raises(ValueError, match="a code with 6 logical qubit"):
        CssCode(
            hz=tesseract.hz,
            hx=tesseract.hx,
            lz=tesseract.lz,
            lx=tesseract.lx,
            distance_search_weight=4,
        )

    # The subsystem record does not present the members the CSS route reads, so
    # that route refuses it rather than building matrices the gauge is missing
    # from -- a stabilizer code the caller never declared.
    assert not isinstance(tesseract, StabilizerCode)
    with pytest.raises(TypeError, match="code must be a StabilizerCode"):
        css_code_matrices(tesseract)


def test_the_column_swap_is_an_automorphism_of_the_gauge_group() -> None:
    tesseract = tesseract_code()
    swap = tesseract_column_swap()
    generators = [
        (
            (_mask(_supports(operator)), 0)
            if operator.z_wires
            else (0, _mask(_supports(operator)))
        )
        for operator in tesseract.stabilizer_generators + tesseract.gauge_generators
    ]
    assert len(generators) == 14

    group = {(0, 0)}
    for x_part, z_part in generators:
        group |= {(x_image ^ x_part, z_image ^ z_part) for x_image, z_image in group}
    # The generators are independent, so the group has one element per subset of
    # them rather than fewer.
    assert len(group) == 2**14

    images = {
        (_permute(x_part, swap), _permute(z_part, swap)) for x_part, z_part in group
    }
    assert images <= group
    # Conjugation by a wire permutation is injective, so an image contained in a
    # finite group of the same size is the whole group.
    assert images == group


def test_the_column_swap_implements_two_cnots_on_the_declared_logical_order() -> None:
    tesseract = tesseract_code()
    swap = tesseract_column_swap()
    z_logicals = _z_logicals(tesseract)
    x_logicals = _x_logicals(tesseract)
    z_span = [
        _mask(_supports(operator))
        for operator in tesseract.stabilizer_generators[:5]
        + tesseract.gauge_generators[:2]
    ]
    x_span = [
        _mask(_supports(operator))
        for operator in tesseract.stabilizer_generators[5:]
        + tesseract.gauge_generators[2:]
    ]

    def class_of(image: int, logicals: tuple[Pauli, ...], span: list[int]) -> int:
        """Return the logical combination ``image`` is equivalent to, or -1."""

        for bits in range(1 << len(logicals)):
            candidate = image
            for index, logical in enumerate(logicals):
                if bits >> index & 1:
                    candidate ^= _mask(_supports(logical))
            if in_span(candidate, span):
                return bits
        return -1

    measured_z = tuple(
        class_of(_permute(_mask(_supports(logical)), swap), z_logicals, z_span)
        for logical in z_logicals
    )
    measured_x = tuple(
        class_of(_permute(_mask(_supports(logical)), swap), x_logicals, x_span)
        for logical in x_logicals
    )

    # What CNOT(control -> target) does to a logical pair: the control's X
    # operator gains the target's, the target's Z operator gains the control's,
    # and the two operators of the other pair do not move. Nothing else in either
    # family moves, so the measured action below is the whole of the claim. The
    # pairs come from the family rather than from this test, which is what makes
    # the comparison two statements instead of one written twice.
    pairs = tesseract_free_cnot_pairs()
    assert len({index for pair in pairs for index in pair}) == 4
    bases = tuple(1 << index for index in range(4))
    expected_x = tuple(
        bits | sum(1 << target for control, target in pairs if control == index)
        for index, bits in enumerate(bases)
    )
    expected_z = tuple(
        bits | sum(1 << control for control, target in pairs if target == index)
        for index, bits in enumerate(bases)
    )

    assert measured_z == expected_z
    assert measured_x == expected_x
    # A control: the identity permutation would have fixed both families, so the
    # two combinations above are not what every permutation produces.
    assert expected_z != bases and expected_x != bases
    assert measured_z != bases and measured_x != bases
    assert tuple(range(16)) != swap


def test_the_published_permutation_is_the_layout_involution() -> None:
    swap = tesseract_column_swap()

    # The permutation moves a wire to a wire and nowhere else, so its image is the
    # sixteen wires rather than a tuple that happens to hold sixteen numbers.
    assert sorted(swap) == list(range(16))
    # Swapping the first and third column of each four-wire row is an involution,
    # and that is a property the derivation has to have rather than one the
    # published tuple might have by luck.
    assert all(swap[swap[position]] == position for position in range(16))
    # The layout reading: a wire whose lowest coordinate is clear exchanges the
    # column above it, and one whose lowest coordinate is set does not move. The
    # two halves are stated separately so that a derivation that moved every wire
    # cannot pass by matching the second half alone.
    moved = tuple(position for position in range(16) if swap[position] != position)
    assert moved == (0, 2, 4, 6, 8, 10, 12, 14)
    assert all(swap[position] == position ^ 2 for position in moved)

    # The external statement: this is the tuple the framework this package
    # replaces publishes as the family's ``column_swap``. A derivation is only
    # worth calling one if it lands on the published permutation rather than on
    # some other involution of the same layout.
    assert swap == _CUDAQ_COLUMN_SWAP
    assert swap != tuple(range(16))


def test_the_public_namespace_publishes_the_record() -> None:
    import flagquantum.qec as qec

    assert qec.SubsystemCode is SubsystemCode
    assert qec.tesseract_code is tesseract_code
    assert qec.tesseract_column_swap is tesseract_column_swap
    assert qec.tesseract_free_cnot_pairs is tesseract_free_cnot_pairs
    for name in (
        "SubsystemCode",
        "tesseract_code",
        "tesseract_column_swap",
        "tesseract_free_cnot_pairs",
    ):
        assert name in qec.__all__
