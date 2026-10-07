"""Tests for the punctured Reed-Muller family and the convention it states.

The family is the first route in this package written on the Boolean cube rather
than on a lattice or on a group, and the thing it adds that the routes beside it do
not is a *convention*: the two punctured blocks can state the two check families
either way round, and the two readings are the same code with the two Pauli families
labelled the other way. So these tests measure three things rather than one -- that
the parameters are the code the family claims, that the declared convention is the
one the contract names and that exchanging the blocks really does exchange the two
families, and that the least logical operator of the lighter family really has the
weight the record reports.

The third is the one worth the enumeration it costs. A record's distance is computed
by a search that stops at ``distance_search_weight``, so a record could report seven
because seven is the least operator and could equally report seven because six was
never tried. What settles it is an exhaustive search of the same matrices from the
outside, at every weight below seven, which the first test below does rather than
trusting the number that came back.
"""

from __future__ import annotations

from itertools import combinations

import pytest

from flagquantum.qec import (
    CssCode,
    Pauli,
    build_memory_circuit,
    qldpc_code,
    reed_muller_code,
)
from flagquantum.qec.gf2 import in_span, reduce_rows, reduce_vector

pytestmark = pytest.mark.unit

# The two widths this route builds. Four is the smallest cube whose punctured pair
# states a code at all and five is the largest the distance search pays for.
_SMALL = 4
_LARGE = 5

# The weight the lighter family's least logical operator has, and therefore the
# bound the search needs. A smaller bound refuses the record instead of reporting a
# distance, which one of the tests below measures.
_LIGHT = 7

# The heavier family's least weight. It is what the enumeration below stops short
# of, so the two families' searches are not the same search.
_HEAVY = 3


def _bits(rows: list[list[int]]) -> list[int]:
    """Pack each 0/1 row into one integer, column ``i`` becoming bit ``i``.

    Bit ``i`` of a mask is column ``i`` of the row, which is the order the masks in
    :mod:`flagquantum.qec.reed_muller` carry and the order the GF(2) helpers here
    take. The library's own record states its matrices as rows of 0s and 1s, so this
    is the one place the two representations meet.
    """

    return [sum(1 << index for index, bit in enumerate(row) if bit) for row in rows]


def _pack(observable: Pauli, code: CssCode) -> int:
    """Return one observable as a bit mask over the code's own data wires."""

    positions = {wire: index for index, wire in enumerate(code.data_wires)}
    return sum(1 << positions[wire] for wire in observable.support)


def _overlaps(left: int, right: int) -> bool:
    """Whether two bit masks meet on an odd number of places."""

    return bin(left & right).count("1") % 2 == 1


def _least_weight(
    commuting: list[int], stabilizer: list[int], width: int, ceiling: int
) -> int | None:
    """Return the least weight up to ``ceiling`` of a logical operator, or nothing.

    The reading is the one the record's own search uses, restated outside the
    record so that the record's answer is checked rather than repeated: a subset of
    the data qubits commutes with every row of the opposite family, and it is a
    logical operator rather than a stabilizer when it is outside the span of its own
    family's rows.
    """

    pivots = reduce_rows(list(stabilizer))
    for weight in range(1, ceiling + 1):
        for positions in combinations(range(width), weight):
            candidate = 0
            for position in positions:
                candidate |= 1 << position
            if any(_overlaps(candidate, row) for row in commuting):
                continue
            if reduce_vector(candidate, pivots):
                return weight
    return None


def _code(m: int = _SMALL) -> CssCode:
    """Return the instance most tests measure, which is the fifteen-qubit one."""

    return reed_muller_code(m)


def _cube_block(degree: int, variables: int) -> list[int]:
    """Return a punctured cube block's monomials, restated outside the record.

    The rule is written a second time here on purpose. A monomial of degree at most
    ``degree`` is the set of cube points at which every variable it names is one, so
    it is a bit mask over those points, and the punctured block is the block on the
    points below the one naming every variable. Reading that here rather than
    importing the module's own helpers is what makes a damaged monomial rule
    *noticed* -- mutating that rule was measured to redden this suite, and a test
    that called the module's helper would have moved with the damage and stayed
    green -- and it is what lets the block be stated without also stating the
    helper's answer, which the check families below are compared against.
    """

    return [
        sum(
            1 << point
            for point in range((1 << variables) - 1)
            if all(point >> bit & 1 for bit in support)
        )
        for size in range(degree + 1)
        for support in combinations(range(variables), size)
    ]


def test_the_smallest_instance_is_the_fifteen_qubit_code() -> None:
    """Length, ancillas, logical qubits and both family distances, read off.

    The fifteen-qubit instance is the one the family's name usually means: the
    punctured Reed-Muller code whose ``k`` is one and whose distance is three. The
    two family distances are not equal here, which is the whole reason the record
    states them separately rather than as one number.
    """

    code = _code()

    assert code.num_data_qubits == 15
    assert code.num_ancilla_qubits == 14
    assert len(code.logical_observables) // 2 == 1
    assert code.x_distance == _LIGHT
    assert code.z_distance == _HEAVY
    assert code.distance == _HEAVY


def test_the_larger_instance_carries_eleven_logical_qubits() -> None:
    """The width this route reaches furthest is the one worth stating it on.

    Thirty-one data qubits with eleven logical qubits each three errors apart is
    what the family is for: a lattice patch of thirty-one qubits carries one logical
    qubit at the same distance. The two family distances stay seven and three, so
    the growth is in ``k`` rather than in the distance, and the search bound the
    record needs does not move with the width.
    """

    code = _code(_LARGE)

    assert code.num_data_qubits == 31
    assert len(code.logical_observables) // 2 == 11
    assert (code.x_distance, code.z_distance) == (_LIGHT, _HEAVY)
    assert code.distance == _HEAVY


@pytest.mark.parametrize("m", [_SMALL, _LARGE])
def test_the_lighter_family_has_nothing_below_seven(m: int) -> None:
    """The reported seven is the least weight, measured rather than assumed.

    The record's own search stops at its bound, so it would report seven for a code
    whose true least X-type operator had weight seven and for one whose operator of
    weight four the bound never reached. The enumeration here runs outside the
    record over the same matrices and finds nothing at any weight below seven, which
    is the statement the record's answer needs and cannot make about itself.
    """

    code = _code(m)
    width = code.num_data_qubits
    # The X-type family: it commutes with the Z-type checks and lies outside the
    # X-type stabilizer span, in the record's own naming.
    z_checks = _bits([list(row) for row in code.hz])
    x_checks = _bits([list(row) for row in code.hx])

    assert _least_weight(z_checks, x_checks, width, _LIGHT - 1) is None
    assert _least_weight(z_checks, x_checks, width, _LIGHT) == _LIGHT
    # The mirror statement for the other family, which is the heavier one.
    assert _least_weight(x_checks, z_checks, width, _HEAVY - 1) is None
    assert _least_weight(x_checks, z_checks, width, _HEAVY) == _HEAVY


@pytest.mark.parametrize("m", [_SMALL, _LARGE])
def test_the_checks_are_the_duals_of_the_two_punctured_blocks(m: int) -> None:
    """Each check family is the dual of its own punctured block.

    A row of the X-type checks has to evaluate to zero on every punctured monomial
    of degree at most ``m - 2``, because that is what "the dual of the punctured
    block" means, and the Z-type rows have to do the same for degree at most
    ``m - 3``. Measuring it here is what separates this rule from a pair of matrices
    that merely commute. The two blocks are two degrees apart and therefore nested,
    so the duals are nested the other way: every X-type check is orthogonal to both
    blocks, and the Z-type checks are the family that reaches the finer block --
    which is the measured asymmetry that says the two blocks are not the same block.
    """

    code = _code(m)
    higher = _cube_block(m - 2, m)
    lower = _cube_block(m - 3, m)

    x_checks = _bits([list(entry) for entry in code.hx])
    z_checks = _bits([list(entry) for entry in code.hz])

    for row in x_checks:
        assert all(not _overlaps(row, monomial) for monomial in higher)
        assert all(not _overlaps(row, monomial) for monomial in lower)
    for row in z_checks:
        assert all(not _overlaps(row, monomial) for monomial in lower)
    # The Z-type family is the wider one, so it cannot be inside the narrower dual:
    # every Z-type row meets at least one monomial of the higher-degree block.
    assert all(any(_overlaps(row, monomial) for monomial in higher) for row in z_checks)
    assert len(z_checks) > len(x_checks)


@pytest.mark.parametrize("m", [_SMALL, _LARGE])
def test_every_declared_operator_is_a_logical_operator(m: int) -> None:
    """Each observable commutes with the opposite family and leaves its own span.

    The record derives these operators rather than taking them from the caller, so
    the two invariants are the only thing standing between a derived basis and a
    list of vectors: an operator that anticommuted with a check would leave the code
    space, and one inside its own family's span would be a stabilizer whose outcome
    is already fixed.
    """

    code = _code(m)
    checks = [check.stabilizer for check in code.checks]
    observables = list(code.logical_observables)

    for observable in observables:
        for check in checks:
            opposite = bool(observable.z_wires) != bool(check.z_wires)
            if opposite:
                assert observable.commutes_with(check)
        # Its own family's rows, as masks over the code's own wire order.
        positions = {wire: index for index, wire in enumerate(code.data_wires)}
        mask = sum(1 << positions[wire] for wire in observable.support)
        if observable.z_wires:
            family = _bits([list(row) for row in code.hz])
        else:
            family = _bits([list(row) for row in code.hx])
        assert not in_span(mask, reduce_rows(family))


@pytest.mark.parametrize("m", [_SMALL, _LARGE])
def test_the_two_families_pair_as_the_identity(m: int) -> None:
    """Each Z-type operator anticommutes with exactly one X-type operator.

    A complement basis of each null space is an arbitrary basis, so nothing forces
    the pairing between the two families to be the identity; what forces it is the
    symplectic elimination this route runs after deriving them, which it has in
    common with the generic route because it calls the same two helpers. The weaker
    property the record certifies is that the pairing has full rank, so this test is
    what separates the two: a permuted pairing passes the record and fails here, and
    a route that skipped the elimination entirely was measured to pass every other
    test in this file. The order is what a caller reads through, since ``lz[i]`` and
    ``lx[i]`` are then partners, and the larger instance is the one that makes the
    statement non-trivial: eleven logical qubits leave a pairing that could be
    permuted in many ways.
    """

    code = _code(m)
    z_observables = [
        observable for observable in code.logical_observables if observable.z_wires
    ]
    x_observables = [
        observable for observable in code.logical_observables if observable.x_wires
    ]
    pairing = [
        [0 if z.commutes_with(x) else 1 for x in x_observables] for z in z_observables
    ]
    size = len(z_observables)

    assert size == len(x_observables)
    assert pairing == [
        [1 if row == column else 0 for column in range(size)] for row in range(size)
    ]


def test_exchanging_the_two_blocks_exchanges_the_two_families() -> None:
    """The convention names which family is the light one, and the mirror is reachable.

    The two readings of the family differ by which punctured block states which
    check family. Reading the record with the blocks exchanged has to exchange the
    two distances and leave everything else where it was, which is what makes the
    convention in the module docstring a convention rather than a claim about the
    code: one record is the other with the two Pauli families renamed.
    """

    code = _code()
    mirror = qldpc_code(hz=code.hx, hx=code.hz, distance_search_weight=_LIGHT)

    assert (code.x_distance, code.z_distance) == (_LIGHT, _HEAVY)
    assert (mirror.x_distance, mirror.z_distance) == (_HEAVY, _LIGHT)
    assert mirror.num_data_qubits == code.num_data_qubits
    assert mirror.distance == code.distance
    assert len(mirror.logical_observables) == len(code.logical_observables)


def test_the_direct_route_states_the_same_code() -> None:
    """The family is a rule over the same record the matrix route returns.

    This family derives its logical operators through the same two helpers the
    direct route uses, so handing the family's own matrices to that route has to
    return the same parameters. It is what says the family adds a rule rather than a
    second record type, and the control is that the direct route is given no
    knowledge of the family at all.
    """

    code = _code()
    direct = qldpc_code(hz=code.hz, hx=code.hx, distance_search_weight=_LIGHT)

    assert direct.num_data_qubits == code.num_data_qubits
    assert direct.num_ancilla_qubits == code.num_ancilla_qubits
    assert (direct.x_distance, direct.z_distance) == (code.x_distance, code.z_distance)
    assert len(direct.logical_observables) == len(code.logical_observables)


@pytest.mark.parametrize("m", [_SMALL, _LARGE])
def test_the_record_runs_a_memory_experiment(m: int) -> None:
    """The record is a stabilizer code rather than a pair of matrices in a box.

    A code that no experiment can be built from is a record of matrices, and the
    experiment is the shortest statement that the checks, the wire layout and the
    logical operators are the ones a caller would measure. The larger instance is
    the one that matters here: eleven logical qubits means eleven observables, so a
    route that carried only the first one would pass on the small instance and lose
    ten operators on the large one.
    """

    code = _code(m)
    circuit = build_memory_circuit(code, rounds=2)

    assert len(circuit.detectors) > 0
    assert len(circuit.observables.observables) == len(code.logical_observables) // 2
    assert all(entry.pauli.z_wires for entry in circuit.observables.observables)


@pytest.mark.parametrize("m", [0, 1, 2, 3, -1])
def test_a_cube_too_small_to_state_a_code_is_refused(m: int) -> None:
    """Below four the punctured pair states no stabilizer group at all.

    The refusal is not a matter of this route's cost: the two blocks of a cube that
    small are one and zero degrees apart, so their duals carry more rows than the
    cube has points and rows that many cannot all commute. The message says so
    rather than reporting a distance search as the reason.
    """

    with pytest.raises(ValueError, match="states no code"):
        reed_muller_code(m)


@pytest.mark.parametrize("m", [6, 7, 12])
def test_a_cube_wider_than_the_search_pays_for_is_refused(m: int) -> None:
    """Above five the code exists and this route does not build it.

    The rule is stated for every width at or above four, so the refusal has to be
    about the route rather than about the family, and it has to name the search as
    the reason. A caller who wants a wider instance still has the matrices and the
    direct route, which is what the message says by omission: nothing here claims
    the code does not exist.
    """

    with pytest.raises(ValueError, match="this route does not build"):
        reed_muller_code(m)


@pytest.mark.parametrize("m", [True, False, 4.0, "4", None])
def test_a_width_that_is_not_an_integer_is_refused(m: object) -> None:
    """A width is a count, and a count that is not a whole number is a caller bug."""

    with pytest.raises(TypeError, match="must be an integer"):
        reed_muller_code(m)  # type: ignore[arg-type]


def test_a_bound_that_cannot_reach_the_lighter_family_is_refused() -> None:
    """The default bound is not decoration: a smaller one refuses the record.

    The heavier family's least operator has weight three, so a bound of three is
    reached by that family and not by the other one, and the record refuses rather
    than reporting the smaller of two numbers one of which was never searched for.
    The test asserts the direction as well, because a bound of seven succeeding is
    what says the refusal was about the bound rather than about the code.
    """

    with pytest.raises(
        ValueError, match="no X-type logical operator of weight at most"
    ):
        reed_muller_code(_SMALL, distance_search_weight=_HEAVY)

    assert reed_muller_code(_SMALL, distance_search_weight=_LIGHT).x_distance == _LIGHT


def test_the_distance_search_costs_more_as_the_bound_grows() -> None:
    """The search is the route's whole cost, and the bound is where it is paid.

    A record built with a bound of seven and one built with a bound of nine report
    the same distances, because the least operator is at seven and the search stops
    when it finds one. The parameter is therefore a ceiling on the search rather
    than an input to the answer, which is what lets the refusal above be about the
    bound and not about the code.
    """

    code = reed_muller_code(_SMALL, distance_search_weight=_LIGHT + 2)

    assert (code.x_distance, code.z_distance) == (_LIGHT, _HEAVY)


def test_the_package_exports_the_constructor() -> None:
    """The family is reachable from the package rather than only from its module.

    A name a module defines but ``__all__`` does not carry is a name the package
    does not promise, and no other test here can see the difference because they all
    import the constructor from the package. The assertion is made on ``__all__``
    itself, which is the same statement the sibling family tests make.
    """

    import flagquantum.qec as qec

    assert hasattr(qec, "reed_muller_code")
    assert "reed_muller_code" in qec.__all__
    assert qec.reed_muller_code is reed_muller_code
