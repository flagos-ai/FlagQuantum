"""Tests for the route that derives a code's logical operators from its checks.

The route is the general case of what the derived families do for themselves: it
takes the two parity-check matrices of a Calderbank-Shor-Steane code and returns
the record those matrices define, with the logical operators computed rather than
handed in. What these tests hold is therefore not the shape of a family but the
two halves of that computation -- that the operators are logical operators of the
matrices at all, and that they are the declared ones rather than merely some
basis.

The oracle is a code whose logical operators are stated somewhere else in this
package, which is why every family test starts from a declared record and reads
its matrices back out. Two properties are deliberately separated. The parameters
``(n, ancilla, k, d)`` are the weak property: a wrong basis of the right quotient
reproduces them. The strong property is that every derived operator differs from
the declared one by a stabilizer, so a representation that drifts inside its own
class is caught rather than absorbed.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import pytest

from flagquantum.qec import (
    CssCode,
    Pauli,
    SteaneCode,
    bivariate_bicycle_code,
    build_memory_circuit,
    certify_logical_product,
    css_code_matrices,
    qldpc_code,
    toric_code,
)
from flagquantum.qec.gf2 import in_span

pytestmark = pytest.mark.unit

# The Steane code's check matrix, written here rather than read from the record
# because a route that takes a caller's matrix must be given one.
_STEANE_CHECKS = [
    [0, 0, 0, 1, 1, 1, 1],
    [0, 1, 1, 0, 0, 1, 1],
    [1, 0, 1, 0, 1, 0, 1],
]

# The bivariate-bicycle instance this file uses where a multi-logical code is
# wanted: three-term polynomials over Z_3 * Z_3, which is [[18, 4, 4]].
_BICYCLE = ((0, 0), (0, 1), (1, 0)), ((0, 0), (0, 1), (2, 1))


def _pack(row: Sequence[int]) -> int:
    """Pack a 0-or-1 row into a mask whose bit ``i`` is the row's ``i``-th entry."""

    return sum(bit << index for index, bit in enumerate(row))


def _support(observable: Pauli, code: CssCode) -> int:
    """Pack one observable's support over the code's data wires into a mask."""

    positions = {wire: index for index, wire in enumerate(code.data_wires)}
    return sum(1 << positions[wire] for wire in observable.support)


def _families(code: CssCode) -> tuple[list[Pauli], list[Pauli]]:
    """Return one record's Z-type and X-type logical operators, in its own order."""

    return (
        [observable for observable in code.logical_observables if observable.z_wires],
        [observable for observable in code.logical_observables if observable.x_wires],
    )


def _torus(size: int) -> tuple[CssCode, CssCode]:
    """Return the declared torus and the record the route derives from its checks."""

    declared = toric_code(size)
    blocks = css_code_matrices(declared)
    route = qldpc_code(
        hz=blocks.hz.tolist(),
        hx=blocks.hx.tolist(),
        distance_search_weight=size,
    )
    return declared, route


def _bicycle() -> tuple[CssCode, CssCode]:
    """Return the declared bicycle family and the record the route derives."""

    declared = bivariate_bicycle_code(3, 3, *_BICYCLE, distance_search_weight=4)
    route = qldpc_code(hz=declared.hz, hx=declared.hx, distance_search_weight=4)
    return declared, route


#: One oracle per family whose logical operators are declared elsewhere, each
#: with the distance bound its own search needs. The Steane code is the only one
#: here whose two check families are the same matrix, and the bicycle is the only
#: one whose declared operators come from a rule rather than from a table, so the
#: four cover both halves of that split.
_ORACLES: dict[str, Callable[[], tuple[CssCode, CssCode]]] = {
    "steane": lambda: (
        SteaneCode(),
        qldpc_code(hz=_STEANE_CHECKS, hx=_STEANE_CHECKS),
    ),
    "torus-3": lambda: _torus(3),
    "torus-5": lambda: _torus(5),
    "bicycle": _bicycle,
}


def test_the_route_reproduces_a_declared_family_from_its_checks() -> None:
    """Three checks a side state the Steane code, so the route must find it.

    The declared record is the oracle, and it is reached with only the checks as
    input: no logical operator is passed in, and the two numbers that could be
    copied from the declared record -- the count and the distance -- are the two
    the test reads back.
    """

    declared = SteaneCode()
    route = qldpc_code(hz=_STEANE_CHECKS, hx=_STEANE_CHECKS)

    assert route.num_data_qubits == declared.num_data_qubits == 7
    assert route.num_ancilla_qubits == 6
    assert route.distance == declared.distance == 3
    z_logicals, x_logicals = _families(route)
    assert len(z_logicals) == len(x_logicals) == 1, (
        "three checks a side leave the Steane code one logical qubit, and the "
        f"derivation reported {len(z_logicals)} Z-type and {len(x_logicals)} "
        "X-type operators"
    )


@pytest.mark.parametrize("family", sorted(_ORACLES))
def test_the_derived_operators_are_the_declared_ones_modulo_the_checks(
    family: str,
) -> None:
    """Every derived operator is its declared partner up to a stabilizer.

    This is the property the parameters cannot state. A complement of the
    stabilizer span is not unique, so a route that picked a different
    representative of the same logical class would report the same ``k`` and the
    same distance and still be wrong about which operator it handed a caller. The
    difference between the two representatives is required to lie in the span of
    the checks of the family's own type, which is exactly the claim that they are
    the same class.
    """

    declared, route = _ORACLES[family]()
    assert route.data_wires == declared.data_wires
    declared_blocks = css_code_matrices(declared)
    z_checks = [_pack(row) for row in declared_blocks.hz.tolist()]
    x_checks = [_pack(row) for row in declared_blocks.hx.tolist()]
    declared_z, declared_x = _families(declared)
    route_z, route_x = _families(route)

    assert len(route_z) == len(declared_z), (
        f"the {family} route derived {len(route_z)} Z-type operators where the "
        f"declared record states {len(declared_z)}"
    )
    assert len(route_x) == len(declared_x), (
        f"the {family} route derived {len(route_x)} X-type operators where the "
        f"declared record states {len(declared_x)}"
    )
    for derived, stated in zip(route_z, declared_z, strict=True):
        assert in_span(_support(derived, route) ^ _support(stated, route), z_checks), (
            f"the {family} route's Z-type operator {derived.to_text()} differs from "
            f"the declared {stated.to_text()} by something outside the Z-type checks, "
            "so the two are different logical classes and not two representatives of "
            "one"
        )
    for derived, stated in zip(route_x, declared_x, strict=True):
        assert in_span(_support(derived, route) ^ _support(stated, route), x_checks), (
            f"the {family} route's X-type operator {derived.to_text()} differs from "
            f"the declared {stated.to_text()} by something outside the X-type checks, "
            "so the two are different logical classes and not two representatives of "
            "one"
        )


def test_the_logical_count_is_the_rank_and_not_the_check_row_count() -> None:
    """A repeated check states one constraint twice, so it adds no logical qubit.

    Nothing forbids a caller from writing a check twice, and the two numbers that
    could be read off the input disagree about what that means: counting rows
    would report one fewer logical qubit for the repeated matrix. The count is
    the rank, so the two records agree on ``k`` and on the distance while the
    ancilla count -- which does follow the rows, because each row is a check that
    gets measured -- moves by one.
    """

    plain = qldpc_code(hz=[[1, 1, 1, 1]], hx=[], distance_search_weight=4)
    repeated = qldpc_code(
        hz=[[1, 1, 1, 1], [1, 1, 1, 1]], hx=[], distance_search_weight=4
    )

    assert plain.num_ancilla_qubits == 1
    assert repeated.num_ancilla_qubits == 2
    assert len(_families(plain)[0]) == len(_families(repeated)[0]) == 3, (
        "one check on four data qubits leaves three logical qubits whether it is "
        "written once or twice, because the count is the rank of the check matrix "
        "and not the number of rows it was written on"
    )
    assert plain.distance == repeated.distance == 1


#: The oracles that leave more than one logical qubit, which are the only ones a
#: pairing between the two families can be off-diagonal for.
_PAIRED: tuple[str, ...] = ("torus-3", "torus-5", "bicycle")


@pytest.mark.parametrize("family", _PAIRED)
def test_the_two_families_pair_as_the_identity(family: str) -> None:
    """Each Z-type operator anticommutes with exactly one X-type operator.

    A complement basis is arbitrary, so nothing forces the pairing between the two
    families to be the identity; the elimination the route runs after deriving them
    is what does. The record certifies the weaker property -- that the pairing has
    full rank, because a singular pairing is not a code -- so a pairing that is a
    permutation of the identity passes that and fails here.

    The bicycle is the case that observes the elimination rather than merely
    containing it. The basis its derivation computes *before* the elimination pairs
    as ``[[0, 0, 1, 0], [0, 0, 1, 1], [1, 1, 0, 1], [0, 1, 1, 0]]``, so a route that
    swapped columns until the diagonal was clear and stopped there would leave three
    of those ones off the diagonal. The two tori are the contrast: their derived
    bases already pair diagonally, so running the elimination over them holds the
    other half of the statement -- that it does not disturb a pairing that was
    already clean.
    """

    _, route = _ORACLES[family]()
    z_logicals, x_logicals = _families(route)
    size = len(z_logicals)
    assert len(x_logicals) == size > 1
    pairing = [[0 if z.commutes_with(x) else 1 for x in x_logicals] for z in z_logicals]

    assert pairing == [
        [1 if row == column else 0 for column in range(size)] for row in range(size)
    ], (
        f"the derived {family} families must pair as the identity; they pair as "
        f"{pairing} instead, so the elimination did not finish putting them into the "
        "symplectic basis"
    )


def test_the_distance_is_searched_rather_than_read_off_the_derived_operators() -> None:
    """A derived representative can be far heavier than the code's distance.

    The route reports the least weight over the whole code, so the weights of the
    operators it happened to derive are not the distance. The distance-three
    torus is the case: one of its two derived Z-type operators spans seven data
    qubits and the record still reports three, because the search runs over the
    code and not over the two rows the derivation produced.
    """

    _, route = _torus(3)
    z_logicals, x_logicals = _families(route)

    assert route.distance == 3
    assert max(observable.weight for observable in z_logicals) == 7
    assert max(observable.weight for observable in x_logicals) == 5


def test_one_check_family_still_states_both_logical_families() -> None:
    """A code with only Z-type checks derives an X-type family as well.

    The count is the rank of both matrices together, so a family of checks that
    is empty leaves the whole space for the other family's logical operators
    rather than leaving that family unstated. The declared repetition record is
    the contrast: it writes its single logical operator as a Z-type one and
    leaves the X-type block empty, while the route states one of each and the
    memory circuit declares the Z-type one.
    """

    route = qldpc_code(hz=[[1, 1, 0], [0, 1, 1]], hx=[])
    z_logicals, x_logicals = _families(route)

    assert route.num_data_qubits == 3
    assert route.num_ancilla_qubits == 2
    assert len(z_logicals) == len(x_logicals) == 1
    assert z_logicals[0].z_wires == (0,)
    assert x_logicals[0].x_wires == (0, 1, 2)

    memory = build_memory_circuit(route, rounds=2)
    assert len(memory.observables) == 1
    assert certify_logical_product(route, z_logicals[0]) == z_logicals[0]


def test_the_route_and_the_family_state_the_same_code() -> None:
    """The family and the direct route reach one record, not two.

    The bicycle family derives its own logical operators from a pair of
    polynomials and the route derives them from the matrices those polynomials
    produce, so the two share the derivation rather than each carrying one. The
    test feeds the family's own matrices back through the route: a second copy of
    the elimination that drifted would show up as a differing count or distance
    here rather than as two green suites.
    """

    declared, route = _bicycle()

    assert route.hz == declared.hz
    assert route.hx == declared.hx
    assert route.num_data_qubits == declared.num_data_qubits == 18
    assert route.num_ancilla_qubits == declared.num_ancilla_qubits == 18
    assert len(_families(route)[0]) == len(_families(declared)[0]) == 4
    assert route.distance == declared.distance == 4


def test_a_pair_that_states_no_data_qubit_is_refused() -> None:
    """Two empty blocks state no width, and a width cannot be derived from them.

    The derivation runs at a stated number of columns, so this is the route's own
    refusal rather than the record's: it is raised before any elimination, and it
    names the two blocks rather than the width it failed to find. An empty block
    beside a stated one is the different case and is accepted, which the test
    holds on the same call.
    """

    with pytest.raises(ValueError, match="neither hz nor hx states a data qubit"):
        qldpc_code(hz=[], hx=[])

    accepted = qldpc_code(hz=[[1, 1]], hx=[])
    assert accepted.num_data_qubits == 2


def test_a_pair_whose_families_do_not_commute_is_refused() -> None:
    """Two checks that anticommute generate no stabilizer group.

    The derivation on its own would answer this input -- it returns one operator
    per family for it -- so what refuses it is the record the answer is handed to,
    and the refusal names the two rows that clash rather than the width or the
    count. A route that stopped at its own answer would return a code here.
    """

    with pytest.raises(ValueError, match="hz row 0 and hx row 0 act on an odd number"):
        qldpc_code(hz=[[1, 0]], hx=[[1, 0]])


def test_a_pair_that_leaves_no_logical_qubit_is_refused() -> None:
    """Checks that span every data qubit describe a stabilizer state.

    The refusal is the record's and not the route's, and it is the count that
    triggers it: the derivation returns no operator at all here, because the
    complement of the stabilizer span inside the null space is empty. The message
    states the rank it measured rather than the rows it was given, which is the
    same distinction the count test measures from the other side.
    """

    with pytest.raises(ValueError, match="no logical qubit is left"):
        qldpc_code(hz=[[1, 1], [1, 1]], hx=[[1, 1]])


def test_a_row_that_acts_on_no_data_qubit_is_refused() -> None:
    """A zero row states an operator over nothing rather than a check.

    The matrix is a valid matrix of 0s and 1s, so the shape checks pass and the
    record is what refuses it. This is the case a caller is most likely to reach
    by building rows programmatically and leaving one empty.
    """

    with pytest.raises(ValueError, match="hz row 0 acts on no data qubit"):
        qldpc_code(hz=[[0, 0]], hx=[[1, 1]])


def test_a_bound_the_search_does_not_reach_is_refused() -> None:
    """The record does not report a distance nothing checked.

    The bound is the caller's, so the refusal has to be theirs to see: the same
    matrices are accepted once the bound reaches the code's own distance, which
    is what separates a bound that was too small from a pair that states no code.
    """

    with pytest.raises(ValueError, match="no X-type logical operator of weight"):
        qldpc_code(hz=[[1, 1, 0], [0, 1, 1]], hx=[], distance_search_weight=1)

    accepted = qldpc_code(hz=[[1, 1, 0], [0, 1, 1]], hx=[], distance_search_weight=3)
    assert accepted.distance == 1


@pytest.mark.parametrize(
    ("block", "message"),
    [
        (1, "which is a sequence of sequences"),
        ([1], "hz row 0 must be a sequence of 0s and 1s"),
        ([[2, 0]], "hz row 0 carries 2"),
        ([[1.5, 0]], "hz row 0 carries 1.5"),
    ],
)
def test_a_block_that_is_not_a_matrix_of_zeros_and_ones_is_refused(
    block: object, message: str
) -> None:
    """The matrices are read the way the record reads them, refusals included.

    A caller who passes a tensor block is the case worth naming: the message says
    to read it with ``.tolist()`` first, so the fix is in the message rather than
    in a second route that accepts arrays.
    """

    with pytest.raises((TypeError, ValueError), match=message):
        qldpc_code(hz=block, hx=[[1, 1]])


def test_the_package_exports_the_constructor() -> None:
    """``qldpc_code`` is reachable from the package, not only from its module.

    A module-level function that is missing from the package's ``__all__`` is
    invisible to the public surface while every gate stays green, which is what
    this asserts against rather than the import alone.
    """

    from flagquantum import qec

    assert "qldpc_code" in qec.__all__, (
        "the package publishes the constructor it derives its records with, so a "
        "name that is missing from __all__ is a public surface that is not there"
    )
    assert qec.qldpc_code is qldpc_code
