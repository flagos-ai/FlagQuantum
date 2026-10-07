"""Tests for the bivariate-bicycle family and the logical basis it derives.

The family is the second route in this package that derives a code from a rule
rather than taking matrices from a caller, and unlike the two derived families
beside it -- the triangular colour patch and the square-lattice torus -- it also
derives the logical operators, because a pair of polynomials names no curve a
reader could write down. What these tests measure is therefore two things at
once: that the parameters produce the code the family claims, and that the
operators the record reports really are logical operators of the matrices rather
than something asserted about them.
"""

from __future__ import annotations

from collections import Counter

import pytest

from flagquantum.qec import (
    CssCode,
    Pauli,
    bivariate_bicycle_code,
    build_memory_circuit,
    certify_logical_product,
    css_code_matrices,
    derive_anticommuting_logical_product,
    toric_code,
)
from flagquantum.qec.gf2 import in_span, rank

pytestmark = pytest.mark.unit

# The smallest instance this file builds and the only one whose distance the
# search reaches immediately: three-term polynomials over Z_3 * Z_3, which is the
# eighteen-qubit member of the family.
_SMALL = ((0, 0), (0, 1), (1, 0)), ((0, 0), (0, 1), (2, 1))

# The published instance: [[72, 12, 6]] over Z_6 * Z_6. Its build is dominated by
# the distance search rather than by the elimination, which is why the one test
# that uses it carries the slow marker.
_PUBLISHED = ((3, 0), (0, 1), (0, 2)), ((0, 3), (1, 0), (2, 0))


def _bits(rows: list[list[int]]) -> list[int]:
    """Pack each 0/1 row into one integer, which is what the GF(2) kernel takes."""

    return [int("".join(str(bit) for bit in row), 2) for row in rows]


def _small() -> CssCode:
    """Return the eighteen-qubit instance, which is what most tests measure."""

    return bivariate_bicycle_code(3, 3, *_SMALL, distance_search_weight=4)


def _pack(observable: Pauli, code: CssCode) -> int:
    """Return one observable as a bit mask over the code's own data wires."""

    positions = {wire: index for index, wire in enumerate(code.data_wires)}
    return sum(1 << positions[wire] for wire in observable.support)


def _supports(code: CssCode, *, family: str) -> tuple[tuple[int, ...], ...]:
    """Return one family's check supports, in the record's own order."""

    attribute = "z_wires" if family == "z" else "x_wires"
    return tuple(
        tuple(getattr(check.stabilizer, attribute))
        for check in code.checks
        if getattr(check.stabilizer, attribute)
    )


def test_the_smallest_instance_reproduces_its_parameters() -> None:
    """Eighteen data qubits, four logical qubits, distance four.

    Every number here is read off the record rather than off the polynomials, so
    the assertion is about what the construction produced. The logical count is
    the measurement that matters: the check matrices of this instance have
    eighteen rows each and rank seven, so ``18 - 7 - 7`` leaves four, and a
    construction that counted rows instead of rank would report no logical qubit
    at all and refuse the instance.
    """

    code = _small()
    matrices = css_code_matrices(code)
    z_rank = rank(_bits(matrices.hz.tolist()))
    x_rank = rank(_bits(matrices.hx.tolist()))

    assert code.num_data_qubits == 18
    assert code.num_ancilla_qubits == 18
    assert z_rank == x_rank == 7
    assert code.num_data_qubits - z_rank - x_rank == 4
    assert len(code.logical_observables) == 8
    assert code.distance == 4
    assert code.x_distance == code.z_distance == 4


@pytest.mark.parametrize(
    ("x_period", "y_period", "distance"),
    ((2, 2, 2), (2, 3, 2), (3, 3, 3), (4, 2, 2), (3, 4, 3)),
)
def test_the_two_periods_state_the_qubit_count(
    x_period: int, y_period: int, distance: int
) -> None:
    """The data qubits are two copies of the group, one per check family.

    A bivariate-bicycle code is Calderbank-Shor-Steane, so its two check blocks
    are indexed by the same group and each family holds one check per element. The
    count is stated here as arithmetic on the periods rather than as the length of
    anything the builder returned.

    The two-term pair ``A = 1 + x``, ``B = 1 + y`` carries the assertion across
    five sizes for the cost of five eliminations, and the distance is asserted
    beside the count because the distance is what the bound was set to: a bound
    silently larger than the distance would leave the count untested.
    """

    code = bivariate_bicycle_code(
        x_period,
        y_period,
        ((0, 0), (1, 0)),
        ((0, 0), (0, 1)),
        distance_search_weight=distance,
    )
    elements = x_period * y_period

    assert code.num_data_qubits == 2 * elements
    assert code.num_ancilla_qubits == 2 * elements
    assert code.distance == distance
    assert len(_supports(code, family="z")) == elements
    assert len(_supports(code, family="x")) == elements


def test_every_check_spans_at_most_six_data_qubits() -> None:
    """The weight is bounded by the polynomials, not by the periods.

    This is the family's reason for existing: a check is supported on the images of
    one group element under the terms of one polynomial, on one side of the block
    split, so a three-term ``A`` and a three-term ``B`` state checks of weight at
    most six however large the periods are. The square torus is measured beside it
    as the control: its checks have weight four at every size, so a weight-six
    bound is not merely what the measurement returns for every code.
    """

    code = _small()
    weights = Counter(
        len(support)
        for family in ("z", "x")
        for support in _supports(code, family=family)
    )

    assert set(weights) <= {1, 2, 3, 4, 5, 6}
    assert max(weights) == 6
    assert sum(weights.values()) == 18

    torus = toric_code(3)
    torus_weights = {
        len(support)
        for family in ("z", "x")
        for support in _supports(torus, family=family)
    }
    assert torus_weights == {4}


def test_the_two_check_families_commute() -> None:
    """The construction's validity, asserted between the record's own checks.

    The record's constructor refuses matrices whose two families do not commute,
    so a passing construction is already evidence of this. The assertion is made
    anyway, and against the record's checks rather than against the polynomials,
    because otherwise a construction that had built the wrong pair of matrices
    would look like a proof of the right one.
    """

    code = _small()
    for z_support in _supports(code, family="z"):
        z_check = Pauli(z_wires=z_support)
        for x_support in _supports(code, family="x"):
            assert z_check.commutes_with(Pauli(x_wires=x_support))


def test_the_logical_operators_are_read_out_of_the_matrices() -> None:
    """Both families' operators are derived, and they are operators of this code.

    Each derived operator is measured on the three properties that make it one: it
    commutes with every check of both families, it is not a combination of its own
    family's checks, and its family holds one operator per logical qubit the
    matrices leave. The second property is the one a construction could get wrong
    silently: a row of a check matrix commutes with the checks too, and would be
    reported as a logical operator by a route that never asked whether the row was
    a stabilizer.
    """

    code = _small()
    checks = [
        Pauli(z_wires=check.stabilizer.z_wires, x_wires=check.stabilizer.x_wires)
        for check in code.checks
    ]
    z_observables = [
        observable for observable in code.logical_observables if observable.z_wires
    ]
    x_observables = [
        observable for observable in code.logical_observables if observable.x_wires
    ]

    assert len(z_observables) == len(x_observables) == 4
    for observable in (*z_observables, *x_observables):
        for check in checks:
            assert observable.commutes_with(check)

    for family, observables, own in (
        ("z", z_observables, code.hz),
        ("x", x_observables, code.hx),
    ):
        own_rows = _bits([list(row) for row in own])
        for observable in observables:
            assert not in_span(_pack(observable, code), own_rows), family


def test_the_two_families_pair_as_the_identity() -> None:
    """Each Z-type operator anticommutes with exactly one X-type operator.

    A complement basis of each null space is an arbitrary basis, so nothing forces
    the pairing between the two families to be the identity; what forces it is the
    symplectic elimination the family runs after deriving them. The weaker
    property the record certifies is that the pairing has full rank, so this test
    is what separates the two: a permuted pairing passes the record and fails
    here. The order is what a caller reads through, since ``lz[i]`` and ``lx[i]``
    are then partners.
    """

    code = _small()
    z_observables = [
        observable for observable in code.logical_observables if observable.z_wires
    ]
    x_observables = [
        observable for observable in code.logical_observables if observable.x_wires
    ]
    pairing = [
        [0 if z.commutes_with(x) else 1 for x in x_observables] for z in z_observables
    ]

    assert pairing == [
        [1 if row == column else 0 for column in range(4)] for row in range(4)
    ]


def test_the_derived_observables_drive_both_memory_frames() -> None:
    """The operators a caller reads out are the ones the derivation produced.

    A logical operator that commutes with every check and lies outside the
    stabilizer span is still only a candidate until the route that consumes it
    accepts it. Both memory frames are therefore built on the derived operators:
    the Z-basis frame declares one observable per derived Z-type operator, and the
    rotated frame declares the X-type partner of the first of them, which the
    record certifies and reads out on exactly its own X wires.
    """

    code = _small()
    z_observables = [
        observable for observable in code.logical_observables if observable.z_wires
    ]

    memory = build_memory_circuit(code, rounds=2)
    assert len(memory.observables) == len(z_observables)
    assert [
        observable.pauli for observable in memory.observables.observables
    ] == z_observables
    assert certify_logical_product(code, z_observables[0]) == z_observables[0]

    partner = derive_anticommuting_logical_product(code, 0)
    rotated = build_memory_circuit(code, rounds=2, product=partner)
    assert len(rotated.observables) == 1
    assert rotated.x_readout_wires == partner.x_wires
    assert partner.x_wires


def test_a_monomial_named_twice_states_no_term() -> None:
    """A polynomial over GF(2) has coefficients, so a repeated term cancels.

    The terms are read as a sum rather than as a set, and the difference is
    visible: naming a monomial the polynomial does not otherwise carry, twice,
    states the same polynomial as not naming it at all. The two records are
    compared through their check matrices, so the test measures the arithmetic
    rather than the record's canonical form.
    """

    code = _small()
    a_terms, b_terms = _SMALL
    repeated = bivariate_bicycle_code(
        3,
        3,
        (*a_terms, (2, 2), (2, 2)),
        b_terms,
        distance_search_weight=4,
    )

    assert repeated.hz == code.hz
    assert repeated.hx == code.hx


def test_exponents_are_read_modulo_their_own_period() -> None:
    """An exponent outside the period wraps rather than being refused.

    Writing ``x ** -1`` is writing the last power of ``x``, and a rule that took
    the exponent literally would either index outside the group or silently drop
    the term. The wrapped pair below is the published one with each exponent
    negative where the period allows, so the equality is between two spellings of
    one polynomial rather than between a polynomial and itself.

    The two periods then differ, which is what makes it visible that each exponent
    is read against its own: ``x ** 2`` is the identity in a coordinate of order two
    and ``y ** 2`` is not, so the two sums below cancel to the zero polynomial and
    the pair of them would cancel under neither reading if the exponents were read
    against each other's period. The refusal is matched rather than left as any
    error, because a construction that read the exponent against the wrong period
    would go on to build a code instead of stopping.
    """

    code = _small()
    wrapped = bivariate_bicycle_code(
        3,
        3,
        ((-2, 0), (0, 1), (0, 0)),
        ((0, 0), (0, -2), (2, 1)),
        distance_search_weight=4,
    )

    assert wrapped.hz == code.hz
    assert wrapped.hx == code.hx

    for cancelling in (((0, 0), (2, 0)), ((0, 0), (0, 3))):
        with pytest.raises(ValueError, match="a_terms names no monomial"):
            bivariate_bicycle_code(2, 3, cancelling, ((0, 1),))


def test_a_bound_the_search_does_not_reach_is_refused() -> None:
    """The default bound is three, so the distance-four instance is refused.

    The refusal is the record's own, restated by this route rather than replaced:
    a distance nothing checked is not reported, and the message names the bound so
    that the caller knows which number to raise. The same parameters are then
    built at a bound the search reaches, so the refusal is about the bound and not
    about the parameters.
    """

    with pytest.raises(
        ValueError, match="no X-type logical operator of weight at most 3"
    ):
        bivariate_bicycle_code(3, 3, *_SMALL)

    assert _small().distance == 4


@pytest.mark.parametrize("period", (0, 1, -3))
def test_a_trivial_cyclic_coordinate_is_refused(period: int) -> None:
    """A coordinate of order below two is a one-dimensional bicycle, not this one.

    ``y_period = 1`` makes ``y`` trivial, which collapses the two-variable rule to
    a single-variable one whose codes are a different family with different
    parameters. The route refuses it rather than building the smaller family under
    this family's name.
    """

    with pytest.raises(ValueError, match="at least two"):
        bivariate_bicycle_code(3, period, ((0, 0),), ((1, 0),))


def test_a_polynomial_that_names_no_monomial_is_refused() -> None:
    """Naming one monomial twice states the zero polynomial, which states no check.

    An all-zero check block would build a code whose X-type family is empty and
    whose logical operators are then the whole space, so the record would report a
    code the parameters do not describe. The route refuses the parameters instead.
    """

    with pytest.raises(ValueError, match="names no monomial"):
        bivariate_bicycle_code(3, 3, ((0, 0), (0, 0)), ((1, 0),))


@pytest.mark.parametrize(
    ("a_terms", "b_terms", "error"),
    (
        (((0, 0, 0),), ((1, 0),), ValueError),
        (((0, 0),), ((1,),), ValueError),
        ((5,), ((1, 0),), TypeError),
        (("ab",), ((1, 0),), TypeError),
        (((0.0, 0),), ((1, 0),), TypeError),
        (((0, True),), ((1, 0),), TypeError),
    ),
)
def test_a_term_that_is_not_a_pair_of_integers_is_refused(
    a_terms: object, b_terms: object, error: type[Exception]
) -> None:
    """A term is a pair of integers, and every other shape is named as it is.

    The three shapes are separated because they fail for different reasons: a
    three-element term states no monomial, a bare integer is not a term at all,
    and a term whose exponent is a float or a boolean is a term this route would
    have to guess the group element of. Boolean exponents are included because
    ``True`` is an integer in Python and is not an exponent here.
    """

    with pytest.raises(error):
        bivariate_bicycle_code(3, 3, a_terms, b_terms)  # type: ignore[arg-type]


@pytest.mark.slow
def test_the_published_instance_reproduces_its_parameters() -> None:
    """The family's claim is checked against an instance nobody here chose.

    ``[[72, 12, 6]]`` over ``Z_6 * Z_6`` is the smallest published member of the
    family, and it is the only assertion in this file whose expected numbers come
    from outside the repository. It is marked slow because the distance search
    dominates the build -- about eighteen seconds, against no measurable time for
    the eighteen-qubit instance -- and a published distance is what the search is
    for, so building it at a lower bound would prove a different statement.

    The k here is the one a row count would get wrong by a wide margin: the two
    check families have rank 30 each over 72 data qubits, so the twelve logical
    qubits are invisible to a route that subtracts the number of checks.
    """

    code = bivariate_bicycle_code(6, 6, *_PUBLISHED, distance_search_weight=6)
    matrices = css_code_matrices(code)
    z_rank = rank(_bits(matrices.hz.tolist()))
    x_rank = rank(_bits(matrices.hx.tolist()))

    assert code.num_data_qubits == 72
    assert code.num_data_qubits - z_rank - x_rank == 12
    assert code.distance == 6
    assert code.x_distance == code.z_distance == 6


def test_the_bicycle_reaches_further_than_the_torus_at_the_same_qubit_count() -> None:
    """The comparison that says why the family is here, at the size where both fit.

    Eighteen data qubits is the smallest count the two families share, and the
    torus at that count is a three-by-three lattice whose two logical qubits are
    three errors apart. The bicycle's four logical qubits are four errors apart on
    the same number of qubits, at checks of higher weight. Both numbers are read
    off the two records rather than argued, and the torus is measured through the
    same route as the control: a test that only stated the bicycle's parameters
    could not tell a genuinely better family from a mis-measured one.
    """

    torus = toric_code(3)
    bicycle = _small()

    assert torus.num_data_qubits == bicycle.num_data_qubits == 18
    assert len(torus.logical_observables) // 2 == 2
    assert len(bicycle.logical_observables) // 2 == 4
    assert torus.distance == 3
    assert bicycle.distance == 4


def test_the_package_exports_the_constructor() -> None:
    """The family is reachable from the package rather than only from its module.

    A name that a module defines but ``__all__`` does not carry is a name the
    package does not promise, and the difference is invisible to every other test
    here: they import the constructor from the package, which resolves through the
    module whether or not the export list names it. The assertion is therefore made
    on ``__all__`` itself and against the object the package holds, which is the
    same statement ``test_toric_code.py`` makes about the other derived family.
    """

    import flagquantum.qec as qec

    assert hasattr(qec, "bivariate_bicycle_code")
    assert "bivariate_bicycle_code" in qec.__all__
    assert qec.bivariate_bicycle_code is bivariate_bicycle_code
