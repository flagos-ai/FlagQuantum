"""The bosonic operator algebra and its dense value on a truncated Fock space.

The numeric assertions in this file are checked against an oracle written here rather than
against the algebra under test.  The oracle applies a raw factor sequence, rightmost factor
first, to an occupation basis state, using only the oscillator action
``a |n> = sqrt(n) |n - 1>`` and ``ad |n> = sqrt(n + 1) |n + 1>``; it never applies the
commutation relation, and it never reads a canonical monomial.  A dense matrix is then
assembled column by column.  An error in the normal-ordering rewrite, in the placement of
the creation and annihilation matrices, or in the truncation therefore shows up as a
disagreement between two matrices rather than as a disagreement the algebra can absorb.

The one statement that needs care is the relation between the two evaluations.  This module
applies the commutation relations to untruncated monomials, so ``a ad - ad a`` is the
identity, while the commutator of the two ladder matrices truncated to the same level count
differs from the identity at the top level.  The reconciliations used below are the same
ones the reference implementation uses: build the raw evaluation on a space with a few more
levels than needed and then keep the levels that were asked for, which is what the extra
room in the oracle reproduces.
"""

import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import flagquantum as fq
from flagquantum.observables import boson as boson_module
from flagquantum import observables
from flagquantum.observables.boson import (
    BosonOperator,
    BosonTerm,
    annihilate,
    create,
    identity,
    momentum,
    number,
    position,
)

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]

# A raw expansion is a list of (coefficient, factor sequence) pairs, where a factor is
# (degree, is_creation) and the sequence is applied from its last element to its first.
_Ladder = tuple[int, bool]
_RawTerm = tuple[complex, tuple[_Ladder, ...]]
_Raw = list[_RawTerm]

_SQRT = math.sqrt


def _ladder(degree: int, *, creation: bool) -> _Raw:
    """Write one ladder operator as a raw expansion of a single factor."""

    return [(1.0 + 0.0j, ((degree, creation),))]


def _scaled(factor: complex, expansion: _Raw) -> _Raw:
    return [(factor * coefficient, sequence) for coefficient, sequence in expansion]


def _plus(*expansions: _Raw) -> _Raw:
    return [item for expansion in expansions for item in expansion]


def _times(left: _Raw, right: _Raw) -> _Raw:
    """Concatenate two raw expansions, applying no commutation relation."""

    return [
        (left_coefficient * right_coefficient, left_sequence + right_sequence)
        for left_coefficient, left_sequence in left
        for right_coefficient, right_sequence in right
    ]


_A0 = _ladder(0, creation=False)
_AD0 = _ladder(0, creation=True)
_A1 = _ladder(1, creation=False)
_AD1 = _ladder(1, creation=True)

_POSITION_0 = _plus(_scaled(0.5, _AD0), _scaled(0.5, _A0))
_MOMENTUM_0 = _plus(_scaled(0.5j, _AD0), _scaled(-0.5j, _A0))
_POSITION_1 = _plus(_scaled(0.5, _AD1), _scaled(0.5, _A1))
_MOMENTUM_1 = _plus(_scaled(0.5j, _AD1), _scaled(-0.5j, _A1))

_SPACE = {0: 3, 1: 4}


def _occupations(index: int, dimensions: dict[int, int]) -> dict[int, int]:
    """Read a basis index as the occupation of every degree.

    The lowest degree is the least significant position, which is the same convention the
    module places its Kronecker factors in with the lowest degree leftmost.
    """

    levels: dict[int, int] = {}
    for degree in sorted(dimensions, reverse=True):
        levels[degree] = index % dimensions[degree]
        index //= dimensions[degree]
    return levels


def _index(levels: dict[int, int], dimensions: dict[int, int]) -> int:
    index = 0
    for degree in sorted(dimensions, reverse=True):
        index = index * dimensions[degree] + levels[degree]
    return index


def _side(dimensions: dict[int, int]) -> int:
    side = 1
    for dimension in dimensions.values():
        side *= dimension
    return side


def _dense(expansion: _Raw, dimensions: dict[int, int]) -> np.ndarray:
    """Evaluate a raw expansion on a truncated Fock space, factors applied last first."""

    side = _side(dimensions)
    matrix = np.zeros((side, side), dtype=complex)
    for column in range(side):
        occupied = _occupations(column, dimensions)
        for coefficient, sequence in expansion:
            levels = dict(occupied)
            amplitude = complex(coefficient)
            for degree, is_creation in reversed(sequence):
                level = levels[degree]
                if is_creation:
                    if level + 1 >= dimensions[degree]:
                        amplitude = 0.0
                        break
                    amplitude *= _SQRT(level + 1)
                    levels[degree] = level + 1
                else:
                    if level == 0:
                        amplitude = 0.0
                        break
                    amplitude *= _SQRT(level)
                    levels[degree] = level - 1
            if amplitude != 0.0:
                matrix[_index(levels, dimensions), column] += amplitude
    return matrix


def _agrees_with_raw(
    operator: BosonOperator, expansion: _Raw, dimensions: dict[int, int]
) -> bool:
    """Compare the algebra against the raw expansion, which needs extra levels.

    The canonical form contracts ``a ad`` into ``ad a + 1`` before any space is chosen, so
    the identity it produces is the truncation of an operator that acted on an unlimited
    space.  A raw expansion can only reproduce that if it is given room for every excursion
    above the kept levels, which is what the extra levels here are; a prefix never rises
    more than two levels, so three is more room than any expansion in this file uses.
    """

    roomy = {degree: dimension + 3 for degree, dimension in dimensions.items()}
    roomy_matrix = _dense(expansion, roomy)
    into_roomy = [
        _index(_occupations(index, dimensions), roomy)
        for index in range(_side(dimensions))
    ]
    expected = roomy_matrix[np.ix_(into_roomy, into_roomy)]
    return np.allclose(operator.to_matrix(dimensions).numpy(), expected)


def _power(operator: BosonOperator, exponent: int) -> BosonOperator:
    result = identity()
    for _ in range(exponent):
        result = result * operator
    return result


def test_create_annihilate_and_number_state_their_monomial() -> None:
    assert create(2).terms == (BosonTerm(1.0, (2,), ()),)
    assert annihilate(2).terms == (BosonTerm(1.0, (), (2,)),)
    assert number(2).terms == (BosonTerm(1.0, (2,), (2,)),)


def test_repeated_factors_are_powers_rather_than_zero() -> None:
    """Unlike the fermionic algebra, two equal factors multiply instead of vanishing."""

    assert (create(0) * create(0)).terms == (BosonTerm(1.0, (0, 0), ()),)
    assert (annihilate(0) * annihilate(0)).terms == (BosonTerm(1.0, (), (0, 0)),)
    assert (create(0) * annihilate(0)).terms == (
        BosonTerm(1.0, (), ()),
        BosonTerm(1.0, (0,), (0,)),
    )
    assert (create(0) * create(0) * create(0)).terms == (
        BosonTerm(1.0, (0, 0, 0), ()),
    )
    assert BosonOperator().terms == ()


def test_terms_that_share_a_monomial_are_combined_and_zero_terms_are_dropped() -> None:
    combined = BosonOperator(
        (BosonTerm(1.0, (0,), ()), BosonTerm(2.0, (0,), ()), BosonTerm(4.0, (1,), ()))
    )
    assert combined.terms == (BosonTerm(3.0, (0,), ()), BosonTerm(4.0, (1,), ()))
    assert BosonOperator((BosonTerm(0.0, (0,), ()),)).terms == ()
    assert (create(0) * 0).terms == ()
    assert (0 * create(0)).terms == ()


def test_addition_subtraction_and_scalar_multiplication_are_the_algebra_operations() -> (
    None
):
    left = create(0) + create(1)
    assert left.terms == (BosonTerm(1.0, (0,), ()), BosonTerm(1.0, (1,), ()))
    assert (left - create(1)).terms == (BosonTerm(1.0, (0,), ()),)
    assert (left + create(1)).terms == (
        BosonTerm(1.0, (0,), ()),
        BosonTerm(2.0, (1,), ()),
    )
    assert (2j * create(0)).terms == (BosonTerm(2j, (0,), ()),)
    assert (create(0) * 2j).terms == (BosonTerm(2j, (0,), ()),)
    assert (-create(0)).terms == (BosonTerm(-1.0, (0,), ()),)
    assert left.degrees == (0, 1)
    assert BosonOperator().degrees == ()


def test_a_scalar_multiplication_refuses_a_non_numeric_factor() -> None:
    """A refused scalar leaves the reflected operator to report the mismatch."""

    with pytest.raises(TypeError, match="multiply"):
        _ = create(0) * "two"
    with pytest.raises(TypeError, match="unsupported operand"):
        _ = create(0) * object()
    # A boolean is an integer to Python but not a coefficient to this algebra.
    with pytest.raises(TypeError, match="unsupported operand"):
        _ = create(0) * True


def test_a_sum_refuses_a_value_that_is_not_an_operator() -> None:
    """A sum is a sum of operators, so a scalar is refused rather than promoted."""

    with pytest.raises(TypeError, match="unsupported operand"):
        _ = create(0) + 1.0
    with pytest.raises(TypeError, match="unsupported operand"):
        _ = create(0) - 1.0


def test_creation_and_annihilation_obey_the_canonical_commutation_relations() -> None:
    for degree in range(3):
        assert annihilate(degree).commutator(create(degree)) == identity()
        assert create(degree).commutator(annihilate(degree)) == -identity()
        assert annihilate(degree).commutator(annihilate(degree)) == BosonOperator()
        assert create(degree).commutator(create(degree)) == BosonOperator()
    for left, right in ((0, 1), (1, 2), (0, 2), (2, 0)):
        assert annihilate(left).commutator(create(right)) == BosonOperator()
        assert create(left).commutator(create(right)) == BosonOperator()
        assert annihilate(left).commutator(annihilate(right)) == BosonOperator()
    assert create(0) * create(1) == create(1) * create(0)
    assert annihilate(0) * annihilate(1) == annihilate(1) * annihilate(0)
    assert number(1) * number(0) == number(0) * number(1)


def test_the_commutator_terminates_at_the_first_contraction() -> None:
    """``[a**2, ad**2] = 4 ad a + 2``, written out one contraction at a time."""

    assert (_power(annihilate(0), 2) * _power(create(0), 2)).terms == (
        BosonTerm(2.0, (), ()),
        BosonTerm(4.0, (0,), (0,)),
        BosonTerm(1.0, (0, 0), (0, 0)),
    )


def test_the_adjoint_exchanges_the_two_groups() -> None:
    term = BosonTerm(1j, (0, 1), (2,))
    assert term.dagger() == BosonTerm(-1j, (2,), (0, 1))
    assert term.dagger().dagger() == term
    operator = 2.0 * create(0) + 3j * annihilate(1)
    assert operator.dagger() == 2.0 * annihilate(0) - 3j * create(1)
    assert operator.dagger().dagger() == operator
    assert BosonOperator().dagger() == BosonOperator()


def test_hermiticity_is_an_exact_structural_test() -> None:
    assert (create(0) + annihilate(0)).is_hermitian()
    assert (1j * (create(0) - annihilate(0))).is_hermitian()
    assert number(0).is_hermitian()
    assert identity().is_hermitian()
    assert not create(0).is_hermitian()
    assert not (create(0) + 1e-30j * annihilate(0)).is_hermitian()


def test_commutator_and_anticommutator_follow_their_definitions() -> None:
    left, right = create(0) + 1j * create(1), annihilate(0)
    assert left.commutator(right) == left * right - right * left
    assert left.anticommutator(right) == left * right + right * left
    assert annihilate(0).anticommutator(create(0)) == identity() + 2 * number(0)
    with pytest.raises(TypeError, match="two bosonic operators"):
        left.commutator(1.0)
    with pytest.raises(TypeError, match="two bosonic operators"):
        left.anticommutator(1.0)


def test_position_and_momentum_are_hermitian_and_satisfy_the_canonical_commutator() -> (
    None
):
    assert position(0) == 0.5 * (create(0) + annihilate(0))
    assert momentum(0) == 0.5j * (create(0) - annihilate(0))
    assert position(0).is_hermitian()
    assert momentum(0).is_hermitian()
    # [x, p] = i/2 for x = (a + ad)/2 and p = i(ad - a)/2, which is [a, ad] = 1 halved.
    assert position(0).commutator(momentum(0)) == 0.5j * identity()
    assert position(0).commutator(position(1)) == BosonOperator()
    assert np.allclose(
        position(1).to_matrix({1: 3}).numpy(),
        _dense(_POSITION_1, {1: 3}),
    )


def test_the_number_operator_counts_levels() -> None:
    assert number(0).to_matrix({0: 5}).diagonal().numpy() == pytest.approx(
        np.arange(5, dtype=complex)
    )
    assert np.allclose(number(0).to_matrix(_SPACE).numpy(), _dense(_times(_AD0, _A0), _SPACE))


def test_the_zero_operator_and_the_unit_are_values_of_the_algebra() -> None:
    assert identity().to_matrix().numpy() == pytest.approx(np.eye(1))
    assert BosonOperator().to_matrix().numpy() == pytest.approx(np.zeros((1, 1)))
    assert identity(3) == identity()
    assert identity().to_matrix({0: 3, 1: 2}).numpy() == pytest.approx(np.eye(6))
    assert (create(0) + BosonOperator()).terms == create(0).terms
    assert (create(0) * identity()).terms == create(0).terms


def test_an_undeclared_space_defaults_every_acted_on_degree_to_two_levels() -> None:
    assert create(0).to_matrix().shape == (2, 2)
    assert (create(0) * create(1)).to_matrix().shape == (4, 4)
    assert number(0).to_matrix().diagonal().numpy() == pytest.approx([0.0, 1.0])


def test_the_dense_ladder_matrices_are_the_oscillator_action() -> None:
    """Pin the convention: ``create[n, n - 1] = sqrt(n)`` and its transpose."""

    levels = 5
    expected_create = np.zeros((levels, levels), dtype=complex)
    expected_annihilate = np.zeros((levels, levels), dtype=complex)
    for level in range(levels - 1):
        expected_create[level + 1, level] = _SQRT(level + 1)
        expected_annihilate[level, level + 1] = _SQRT(level + 1)
    assert create(0).to_matrix({0: levels}).numpy() == pytest.approx(expected_create)
    assert annihilate(0).to_matrix({0: levels}).numpy() == pytest.approx(
        expected_annihilate
    )
    assert np.allclose(create(0).to_matrix({0: levels}).numpy(), _dense(_AD0, {0: levels}))
    assert np.allclose(annihilate(0).to_matrix({0: levels}).numpy(), _dense(_A0, {0: levels}))


def test_the_dense_form_of_a_sum_is_the_sum_of_the_dense_forms() -> None:
    operator = 1.5 * position(0) * momentum(1) - 2.0 * number(0)
    assert np.allclose(
        operator.to_matrix(_SPACE).numpy(),
        1.5 * _dense(_times(_POSITION_0, _MOMENTUM_1), _SPACE)
        - 2.0 * _dense(_times(_AD0, _A0), _SPACE),
    )


def test_the_dense_form_of_a_product_matches_the_unrewritten_product() -> None:
    """The rewrite is checked against a product that never had a commutation applied."""

    assert _agrees_with_raw(annihilate(0) * create(0), _times(_A0, _AD0), _SPACE)
    assert _agrees_with_raw(create(0) * annihilate(0), _times(_AD0, _A0), _SPACE)
    assert _agrees_with_raw(
        annihilate(0) * create(0) * create(1),
        _times(_times(_A0, _AD0), _AD1),
        _SPACE,
    )
    assert _agrees_with_raw(
        position(0) * momentum(1),
        _times(_POSITION_0, _MOMENTUM_1),
        _SPACE,
    )
    assert _agrees_with_raw(
        _power(annihilate(0), 2) * _power(create(0), 2),
        _times(_times(_A0, _A0), _times(_AD0, _AD0)),
        _SPACE,
    )


def test_the_lowest_degree_is_the_most_significant_index() -> None:
    dimensions = {0: 2, 1: 3}
    left = annihilate(0).to_matrix({0: 2}).numpy()
    right = create(1).to_matrix({1: 3}).numpy()
    measured = (annihilate(0) * create(1)).to_matrix(dimensions).numpy()
    assert measured == pytest.approx(np.kron(left, right))
    assert not np.allclose(measured, np.kron(right, left))
    assert np.allclose(measured, _dense(_times(_A0, _AD1), dimensions))


def test_a_declared_degree_the_operator_does_not_act_on_becomes_an_identity_factor() -> (
    None
):
    dimensions = {0: 2, 1: 3}
    measured = annihilate(0).to_matrix(dimensions).numpy()
    assert measured == pytest.approx(
        np.kron(annihilate(0).to_matrix({0: 2}).numpy(), np.eye(3))
    )
    assert measured.shape == (6, 6)
    assert number(0).to_matrix(dimensions).numpy().shape == (6, 6)


def test_the_truncation_reconciles_the_canonical_commutator_with_the_ladder_matrices() -> (
    None
):
    """State which operator the truncation makes the relation true of.

    The algebra contracts ``a ad - ad a`` to the identity before a space is chosen, so its
    dense value is the identity at every level count.  The commutator of the two ladder
    matrices truncated to the same count is a different matrix, and the two agree on the
    levels the smaller space can hold once the commutator is built on one level more.
    """

    levels = 5
    ladder = annihilate(0).to_matrix({0: levels}).numpy()
    adjoint = create(0).to_matrix({0: levels}).numpy()
    assert (ladder @ adjoint - adjoint @ ladder)[:4, :4] == pytest.approx(np.eye(4))

    bounded = annihilate(0).to_matrix({0: 4}).numpy()
    bounded_adjoint = create(0).to_matrix({0: 4}).numpy()
    commutator = bounded @ bounded_adjoint - bounded_adjoint @ bounded
    assert commutator == pytest.approx(np.diag([1.0, 1.0, 1.0, -3.0]))
    assert not np.allclose(commutator, np.eye(4))

    canonical = annihilate(0).commutator(create(0))
    assert canonical == identity()
    assert canonical.to_matrix({0: 4}).numpy() == pytest.approx(np.eye(4))


def test_the_dense_form_refuses_a_space_that_does_not_declare_every_acted_on_degree() -> (
    None
):
    with pytest.raises(ValueError, match=r"acts on degree\(s\) \[0\]"):
        create(0).to_matrix({1: 3})
    with pytest.raises(TypeError, match="must map a degree to a level count"):
        create(0).to_matrix([3])
    # A declared degree is read by the same protocol as the degree it is asked about, so a
    # boolean key is refused rather than silently read as degree 1.
    with pytest.raises(TypeError, match="must be an integer"):
        create(0).to_matrix({True: 3})


def test_the_dense_form_refuses_a_truncation_that_keeps_no_level() -> None:
    with pytest.raises(ValueError, match="must keep at least one level"):
        create(0).to_matrix({0: 0})
    with pytest.raises(ValueError, match="non-negative integer"):
        create(0).to_matrix({0: -1})
    with pytest.raises(TypeError, match="truncation must be an integer"):
        create(0).to_matrix({0: 2.0})


def test_the_dense_form_refuses_a_matrix_above_the_byte_ceiling() -> None:
    with pytest.raises(ValueError, match="byte ceiling"):
        create(0).to_matrix({0: 1025})
    with pytest.raises(ValueError, match="byte ceiling"):
        identity().to_matrix(dict.fromkeys(range(11), 2))


def test_a_degree_is_read_by_the_package_integer_protocol() -> None:
    with pytest.raises(TypeError, match="must be an integer"):
        create(True)
    with pytest.raises(TypeError, match="must be an integer"):
        create(0.5)
    with pytest.raises(ValueError, match="non-negative integer"):
        annihilate(-1)
    with pytest.raises(TypeError, match="must be an integer"):
        number("0")
    with pytest.raises(TypeError, match="must be an integer"):
        identity(1.5)
    assert create(np.int64(2)) == create(2)


def test_a_term_refuses_an_unusable_coefficient_or_a_degree_order() -> None:
    with pytest.raises(TypeError, match="numeric coefficient"):
        BosonTerm(True, (0,), ())
    with pytest.raises(ValueError, match="must be finite"):
        BosonTerm(float("inf"), (0,), ())
    with pytest.raises(ValueError, match="ascending order"):
        BosonTerm(1.0, (1, 0), ())
    with pytest.raises(ValueError, match="ascending order"):
        BosonTerm(1.0, (), (2, 1))
    with pytest.raises(TypeError, match="must be an integer"):
        BosonTerm(1.0, (1.5,), ())
    # A repeated degree is a power here rather than the error it is for a fermion.
    assert BosonTerm(1.0, (0, 0), (0,)) == BosonTerm(1.0, (0, 0), (0,))


def test_an_operator_refuses_a_value_that_is_not_a_bosonic_term() -> None:
    with pytest.raises(TypeError, match="holds BosonTerm values"):
        BosonOperator((create(0),))


def test_the_observables_namespace_exposes_the_bosonic_module() -> None:
    """The algebra is reachable from the package namespace, lazily and by submodule.

    The bosonic names are not flattened into `flagquantum.observables`, because `create`
    and `annihilate` already mean the fermionic operators there.  The module is named in
    `__all__` and resolved on first use instead.
    """

    assert "boson" in observables.__all__
    assert boson_module.create is create
    assert boson_module.BosonOperator is BosonOperator
    with pytest.raises(AttributeError, match="has no attribute 'no_such_name'"):
        _ = observables.no_such_name


def test_the_bosonic_algebra_stays_out_of_the_root_namespace() -> None:
    for name in ("BosonOperator", "BosonTerm", "boson", "position", "momentum"):
        assert name not in fq.__all__
    assert not hasattr(fq, "BosonOperator")


def test_the_bosonic_module_is_resolved_on_first_use_rather_than_imported() -> None:
    """Naming the module in `__all__` must not import it, so the read stays explicit.

    The package already reaches torch through `flagquantum.core`, so the property under
    test is the submodule's own laziness: reading the namespace must not add the bosonic
    module to `sys.modules`, and resolving the attribute must.
    """

    probe = (
        "import sys\n"
        "import flagquantum.observables as observables\n"
        "assert 'flagquantum.observables.boson' not in sys.modules, sorted(\n"
        "    name for name in sys.modules if 'observables' in name\n"
        ")\n"
        "assert 'boson' in observables.__all__\n"
        "boson = observables.boson\n"
        "assert 'flagquantum.observables.boson' in sys.modules\n"
        "assert boson.position is observables.boson.position\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert completed.returncode == 0, completed.stderr
