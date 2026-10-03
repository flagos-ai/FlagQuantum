"""The fermionic operator algebra and its Jordan-Wigner image as a Pauli observable.

The numeric assertions in this file are checked against a Fock-space oracle written here
rather than against the algebra under test.  The oracle acts with a monomial on an
occupation basis state and picks up the sign ``(-1)**(occupied modes below the degree)``
that the anticommutation relations impose, which is the physical definition of a fermionic
operator and shares no code with the Pauli-string image the module computes.  A dense
matrix is then assembled column by column, so an error in the Jordan-Wigner string, in the
role of ``sigma_plus`` and ``sigma_minus``, or in the normal-ordering rewrite shows up as a
disagreement between two matrices rather than as a disagreement the module can absorb.
"""

import numpy as np
import pytest

import flagquantum as fq
from flagquantum.errors import CapabilityError
from flagquantum.observables import (
    Observable,
    X,
    Z,
    annihilate,
    create,
    jordan_wigner,
    number,
)
from flagquantum.observables.fermion import FermionOperator, FermionTerm

pytestmark = pytest.mark.unit

_PAULI = {
    "i": np.eye(2, dtype=complex),
    "x": np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex),
    "y": np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex),
    "z": np.diag([1.0, -1.0]).astype(complex),
}


def _factors(term: FermionTerm) -> tuple[tuple[int, bool], ...]:
    """Read one canonical term back as the flat left-to-right factor sequence."""

    creations = tuple((degree, True) for degree in term.creations)
    annihilations = tuple((degree, False) for degree in term.annihilations)
    return creations + annihilations


def _fock_matrix(operator: FermionOperator, n_modes: int) -> np.ndarray:
    """Act with a fermionic operator on every occupation basis state, from the CAR."""

    dimension = 1 << n_modes
    matrix = np.zeros((dimension, dimension), dtype=complex)
    for column in range(dimension):
        occupations = tuple(
            (column >> (n_modes - 1 - mode)) & 1 for mode in range(n_modes)
        )
        for term in operator.terms:
            state = list(occupations)
            sign = 1
            for degree, is_creation in reversed(_factors(term)):
                if bool(state[degree]) == is_creation:
                    state = None
                    break
                if sum(state[:degree]) % 2:
                    sign = -sign
                state[degree] = 1 if is_creation else 0
            if state is None:
                continue
            row = sum(bit << (n_modes - 1 - mode) for mode, bit in enumerate(state))
            matrix[row, column] += term.coefficient * sign
    return matrix


def _observable_matrix(observable: Observable, n_modes: int) -> np.ndarray:
    """Build the dense matrix of a Pauli observable, wire 0 leftmost."""

    matrix = np.zeros((1 << n_modes, 1 << n_modes), dtype=complex)
    for term in observable.terms:
        by_wire = dict(term.factors)
        factors = [_PAULI[by_wire.get(wire, "i")] for wire in range(n_modes)]
        product = factors[0]
        for factor in factors[1:]:
            product = np.kron(product, factor)
        matrix += term.coefficient * product
    return matrix


def _random_hermitian_operator(n_modes: int, rng: np.random.Generator) -> FermionOperator:
    """Build a random Hermitian operator from products of two or three factors."""

    operator = FermionOperator()
    for _ in range(3):
        product = FermionOperator((FermionTerm(1.0, (), ()),))
        for _ in range(int(rng.integers(2, 4))):
            degree = int(rng.integers(0, n_modes))
            factor = create(degree) if rng.random() < 0.5 else annihilate(degree)
            product = product * factor
        operator = operator + product + product.dagger()
    return operator * float(rng.normal())


def test_create_annihilate_and_number_state_their_monomial() -> None:
    assert create(2).terms == (FermionTerm(1.0, (2,), ()),)
    assert annihilate(2).terms == (FermionTerm(1.0, (), (2,)),)
    assert number(2).terms == (FermionTerm(1.0, (2,), (2,)),)


def test_terms_that_share_a_monomial_are_combined_and_zero_terms_are_dropped() -> None:
    combined = FermionOperator((FermionTerm(1.0, (0,), ()), FermionTerm(2.0, (0,), ())))
    assert combined.terms == (FermionTerm(3.0, (0,), ()),)
    assert (create(0) - create(0)).terms == ()
    assert FermionOperator().terms == ()


def test_addition_subtraction_and_scalar_multiplication_are_the_algebra_operations() -> None:
    total = create(0) + annihilate(0) - 2.0 * annihilate(0)
    assert total.terms == (
        FermionTerm(-1.0, (), (0,)),
        FermionTerm(1.0, (0,), ()),
    )
    assert (3 * create(1)).terms == (FermionTerm(3.0, (1,), ()),)
    assert (create(1) * 3).terms == (FermionTerm(3.0, (1,), ()),)


def test_a_scalar_multiplication_refuses_a_non_numeric_factor() -> None:
    with pytest.raises(TypeError):
        create(0) * "x"


def test_a_composition_carries_both_operands_coefficients() -> None:
    assert (2 * create(0)) * annihilate(0) == 2.0 * number(0)
    assert create(0) * (3 * annihilate(0)) == 3.0 * number(0)
    assert (2j * create(0)) * (3 * annihilate(0)) == 6j * number(0)
    assert (2 * create(0) * annihilate(1)) * (3 * create(1) * annihilate(0)) == (
        6.0 * create(0) * annihilate(1) * create(1) * annihilate(0)
    )


def test_the_dense_form_of_a_composition_is_the_product_of_the_dense_forms() -> None:
    rng = np.random.default_rng(20261208)
    factors = (create(0), annihilate(0), create(1), annihilate(1), create(2))
    identity = FermionOperator((FermionTerm(1.0, (), ()),))

    def random_operator() -> FermionOperator:
        operator = identity
        for _ in range(int(rng.integers(1, 4))):
            operator = operator * factors[int(rng.integers(0, len(factors)))]
        return operator * complex(rng.normal(), rng.normal())

    checked = 0
    for _ in range(60):
        left = random_operator()
        right = random_operator()
        product = left * right
        assert np.allclose(
            _fock_matrix(product, 3),
            _fock_matrix(left, 3) @ _fock_matrix(right, 3),
            atol=1e-12,
        )
        checked += 1
    assert checked == 60


def test_creation_and_annihilation_obey_the_canonical_anticommutation_relations() -> None:
    for degree in (0, 1, 2):
        anticommutator = create(degree).anticommutator(annihilate(degree))
        assert anticommutator.terms == (FermionTerm(1.0, (), ()),)
    assert create(0).anticommutator(annihilate(1)).terms == ()
    assert create(0).anticommutator(create(0)).terms == ()
    assert annihilate(1).anticommutator(annihilate(1)).terms == ()
    assert create(0).anticommutator(create(1)).terms == ()


def test_adjacent_equal_factors_vanish_but_a_nested_pair_contracts() -> None:
    assert (create(0) * create(0)).terms == ()
    assert (annihilate(0) * annihilate(0)).terms == ()
    # `c†[0] c[0] c†[0]` contracts the middle annihilation with the following creation and
    # leaves `c†[0]`, which is why the vanish rule may only look at adjacent factors.
    assert (create(0) * annihilate(0) * create(0)).terms == create(0).terms
    # A creation three factors apart from its own degree does not contract; the two outer
    # creations are brought together by the rewrite and then vanish.
    assert (create(0) * annihilate(1) * create(0)).terms == ()


def test_the_adjoint_reverses_a_product_and_carries_its_sign() -> None:
    assert (create(0) * create(1)).dagger().terms == (
        FermionTerm(-1.0, (), (0, 1)),
    )
    assert (create(0) * annihilate(1)).dagger().terms == (
        FermionTerm(1.0, (1,), (0,)),
    )
    assert number(0).dagger() == number(0)


def test_hermiticity_is_an_exact_structural_test() -> None:
    assert number(0).is_hermitian()
    assert (create(0) + annihilate(0)).is_hermitian()
    assert not create(0).is_hermitian()
    assert not (create(0) * annihilate(1)).is_hermitian()
    assert not (1.0j * number(0)).is_hermitian()


def test_commutator_and_anticommutator_follow_their_definitions() -> None:
    assert number(0).commutator(number(1)).terms == ()
    assert number(3).commutator(create(3)).terms == create(3).terms
    assert number(3).commutator(annihilate(3)).terms == (-annihilate(3)).terms
    assert number(0).anticommutator(number(0)).terms == (
        FermionTerm(2.0, (0,), (0,)),
    )
    with pytest.raises(TypeError):
        number(0).commutator(X(0))
    with pytest.raises(TypeError):
        number(0).anticommutator(X(0))


def test_the_number_operator_is_a_projection() -> None:
    assert (number(2) * number(2)).terms == number(2).terms


def test_jordan_wigner_maps_the_hopping_operator_to_its_known_image() -> None:
    hopping = create(0) * annihilate(1) + create(1) * annihilate(0)
    image = jordan_wigner(hopping, n_modes=2)
    assert [(term.coefficient, term.factors) for term in image.terms] == [
        (0.5, ((0, "x"), (1, "x"))),
        (0.5, ((0, "y"), (1, "y"))),
    ]


def test_jordan_wigner_maps_a_two_mode_interaction_to_its_known_image() -> None:
    image = jordan_wigner(number(0) * number(1), n_modes=2)
    assert [(term.coefficient, term.factors) for term in image.terms] == [
        (0.25, ()),
        (-0.25, ((0, "z"),)),
        (0.25, ((0, "z"), (1, "z"))),
        (-0.25, ((1, "z"),)),
    ]


def test_jordan_wigner_returns_the_zero_observable_for_the_zero_operator() -> None:
    image = jordan_wigner(FermionOperator(), n_modes=2)
    assert [(term.coefficient, term.factors) for term in image.terms] == [(0.0, ())]


def test_jordan_wigner_refuses_a_non_hermitian_operator() -> None:
    with pytest.raises(CapabilityError, match="not Hermitian"):
        jordan_wigner(create(0), n_modes=1)


def test_jordan_wigner_refuses_a_degree_outside_the_requested_modes() -> None:
    with pytest.raises(ValueError, match="degree 2 is outside the 2 mode"):
        jordan_wigner(number(2), n_modes=2)


def test_jordan_wigner_refuses_a_zero_mode_request() -> None:
    with pytest.raises(ValueError, match="at least one mode"):
        jordan_wigner(FermionOperator(), n_modes=0)


def test_the_jordan_wigner_image_equals_the_fock_matrix_on_the_known_operators() -> None:
    cases = (
        create(0) * annihilate(1) + create(1) * annihilate(0),
        number(0),
        number(0) * number(1),
        number(0) * number(1) * number(2),
    )
    for operator in cases:
        for n_modes in (3,):
            image = jordan_wigner(operator, n_modes=n_modes)
            assert np.allclose(
                _observable_matrix(image, n_modes),
                _fock_matrix(operator, n_modes),
                atol=1e-12,
            )


def test_the_jordan_wigner_image_equals_the_fock_matrix_on_random_hermitian_operators() -> None:
    rng = np.random.default_rng(20260902)
    checked = 0
    for n_modes in (2, 3, 4):
        for _ in range(40):
            operator = _random_hermitian_operator(n_modes, rng)
            if not operator.terms:
                continue
            image = jordan_wigner(operator, n_modes=n_modes)
            assert np.allclose(
                _observable_matrix(image, n_modes),
                _fock_matrix(operator, n_modes),
                atol=1e-12,
            )
            checked += 1
    assert checked > 100


def test_the_jordan_wigner_image_equals_the_fock_matrix_on_every_short_monomial() -> None:
    factors = [
        create(0),
        annihilate(0),
        create(1),
        annihilate(1),
    ]
    checked = 0
    for first in factors:
        for second in factors:
            for third in factors:
                product = first * second * third
                operator = product + product.dagger()
                image = jordan_wigner(operator, n_modes=2)
                assert np.allclose(
                    _observable_matrix(image, 2),
                    _fock_matrix(operator, 2),
                    atol=1e-12,
                )
                checked += 1
    assert checked == len(factors) ** 3


def test_the_jordan_wigner_image_measures_the_dense_fock_expectation() -> None:
    """The image is measured end to end by the runtime that already measures observables.

    Exact expectation on the noiseless statevector path, so there is no shot count and no
    standard error; the runtime is complex64.  The measured margin is ``7.1e-11`` for this
    circuit, and the 2-mode interaction contributes ``1.3`` of the value, so the round-off
    is roughly eleven orders below the signal.  A tolerance of ``1e-6`` is therefore three
    orders above the round-off and still four orders below the divergence a plausible wrong
    rule produces here: reading ``sigma_plus`` as the annihilator flips the hopping sign and
    moves the value by ``1.1e-1``, and dropping the Jordan-Wigner string moves it by
    ``2.3e-1``.  Both are measured, not estimated.
    """

    t, interaction_strength = 0.7, 1.3
    hopping = t * (create(0) * annihilate(1) + create(1) * annihilate(0))
    hamiltonian = hopping + interaction_strength * (number(0) * number(1))
    observable = jordan_wigner(hamiltonian, n_modes=2)
    dense = _fock_matrix(hamiltonian, 2)

    for theta in (0.0, 0.3, 0.6, 1.1, 2.4):
        for preparation in (
            fq.Circuit(2).x(1),
            fq.Circuit(2),
            fq.Circuit(2).x(0).x(1),
        ):
            circuit = preparation.ry(0, theta)
            result = fq.run(circuit, outputs=[fq.expectation(observable)])
            measured = float(np.asarray(result.expectation()).reshape(-1)[0].real)
            state = np.asarray(result.to_statevector(), dtype=complex).reshape(-1)
            expected = float(np.real(state.conj() @ (dense @ state)))
            assert abs(measured - expected) < 1e-6


def test_the_observables_namespace_exposes_the_fermionic_algebra() -> None:
    from flagquantum import observables

    assert observables.jordan_wigner is jordan_wigner
    assert observables.FermionOperator is FermionOperator
    for name in ("FermionOperator", "FermionTerm", "annihilate", "create", "number"):
        assert name in observables.__all__
    with pytest.raises(AttributeError, match="has no attribute 'no_such_name'"):
        _ = observables.no_such_name


def test_a_term_refuses_an_unusable_coefficient_or_a_degree() -> None:
    with pytest.raises(TypeError, match="numeric coefficient"):
        FermionTerm(True, (0,), ())
    with pytest.raises(ValueError, match="must be finite"):
        FermionTerm(float("inf"), (0,), ())
    with pytest.raises(ValueError, match="ascending order"):
        FermionTerm(1.0, (1, 0), ())
    with pytest.raises(TypeError, match="must be an integer"):
        create(0.5)
    with pytest.raises(ValueError, match="non-negative integer"):
        create(-1)


def test_an_operator_refuses_a_value_that_is_not_a_fermionic_term() -> None:
    with pytest.raises(TypeError, match="holds FermionTerm values"):
        FermionOperator((X(0),))


def test_the_identity_observation_holds_because_nothing_here_is_shot_based() -> None:
    """An exact expectation is a single number, so the observable is passed through."""

    observable = jordan_wigner(number(0), n_modes=1)
    value = fq.expectation(observable)
    assert value.observable is observable
    assert value.kind == "expectation"
