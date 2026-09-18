from __future__ import annotations

import itertools

import pytest

from flagquantum.algorithms.qubo import (
    QuboProblem,
    ising_to_qubo,
    max_cut_qubo,
    qubo_energy,
    qubo_to_ising,
)

pytestmark = pytest.mark.unit


def _assignments(n_variables: int) -> list[tuple[int, ...]]:
    """Every binary assignment, most significant variable first."""
    return [tuple(bits) for bits in itertools.product((0, 1), repeat=n_variables)]


def test_qubo_energy_of_a_known_assignment() -> None:
    """x0 = 1, x1 = 0 on -x0 - 2*x1 + 3*x0*x1 gives -1."""
    problem = QuboProblem(
        n_variables=2, linear={0: -1.0, 1: -2.0}, quadratic={(0, 1): 3.0}
    )
    assert qubo_energy(problem, (1, 0)) == pytest.approx(-1.0)


def test_qubo_energy_includes_the_offset() -> None:
    """The constant term is part of the objective."""
    problem = QuboProblem(n_variables=1, linear={0: 1.0}, quadratic={}, offset=2.5)
    assert qubo_energy(problem, (0,)) == pytest.approx(2.5)
    assert qubo_energy(problem, (1,)) == pytest.approx(3.5)


def test_qubo_energy_validates_the_assignment() -> None:
    """A wrong length and a non-binary value are both refused."""
    problem = QuboProblem(n_variables=2, linear={0: 1.0}, quadratic={})
    with pytest.raises(ValueError):
        qubo_energy(problem, (1,))
    with pytest.raises(ValueError):
        qubo_energy(problem, (1, 2))


def test_round_trip_preserves_energy_on_every_assignment() -> None:
    """ising_to_qubo(qubo_to_ising(p)) agrees with p on all of p's assignments."""
    problem = QuboProblem(
        n_variables=3,
        linear={0: -1.0, 1: 0.5, 2: 2.0},
        quadratic={(0, 1): 3.0, (1, 2): -1.5, (0, 2): -0.75},
    )
    recovered = ising_to_qubo(qubo_to_ising(problem))
    assert recovered.n_variables == problem.n_variables
    assert recovered.offset == pytest.approx(0.0)
    for assignment in _assignments(3):
        assert qubo_energy(recovered, assignment) == pytest.approx(
            qubo_energy(problem, assignment)
        )


def test_round_trip_is_exact_for_a_hamiltonian_that_is_not_a_qubo() -> None:
    """A general Z-basis Hamiltonian survives the round trip through QUBO form."""
    from flagquantum.algorithms.core import Hamiltonian, pauli_term

    hamiltonian = Hamiltonian(
        [
            pauli_term(0.7, "I", (0,)),
            pauli_term(1.3, "Z", (0,)),
            pauli_term(-0.4, "Z", (1,)),
            pauli_term(0.9, "ZZ", (0, 1)),
        ]
    )
    recovered = ising_to_qubo(hamiltonian)

    def ising_value(spins: tuple[int, int]) -> float:
        return 0.7 + 1.3 * spins[0] - 0.4 * spins[1] + 0.9 * spins[0] * spins[1]

    for spins in itertools.product((-1, 1), repeat=2):
        assignment = tuple((spin + 1) // 2 for spin in spins)
        assert qubo_energy(recovered, assignment) == pytest.approx(ising_value(spins))


def test_every_variable_carries_its_own_term() -> None:
    """A variable named only by a pair still gets the Z term the substitution requires."""
    problem = QuboProblem(n_variables=2, linear={0: 1.0}, quadratic={(0, 1): 2.0})
    hamiltonian = qubo_to_ising(problem)
    kinds = sorted(term.pauli for term in hamiltonian.terms)
    assert kinds == ["I", "Z", "Z", "ZZ"]
    constant = next(term for term in hamiltonian.terms if term.pauli == "I")
    assert float(constant.coefficient) == pytest.approx(1.0)
    weights = {
        term.wires[0]: float(term.coefficient)
        for term in hamiltonian.terms
        if term.pauli == "Z"
    }
    assert weights == {0: pytest.approx(1.0), 1: pytest.approx(0.5)}


def test_a_variable_declared_only_in_a_pair_round_trips() -> None:
    """The decisive case: x1 appears in no linear coefficient, so x0 alone drives it."""
    problem = QuboProblem(n_variables=2, linear={0: 1.0}, quadratic={(0, 1): 2.0})
    recovered = ising_to_qubo(qubo_to_ising(problem))
    for assignment in _assignments(2):
        assert qubo_energy(recovered, assignment) == pytest.approx(
            qubo_energy(problem, assignment)
        )


def test_a_problem_with_no_coefficients_still_maps() -> None:
    """The identity term keeps the Hamiltonian constructible when there is nothing else."""
    problem = QuboProblem(n_variables=2, linear={}, quadratic={})
    hamiltonian = qubo_to_ising(problem)
    assert len(hamiltonian.terms) == 1
    assert qubo_energy(ising_to_qubo(hamiltonian), (0, 0)) == pytest.approx(0.0)


def test_ising_to_qubo_rejects_a_non_z_term() -> None:
    """A Hamiltonian outside the Z basis has no QUBO form in this workflow."""
    from flagquantum.algorithms.core import Hamiltonian, pauli_term

    with pytest.raises(ValueError):
        ising_to_qubo(Hamiltonian([pauli_term(1.0, "X", (0,))]))


def test_a_bare_pair_hamiltonian_recovers_both_induced_linear_terms() -> None:
    """A pair contributes a single-wire Z to each endpoint, even with no Z term emitted."""
    from flagquantum.algorithms.core import Hamiltonian, pauli_term

    recovered = ising_to_qubo(Hamiltonian([pauli_term(2.0, "ZZ", (0, 1))]))
    assert recovered.linear == {0: pytest.approx(-4.0), 1: pytest.approx(-4.0)}
    assert recovered.quadratic == {(0, 1): pytest.approx(8.0)}
    assert recovered.offset == pytest.approx(2.0)


def test_a_pair_survives_when_its_endpoint_weight_cancels() -> None:
    """x1's linear coefficient is nonzero while its single-wire weight is exactly zero."""
    problem = QuboProblem(
        n_variables=2, linear={0: 1.0, 1: -1.0}, quadratic={(0, 1): 2.0}
    )
    recovered = ising_to_qubo(qubo_to_ising(problem))
    for assignment in _assignments(2):
        assert qubo_energy(recovered, assignment) == pytest.approx(
            qubo_energy(problem, assignment)
        )


def test_max_cut_matches_a_brute_force_optimum() -> None:
    """The MaxCut QUBO optimum equals the exhaustive maximum over all cuts."""
    problem = max_cut_qubo(((0, 1), (1, 2), (2, 0)), n_nodes=3)
    best = min(qubo_energy(problem, assignment) for assignment in _assignments(3))
    assert best == pytest.approx(-2.0)
