"""What the feature-selection unit builds, checked against an enumerated reference.

Every one of the ``2**m`` subsets of an instance is enumerated, the objective is evaluated
on it from the formula the module documents -- ``-sum r_i + sum s_ij + penalty *
(|S| - n_selected)**2`` -- and that value is compared against the module's own energy for
the same subset. The coefficients of a quadratic binary objective are recoverable from its
values on every subset, so an enumeration pins each coefficient rather than only the
total: no error in one term can hide behind another, which is what makes the presence of
the size penalty a testable property rather than a reading of the code.

The Ising side has a reference path of its own. The objective's binary variable is
``x_i = (1 + s_i) / 2``, so a selected feature sits at spin ``+1``: a ``Z`` product is
``+1`` where a wire's bit is 1 and ``-1`` where it is 0, and a term contributes its
coefficient times that product over its wires. An identity term contributes its
coefficient outright -- it is the constant, and the wire it declares carries the
register's width rather than an operator -- so the Hamiltonian's value at a configuration
is computable from the term list without going through the QUBO form, which is what makes
the mapping's own claim checkable.

Every expected value here was measured by running the construction it checks.
"""

from __future__ import annotations

import itertools

import pytest
import torch

from flagquantum.algorithms.core import Hamiltonian
from flagquantum.algorithms.feature_selection import (
    FeatureSelectionProblem,
    feature_selection_qubo,
)
from flagquantum.algorithms.qubo import (
    QuboProblem,
    ising_to_qubo,
    qubo_energy,
)

pytestmark = pytest.mark.unit

# Instance A: four features, a redundancy for every pair, two of them wanted. Its
# objective is -sum_{i in S} r_i + sum_{i < j in S} s_ij + 3 * (|S| - 2)**2, so the empty
# subset scores 12.0 (the size penalty's offset alone), the pair {0, 2} scores -3.5 and is
# the cheapest of the sixteen, and every other subset is measured below.
_RELEVANCE = [1.5, 0.4, 2.25, 0.9]
_REDUNDANCY = [
    [0.0, 0.8, 0.25, 0.6],
    [0.8, 0.0, 0.45, 0.1],
    [0.25, 0.45, 0.0, 0.95],
    [0.6, 0.1, 0.95, 0.0],
]
_N_SELECTED = 2
_PENALTY = 3.0

# Instance B: five features whose relevance falls with the index and no redundancy at
# all. At a penalty that dominates the scores, the cheapest subset is the two most
# relevant features; at a penalty small enough not to bind, it is all five.
_FALLING_RELEVANCE = [1.0, 0.9, 0.8, 0.7, 0.6]
_DOMINATING_PENALTY = 5.0
_SMALL_PENALTY = 0.05


def _assignments(n_features: int) -> list[tuple[int, ...]]:
    """Every binary assignment, one entry per feature in feature order."""
    return [tuple(bits) for bits in itertools.product((0, 1), repeat=n_features)]


def _selected(assignment: tuple[int, ...]) -> list[int]:
    """The feature indices an assignment selects."""
    return [index for index, bit in enumerate(assignment) if bit]


def _objective(
    selected: list[int],
    *,
    relevance: list[float],
    redundancy: list[list[float]],
    n_selected: int,
    penalty: float,
) -> tuple[float, float, float]:
    """The documented objective over a subset, in its three terms and unexpanded.

    The sum over the subset's pairs is written out here from the formula; it never reads
    the module's quadratic coefficient map, which is the claim under test.
    """
    choice = -sum(relevance[index] for index in selected)
    pairs = sum(
        redundancy[first][second]
        for first in selected
        for second in selected
        if first < second
    )
    size = penalty * (len(selected) - n_selected) ** 2
    return choice, pairs, size


def _problem() -> FeatureSelectionProblem:
    """Instance A, built through the unit under test."""
    return feature_selection_qubo(
        torch.tensor(_RELEVANCE, dtype=torch.float64),
        n_selected=_N_SELECTED,
        penalty=_PENALTY,
        redundancy=torch.tensor(_REDUNDANCY, dtype=torch.float64),
    )


def _falling_problem(*, n_selected: int, penalty: float) -> FeatureSelectionProblem:
    """Instance B at a target size and a weight, built through the unit under test."""
    return feature_selection_qubo(
        torch.tensor(_FALLING_RELEVANCE, dtype=torch.float64),
        n_selected=n_selected,
        penalty=penalty,
    )


def _cheapest(problem: FeatureSelectionProblem, n_features: int) -> tuple[int, ...]:
    """The assignment of least objective value, by enumerating every subset."""
    values = {
        assignment: problem.energy(assignment)
        for assignment in _assignments(n_features)
    }
    return min(values, key=values.__getitem__)


def _hamiltonian_value(hamiltonian: Hamiltonian, assignment: tuple[int, ...]) -> float:
    """The Ising value of an assignment's spin configuration, read off the term list.

    The substitution ``x_i = (1 + s_i) / 2`` puts a selected feature at spin ``+1`` and an
    unselected one at spin ``-1``, so a ``Z`` product is ``+1`` on a wire whose bit is 1
    and ``-1`` on a wire whose bit is 0, and the term is weighted by that product over its
    wires. An identity term is the constant: it is **not** weighted by its wires. The
    declared wire an identity term carries is where the register's width survives the
    round trip, and reading it as an operator would negate the constant. The value is the
    sum of the weighted terms, which is what a spin Hamiltonian means and not what the
    QUBO form computes.
    """
    total = 0.0
    for term in hamiltonian.terms:
        if term.pauli == "I":
            total += float(term.coefficient)
            continue
        product = 1.0
        for wire in term.wires:
            product *= 1.0 if assignment[wire] else -1.0
        total += float(term.coefficient) * product
    return total


def test_the_energy_matches_the_documented_objective_on_every_subset() -> None:
    """All sixteen subsets of instance A, from the formula, against the module."""
    problem = _problem()
    for assignment in _assignments(4):
        choice, pairs, size = _objective(
            _selected(assignment),
            relevance=_RELEVANCE,
            redundancy=_REDUNDANCY,
            n_selected=_N_SELECTED,
            penalty=_PENALTY,
        )
        assert problem.energy(assignment) == pytest.approx(choice + pairs + size)


def test_the_measured_objective_values_of_instance_a() -> None:
    """A worked instance, so the objective is pinned at named subsets and not only in
    the aggregate. Measured: the empty subset is the size penalty's offset, the pair
    {0, 2} is the cheapest subset of all, and the full subset is dear.
    """
    problem = _problem()
    assert problem.energy((0, 0, 0, 0)) == pytest.approx(12.0)
    assert problem.energy((1, 0, 0, 0)) == pytest.approx(1.5)
    assert problem.energy((0, 0, 0, 1)) == pytest.approx(2.1)
    assert problem.energy((1, 0, 1, 0)) == pytest.approx(-3.5)
    assert problem.energy((0, 1, 1, 0)) == pytest.approx(-2.2)
    assert problem.energy((1, 1, 1, 1)) == pytest.approx(10.1)
    assert _cheapest(problem, 4) == (1, 0, 1, 0)


def test_the_coefficient_maps_state_the_size_penalty_expansion() -> None:
    """The three coefficient maps against the expansion the module's docstring states.

    This check is the module's formula transcribed, so it localizes a wrong term; the
    enumeration above is what decides whether the formula is right.
    """
    problem = _problem()
    for index, score in enumerate(_RELEVANCE):
        assert problem.qubo.linear[index] == pytest.approx(
            -score + _PENALTY * (1 - 2 * _N_SELECTED)
        )
    for first in range(4):
        for second in range(first + 1, 4):
            assert problem.qubo.quadratic[first, second] == pytest.approx(
                _REDUNDANCY[first][second] + 2 * _PENALTY
            )
    assert problem.qubo.offset == pytest.approx(_PENALTY * _N_SELECTED**2)


def test_the_size_penalty_is_the_difference_the_energy_carries() -> None:
    """The energy less the two score terms is the size term, and it is zero only on target.

    A build that dropped the penalty would leave the difference at zero for every subset,
    including the ones whose size is not the target, so the assertion on the off-target
    subsets is what makes the penalty's presence a property of the energy rather than of
    the coefficient maps alone.
    """
    problem = _problem()
    for assignment in _assignments(4):
        selected = _selected(assignment)
        choice, pairs, size = _objective(
            selected,
            relevance=_RELEVANCE,
            redundancy=_REDUNDANCY,
            n_selected=_N_SELECTED,
            penalty=_PENALTY,
        )
        # ``difference`` is read off the module's own energy; it vanishes on the target
        # size and is positive off it, which is what the penalty's presence means here.
        difference = problem.energy(assignment) - (choice + pairs)
        assert difference == pytest.approx(size)
        if len(selected) == _N_SELECTED:
            assert difference == pytest.approx(0.0, abs=1e-9)
        else:
            assert difference > 1e-9


def test_a_problem_without_redundancy_keeps_the_size_penalty_in_its_pairs() -> None:
    """Every pair is a pair of features, so the squared size term reaches all of them."""
    problem = feature_selection_qubo(
        torch.tensor(_RELEVANCE, dtype=torch.float64),
        n_selected=_N_SELECTED,
        penalty=_PENALTY,
    )
    assert set(problem.qubo.quadratic) == {
        (0, 1),
        (0, 2),
        (0, 3),
        (1, 2),
        (1, 3),
        (2, 3),
    }
    assert all(
        coefficient == pytest.approx(2 * _PENALTY)
        for coefficient in problem.qubo.quadratic.values()
    )
    for assignment in _assignments(4):
        choice, pairs, size = _objective(
            _selected(assignment),
            relevance=_RELEVANCE,
            redundancy=[[0.0] * 4 for _ in range(4)],
            n_selected=_N_SELECTED,
            penalty=_PENALTY,
        )
        assert problem.energy(assignment) == pytest.approx(choice + pairs + size)


def test_the_cheapest_subset_carries_the_target_size_and_the_top_scores() -> None:
    """At a penalty that dominates the scores, the target size is what the objective buys."""
    problem = _falling_problem(n_selected=2, penalty=_DOMINATING_PENALTY)
    assert _cheapest(problem, 5) == (1, 1, 0, 0, 0)
    assert problem.energy((1, 1, 0, 0, 0)) == pytest.approx(-1.9)

    three = _falling_problem(n_selected=3, penalty=_DOMINATING_PENALTY)
    assert _cheapest(three, 5) == (1, 1, 1, 0, 0)
    assert three.energy((1, 1, 1, 0, 0)) == pytest.approx(-2.7)


def test_a_penalty_that_does_not_bind_leaves_the_scores_to_choose_the_size() -> None:
    """The same instance and the same scores at a weight small enough not to bind.

    The weight is the caller's and the module states no rule for it, so what is checked
    here is the objective's own arithmetic: the size term is what moves the cheapest
    subset off the target size, and at this weight the cheapest subset is every feature.
    """
    problem = _falling_problem(n_selected=2, penalty=_SMALL_PENALTY)
    cheapest = _cheapest(problem, 5)
    assert sum(cheapest) == 5
    # Every relevance is positive, so the score terms alone want every feature: -4.0,
    # against the size term's 0.05 * (5 - 2)**2.
    assert problem.energy(cheapest) == pytest.approx(-3.55)


def test_the_target_size_at_both_of_its_boundaries() -> None:
    """A target of no feature and a target of every feature are both well defined."""
    none_wanted = _falling_problem(n_selected=0, penalty=_DOMINATING_PENALTY)
    assert _cheapest(none_wanted, 5) == (0, 0, 0, 0, 0)

    all_wanted = _falling_problem(n_selected=5, penalty=_DOMINATING_PENALTY)
    assert _cheapest(all_wanted, 5) == (1, 1, 1, 1, 1)
    assert all_wanted.energy((1, 1, 1, 1, 1)) == pytest.approx(-4.0)


def test_a_single_feature_problem_carries_no_pair_coefficient() -> None:
    """With one feature there is no pair, so the quadratic map is empty."""
    problem = feature_selection_qubo(
        torch.tensor([1.0], dtype=torch.float64), n_selected=0, penalty=2.0
    )
    assert problem.qubo.n_variables == 1
    assert problem.qubo.linear == {0: 1.0}
    assert not problem.qubo.quadratic
    assert problem.qubo.offset == pytest.approx(0.0)
    assert problem.energy((0,)) == pytest.approx(0.0)
    assert problem.energy((1,)) == pytest.approx(1.0)


def test_the_energy_carries_the_offset_of_the_size_penalty() -> None:
    """The empty subset's objective is the offset, which is the penalty at ``|S| = 0``."""
    problem = _problem()
    assert problem.qubo.offset == pytest.approx(_PENALTY * _N_SELECTED**2)
    assert problem.energy((0, 0, 0, 0)) == pytest.approx(problem.qubo.offset)
    assert problem.energy((0, 0, 0, 0)) == qubo_energy(problem.qubo, (0, 0, 0, 0))


def test_the_energy_is_the_qubo_energy_of_the_carried_problem() -> None:
    """The evaluation is ``qubo_energy`` on the carried problem, over every subset."""
    problem = _problem()
    for assignment in _assignments(4):
        assert problem.energy(assignment) == qubo_energy(problem.qubo, assignment)


def test_the_ising_form_matches_the_objective_at_every_configuration() -> None:
    """The spin value of every configuration, from the term list, against the energy.

    This path never goes through the QUBO form: it reads the Hamiltonian's Pauli products
    and evaluates them on spins, which is the claim the module's ``to_ising`` makes.
    """
    problem = _problem()
    hamiltonian = problem.to_ising()
    for assignment in _assignments(4):
        assert _hamiltonian_value(hamiltonian, assignment) == pytest.approx(
            problem.energy(assignment)
        ), assignment


def test_the_ising_form_round_trips_through_the_packages_own_mapping() -> None:
    """``ising_to_qubo(problem.to_ising())`` scores every subset the problem's way."""
    problem = _problem()
    recovered = ising_to_qubo(problem.to_ising())
    assert recovered.n_variables == problem.qubo.n_variables
    for assignment in _assignments(4):
        assert qubo_energy(recovered, assignment) == pytest.approx(
            problem.energy(assignment)
        )


def test_the_ising_form_carries_a_term_per_feature_a_pair_and_the_constant() -> None:
    """Measured on instance A: four single-wire terms, six pair terms, one identity."""
    hamiltonian = _problem().to_ising()
    paulis = [term.pauli for term in hamiltonian.terms]
    assert sorted(paulis) == ["I"] + ["Z"] * 4 + ["ZZ"] * 6
    assert len({term.wires for term in hamiltonian.terms if term.pauli == "ZZ"}) == 6


def test_the_energy_refuses_an_assignment_that_is_not_one_bit_per_feature() -> None:
    """The assignment is read by the QUBO form's own energy evaluation."""
    problem = _problem()
    with pytest.raises(ValueError, match="assignment has 3 values"):
        problem.energy((1, 0, 1))
    with pytest.raises(ValueError, match="must be 0 or 1"):
        problem.energy((1, 0, 1, 2))
    with pytest.raises(ValueError, match="must be 0 or 1"):
        problem.energy((1, 0, 1, 0.5))


def test_the_relevance_is_validated() -> None:
    """Every refusal is matched on this module's own wording."""
    good = torch.tensor(_RELEVANCE, dtype=torch.float64)
    with pytest.raises(ValueError, match="must be a torch.Tensor"):
        feature_selection_qubo([1.0, 2.0], n_selected=1, penalty=1.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="one-dimensional"):
        feature_selection_qubo(
            torch.ones(2, 2, dtype=torch.float64), n_selected=1, penalty=1.0
        )
    with pytest.raises(ValueError, match="real floating-point tensor"):
        feature_selection_qubo(
            torch.ones(2, dtype=torch.int64), n_selected=1, penalty=1.0
        )
    with pytest.raises(ValueError, match="at least one feature"):
        feature_selection_qubo(
            torch.ones(0, dtype=torch.float64), n_selected=0, penalty=1.0
        )
    with pytest.raises(ValueError, match="every relevance must be finite"):
        feature_selection_qubo(
            torch.tensor([1.0, float("nan")]), n_selected=1, penalty=1.0
        )
    with pytest.raises(ValueError, match="every relevance must be finite"):
        feature_selection_qubo(
            good.new_tensor([1.0, float("inf")]), n_selected=1, penalty=1.0
        )


def test_the_redundancy_is_validated() -> None:
    """A matrix that does not describe one score per pair is refused."""
    relevance = torch.tensor(_RELEVANCE, dtype=torch.float64)
    with pytest.raises(ValueError, match="must be a torch.Tensor or None"):
        feature_selection_qubo(
            relevance, n_selected=1, penalty=1.0, redundancy=[[0.0]]  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match=r"shape \(4, 4\)"):
        feature_selection_qubo(
            relevance,
            n_selected=1,
            penalty=1.0,
            redundancy=torch.zeros(3, 3, dtype=torch.float64),
        )
    with pytest.raises(ValueError, match="real floating-point tensor"):
        feature_selection_qubo(
            relevance,
            n_selected=1,
            penalty=1.0,
            redundancy=torch.zeros(4, 4, dtype=torch.int64),
        )
    with pytest.raises(ValueError, match="outside the diagonal must be finite"):
        feature_selection_qubo(
            relevance,
            n_selected=1,
            penalty=1.0,
            redundancy=torch.tensor(
                [
                    [0.0, 0.8, 0.25, 0.6],
                    [0.8, 0.0, 0.45, 0.1],
                    [0.25, 0.45, 0.0, float("inf")],
                    [0.6, 0.1, float("inf"), 0.0],
                ]
            ),
        )
    with pytest.raises(ValueError, match=r"0.5 at \(0, 1\) against 0.25 at \(1, 0\)"):
        feature_selection_qubo(
            torch.ones(2, dtype=torch.float64),
            n_selected=1,
            penalty=1.0,
            redundancy=torch.tensor([[0.0, 0.5], [0.25, 0.0]]),
        )


def test_the_diagonal_of_a_redundancy_matrix_is_not_read() -> None:
    """A pair is two features, so the diagonal carries no score and no requirement.

    Measured: a diagonal of ``nan``, of ``inf`` or of any number is accepted, and the
    coefficient maps and every subset's objective are the ones a zero diagonal gives. The
    off-diagonal entries are still held to finiteness and to symmetry, which the test
    above checks.
    """
    for diagonal in (float("nan"), float("inf"), 7.5):
        rows = [
            [
                diagonal if first == second else _REDUNDANCY[first][second]
                for second in range(4)
            ]
            for first in range(4)
        ]
        problem = feature_selection_qubo(
            torch.tensor(_RELEVANCE, dtype=torch.float64),
            n_selected=_N_SELECTED,
            penalty=_PENALTY,
            redundancy=torch.tensor(rows, dtype=torch.float64),
        )
        clean = _problem()
        assert problem.qubo.quadratic == clean.qubo.quadratic
        assert problem.qubo.linear == clean.qubo.linear
        assert problem.qubo.offset == clean.qubo.offset
        for assignment in _assignments(4):
            assert problem.energy(assignment) == clean.energy(assignment)


def test_the_target_size_is_validated() -> None:
    """A size of a subset is a whole count inside the feature count."""
    with pytest.raises(ValueError, match="must be an integer"):
        _falling_problem(n_selected=True, penalty=_DOMINATING_PENALTY)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="must be an integer"):
        _falling_problem(n_selected=2.0, penalty=_DOMINATING_PENALTY)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=r"size in range\(6\)"):
        _falling_problem(n_selected=6, penalty=_DOMINATING_PENALTY)
    with pytest.raises(ValueError, match=r"size in range\(6\)"):
        _falling_problem(n_selected=-1, penalty=_DOMINATING_PENALTY)


def test_the_penalty_weight_is_validated() -> None:
    """A weight prices the size term, so it must be a positive finite number."""
    with pytest.raises(ValueError, match="must be a real number"):
        _falling_problem(n_selected=2, penalty=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="must be a real number"):
        _falling_problem(n_selected=2, penalty="3")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="positive and finite"):
        _falling_problem(n_selected=2, penalty=0.0)
    with pytest.raises(ValueError, match="positive and finite"):
        _falling_problem(n_selected=2, penalty=-1.0)
    with pytest.raises(ValueError, match="positive and finite"):
        _falling_problem(n_selected=2, penalty=float("nan"))
    with pytest.raises(ValueError, match="positive and finite"):
        _falling_problem(n_selected=2, penalty=float("inf"))


def test_the_instance_validates_its_own_fields() -> None:
    """The class reads its fields the way the builder reads its arguments."""
    problem = _problem()
    with pytest.raises(ValueError, match="carried as a QuboProblem"):
        FeatureSelectionProblem(  # type: ignore[arg-type]
            qubo=torch.ones(2), n_selected=1, penalty=1.0
        )
    with pytest.raises(ValueError, match=r"size in range\(5\)"):
        FeatureSelectionProblem(qubo=problem.qubo, n_selected=5, penalty=1.0)
    with pytest.raises(ValueError, match="positive and finite"):
        FeatureSelectionProblem(qubo=problem.qubo, n_selected=1, penalty=0.0)


def test_the_instance_is_keyword_only_and_that_is_what_it_protects() -> None:
    """The spelling is keyword-only, and the two numbers are read by the class, not by it.

    Measured on instance A: a transposed spelling whose weight is an integer no larger
    than the feature count and whose target size is positive constructs, and the two
    fields then hold each other's values; the same transposition with a weight that is a
    fraction, with a weight above the feature count, or with a target size of zero is
    refused, each by the check that reads that field.
    """
    problem = _problem()
    with pytest.raises(TypeError):
        FeatureSelectionProblem(problem.qubo, _N_SELECTED, _PENALTY)  # type: ignore[misc]

    transposed = FeatureSelectionProblem(qubo=problem.qubo, n_selected=3, penalty=2.0)
    assert transposed.n_selected == 3
    assert transposed.penalty == pytest.approx(2.0)

    for fields, reads in (
        ({"n_selected": 3.0, "penalty": 2.0}, "must be an integer"),
        ({"n_selected": 100, "penalty": 2.0}, r"size in range\(5\)"),
        ({"n_selected": 3, "penalty": 0.0}, "positive and finite"),
    ):
        with pytest.raises(ValueError, match=reads):
            FeatureSelectionProblem(qubo=problem.qubo, **fields)  # type: ignore[arg-type]


def test_a_valid_problem_still_constructs_through_the_class() -> None:
    """The refusals above are not the class refusing everything."""
    problem = _problem()
    rebuilt = FeatureSelectionProblem(
        qubo=QuboProblem(
            n_variables=4,
            linear=dict(problem.qubo.linear),
            quadratic=dict(problem.qubo.quadratic),
            offset=problem.qubo.offset,
        ),
        n_selected=_N_SELECTED,
        penalty=_PENALTY,
    )
    assert rebuilt == problem
