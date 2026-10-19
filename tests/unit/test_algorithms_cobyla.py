"""COBYLA: the subproblem, the constraint it restores, the refusals, and the price.

Three of these tests pin measurements rather than behaviour, and each of them is
the evidence for a sentence in the unit's own ``limitations``:

- :func:`test_a_price_that_is_too_low_returns_an_infeasible_point_that_scores_better`
  measures the sentence that says the penalty is a schedule and not a calibrated
  weight. It minimizes one quadratic under one affine constraint twice, changing
  only the starting price, and reports that the cheaper price stops at a point
  which violates the constraint by ``1.0`` and scores ``0.5`` where the feasible
  optimum scores ``2.0``, while the wider price stops at the feasible optimum
  itself. That is what "the exact penalty is not computed" costs a caller, and it
  is why ``feasible`` has to be read beside the value.
- :func:`test_a_price_that_is_frozen_abandons_feasibility_entirely` measures the
  same limitation at its extreme: with the ceiling set to the starting price the
  schedule cannot double, so the run keeps trading the constraint away and reports
  a residual above ``1.0`` on a problem whose feasible optimum is one step from
  where it started.
- :func:`test_a_pauli_energy_is_minimized_under_a_bound` measures the unit against
  a real objective: a two-qubit Pauli energy whose minimum is exactly ``-2`` at a
  zero angle, pushed off that minimum to ``0.9`` by one constraint on the angle.

Every other test here is about a promise the unit makes: the step it takes is the
minimizer of the linearized merit inside the box, the cost it reports is the
number of distinct points it read, the point it returns is a tensor this optimizer
owns, and each refusal happens before a value is stored beside a point.
"""

from __future__ import annotations

import math

import pytest
import torch

import flagquantum as fq
from flagquantum.algorithms import (
    COBYLA_ASSUMPTIONS,
    COBYLA_LIMITATIONS,
    CobylaOptimizer,
    CobylaResult,
)
from flagquantum.algorithms.cobyla import (
    _AffineModel,
    _largest_violation,
    _trust_region_step,
)
from flagquantum.errors import ValidationError

pytestmark = pytest.mark.unit


class _Counter:
    """A scalar function that records how often it was called."""

    def __init__(self, function):
        self._function = function
        self.calls = 0

    def __call__(self, parameters: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return self._function(parameters)


def _origin(count: int = 2) -> torch.Tensor:
    return torch.zeros(count, dtype=torch.float64)


def _offset_to(target: tuple[float, ...]):
    """A separable quadratic whose minimum is exactly ``target``."""

    def objective(parameters: torch.Tensor) -> torch.Tensor:
        shifts = torch.tensor(target, dtype=parameters.dtype)
        return ((parameters - shifts) ** 2).sum()

    return objective


def _energy(parameters: torch.Tensor) -> torch.Tensor:
    """The two-qubit Pauli energy the SPSA and Nelder-Mead examples minimize."""

    circuit = fq.Circuit(2, dtype=torch.complex128)
    circuit = circuit.ry(0, parameters[0]).ry(1, parameters[1]).cx(0, 1)
    outputs = fq.expectation(fq.Z(0) + fq.Z(1))
    return -fq.run(circuit, outputs=outputs).expectation().sum()


def _objective(gradient: tuple[float, float]) -> _AffineModel:
    return _AffineModel(value=0.0, gradient=gradient)


def _models(
    entries: tuple[tuple[float, float, float], ...],
) -> tuple[_AffineModel, ...]:
    """One model per ``(slope along x, slope along y, value at the base point)``."""

    return tuple(
        _AffineModel(value=entry[2], gradient=(entry[0], entry[1])) for entry in entries
    )


def _linearized_merit(
    objective: _AffineModel,
    models: tuple[_AffineModel, ...],
    step: tuple[float, ...],
    penalty: float,
) -> float:
    """The merit the models predict for one step: value plus priced violation."""

    return objective.at(step) + penalty * _largest_violation(
        [model.at(step) for model in models]
    )


def _grid(radius: float, divisions: int = 12) -> list[tuple[float, float]]:
    axis = [radius * (2 * index / divisions - 1) for index in range(divisions + 1)]
    return [(first, second) for first in axis for second in axis]


#: Objective gradients, constraint models, radii, and prices to solve the
#: subproblem over. Every base value is negative on purpose: a constraint starts
#: violated, so the row that carries the violation variable is the one exercised.
_SUBPROBLEM_CASES = (
    ((1.0, 0.0), (), 0.5, 1.0),
    ((0.3, -0.9), ((-1.0, 0.0, -0.2),), 0.4, 1.0),
    ((0.3, -0.9), ((-1.0, 0.0, -0.1), (0.0, -1.0, -0.1)), 0.25, 2.0),
    ((0.0, 0.0), ((1.0, 1.0, -0.3), (1.0, -1.0, 0.2)), 0.4, 1.0),
    ((0.7, 0.7), ((1.0, 1.0, 0.5),), 0.25, 0.5),
    ((0.7, 0.7), ((1.0, 1.0, -0.5), (-1.0, 0.0, 0.5)), 0.5, 3.0),
)


@pytest.mark.parametrize(
    ("gradient", "constraints", "radius", "penalty"), _SUBPROBLEM_CASES
)
def test_the_step_is_the_minimizer_of_the_linearized_merit(
    gradient: tuple[float, float],
    constraints: tuple[tuple[float, float, float], ...],
    radius: float,
    penalty: float,
) -> None:
    objective = _objective(gradient)
    models = _models(constraints)
    step = tuple(_trust_region_step(objective, models, radius=radius, penalty=penalty))
    assert all(abs(value) <= radius + 1e-12 for value in step)
    best = min(
        _linearized_merit(objective, models, candidate, penalty)
        for candidate in _grid(radius)
    )
    assert _linearized_merit(objective, models, step, penalty) <= best + 1e-9


@pytest.mark.parametrize(
    ("gradient", "constraints", "radius", "penalty"), _SUBPROBLEM_CASES
)
def test_no_small_move_from_the_step_improves_the_linearized_merit(
    gradient: tuple[float, float],
    constraints: tuple[tuple[float, float, float], ...],
    radius: float,
    penalty: float,
) -> None:
    objective = _objective(gradient)
    models = _models(constraints)
    step = tuple(_trust_region_step(objective, models, radius=radius, penalty=penalty))
    settled = _linearized_merit(objective, models, step, penalty)
    for index in range(2):
        for shift in (-1e-6, 1e-6):
            moved = list(step)
            moved[index] = min(radius, max(-radius, moved[index] + shift))
            if moved[index] == step[index]:
                continue
            assert (
                _linearized_merit(objective, models, tuple(moved), penalty)
                >= settled - 1e-9
            )


def test_the_step_leaves_an_unconstrained_box_at_the_far_corner() -> None:
    # With no constraint to price, the linearized objective is minimized at the
    # corner the gradient points away from, one coordinate at a time.
    step = _trust_region_step(_objective((2.0, -1.0)), (), radius=0.25, penalty=1.0)
    assert step == [-0.25, 0.25]


def test_the_step_trades_objective_against_a_violation_it_can_repair() -> None:
    objective = _objective((0.38942, 0.38942))
    models = _models(((1.0, 0.0, -0.1), (0.0, 1.0, -0.1)))
    step = _trust_region_step(objective, models, radius=0.5, penalty=1.0)
    # The step is exactly the length at which both constraints are satisfied, and
    # one unit further would cost more objective than the violation it removes.
    assert step == pytest.approx([0.1, 0.1], abs=1e-12)


def test_a_violation_the_box_cannot_repair_is_paid_for_and_reported() -> None:
    objective = _objective((0.0, 0.0))
    models = _models(((1.0, 0.0, -0.6),))
    step = _trust_region_step(objective, models, radius=0.5, penalty=1.0)
    # The best available step covers 0.5 of the 0.6 violation and leaves the rest
    # priced, which is what the subproblem is asked for.
    assert step == pytest.approx([0.5, 0.0], abs=1e-12)


def test_a_subproblem_with_no_constraints_and_no_gradient_steps_nowhere() -> None:
    step = _trust_region_step(
        _AffineModel(value=1.0, gradient=(0.0, 0.0)), (), radius=0.5, penalty=1.0
    )
    assert step == [0.0, 0.0]


def test_a_step_the_tableau_rounds_past_the_box_is_clamped_back_onto_it() -> None:
    # The box is a row of the program, so the tableau's answer is inside it in
    # exact arithmetic -- and the clamp is what makes that true in IEEE
    # arithmetic as well, which is a stronger claim than "inside it up to a
    # rounding". This subproblem is the measured case: the tableau returns
    # `radius * (1 + 2**-52)` on the first coordinate, one unit in the last place
    # outside the box, and the clamp is the only thing that brings it back. The
    # slack in the grid sweep above (`radius + 1e-12`) is why the clamp needs a
    # test of its own rather than being covered by it.
    objective = _AffineModel(
        value=-1.9687574784706794,
        gradient=(-2.9033208669053012, 2.4528476898692464, 2.924410248426126),
    )
    models = (
        _AffineModel(
            value=-0.82458719193151,
            gradient=(
                -0.048400110169496635,
                -2.61701797095117,
                0.5400371852183206,
            ),
        ),
        _AffineModel(
            value=-1.5617649516800016,
            gradient=(
                2.0220175827180658,
                -2.9677865187645036,
                0.14808309015773968,
            ),
        ),
    )
    radius = 1.297197253673893
    step = _trust_region_step(
        objective, models, radius=radius, penalty=37.90996880064371
    )
    assert abs(radius * (1 + 2**-52) - radius) > 0.0
    assert all(abs(value) <= radius for value in step)
    assert max(abs(value) for value in step) == radius


def test_a_concave_constraint_is_what_the_optimism_rule_is_for() -> None:
    # The docstring's second acceptance rule is that the price doubles when the
    # trial point's true violation exceeds the one the linear models predicted,
    # "the case where the models were optimistic about feasibility". A constraint
    # whose feasible region is the inside of a ball is exactly that case, because
    # a concave function's linearization lies above it, so the model reports a
    # violation no larger than the truth. Measured both ways: with the rule, the
    # run leaves the infeasible point that scores better and returns the
    # constrained optimum; with the rule pinned off, the price stops at 2.0 and
    # the run returns `[1.5, 1.5]`, which violates the ball by 3.5 and scores 4.5.
    # The optimum is on the ball's surface along the diagonal, so its value is
    # analytic and is compared to that rather than to a transcript.
    optimum = 2 * (3.0 - math.sqrt(0.5)) ** 2
    result = CobylaOptimizer(rhobeg=0.5, rhoend=1e-8, maxiter=200).minimize(
        _offset_to((3.0, 3.0)),
        torch.tensor([2.5, 2.5], dtype=torch.float64),
        constraints=[lambda v: 1.0 - (v**2).sum()],
    )
    assert result.feasible is True
    assert result.residual == 0.0
    assert result.value == pytest.approx(optimum, abs=1e-9)
    assert result.parameters.tolist() == pytest.approx(
        [math.sqrt(0.5), math.sqrt(0.5)], abs=1e-6
    )


def test_a_quadratic_reaches_its_minimizer_with_no_constraints() -> None:
    result = CobylaOptimizer(rhobeg=0.5, rhoend=1e-10, maxiter=400).minimize(
        _offset_to((1.0, -2.0)), _origin()
    )
    assert result.parameters.tolist() == pytest.approx([1.0, -2.0], abs=1e-9)
    assert result.value == pytest.approx(0.0, abs=1e-18)
    assert result.converged is True
    assert result.feasible is True
    assert result.residual == 0.0
    assert result.violations.numel() == 0
    assert result.violations.dtype is torch.float64


def test_a_run_is_reproducible_bit_for_bit() -> None:
    def run() -> CobylaResult:
        return CobylaOptimizer(rhobeg=0.5, rhoend=1e-8).minimize(
            _energy,
            torch.tensor([0.2, 0.2], dtype=torch.float64),
            constraints=[lambda v: v[0] - 0.9],
        )

    first, second = run(), run()
    assert torch.equal(first.parameters, second.parameters)
    assert first.value == second.value
    assert first.residual == second.residual
    assert first.iterations == second.iterations
    assert first.evaluations == second.evaluations
    assert first.penalty == second.penalty


def test_a_pauli_energy_is_minimized_under_a_bound() -> None:
    # The energy is smallest at a zero angle over the range this run explores, so
    # a floor on the angle is the whole answer: the constraint binds exactly at
    # the floor and the value is the energy there.
    constrained = CobylaOptimizer(rhobeg=0.5, rhoend=1e-8).minimize(
        _energy,
        torch.tensor([0.2, 0.2], dtype=torch.float64),
        constraints=[lambda v: v[0] - 0.9],
    )
    assert float(constrained.parameters[0]) == pytest.approx(0.9, abs=1e-6)
    assert constrained.value == pytest.approx(-2.0 * math.cos(0.9), abs=1e-8)
    assert constrained.violations.tolist() == pytest.approx([0.0], abs=1e-6)
    assert constrained.feasible is True
    assert constrained.converged is True
    unconstrained = CobylaOptimizer(rhobeg=0.5, rhoend=1e-8).minimize(
        _energy, torch.tensor([0.2, 0.2], dtype=torch.float64)
    )
    assert unconstrained.value == pytest.approx(-2.0, abs=1e-12)
    assert float(unconstrained.parameters[0]) == pytest.approx(0.0, abs=1e-6)
    assert unconstrained.value < constrained.value


def test_a_looser_bound_is_the_one_the_run_stops_on() -> None:
    for floor in (0.5, 0.9):
        # The bound is bound into the closure, so the two iterations are two
        # different constraints rather than one late-binding lookup.
        bound = floor
        result = CobylaOptimizer(rhobeg=0.5, rhoend=1e-8).minimize(
            _energy,
            torch.tensor([0.2, 0.2], dtype=torch.float64),
            constraints=[lambda v, bound=bound: v[0] - bound],
        )
        assert float(result.parameters[0]) == pytest.approx(floor, abs=1e-6)
        assert result.value == pytest.approx(-2.0 * math.cos(floor), abs=1e-8)


def test_an_infeasible_start_is_repaired_when_the_price_covers_it() -> None:
    result = CobylaOptimizer(
        rhobeg=0.5, rhoend=1e-10, maxiter=500, penalty=4.0
    ).minimize(
        _offset_to((1.0, 1.0)),
        _origin(),
        constraints=[lambda v: v[0] - 2.0],
    )
    assert result.parameters.tolist() == pytest.approx([2.0, 1.0], abs=1e-7)
    assert result.value == pytest.approx(1.0, abs=1e-7)
    assert result.residual == 0.0
    assert result.feasible is True
    assert result.violations.tolist() == pytest.approx([0.0], abs=1e-9)


def test_a_price_that_is_too_low_returns_an_infeasible_point_that_scores_better() -> (
    None
):
    # One quadratic, one affine constraint, one option changed. The cheaper price
    # settles where the objective is worth more than the constraint costs, so the
    # run stops outside the feasible set at a value the feasible optimum cannot
    # beat; the wider price reaches the feasible optimum itself. Neither run is
    # wrong: each minimized the merit it was given, and only the second price makes
    # that merit agree with the constrained problem.
    objective = _offset_to((3.0, 3.0))
    constraint = lambda v: 4.0 - v[0] - v[1]  # noqa: E731 - one expression
    cheap = CobylaOptimizer(rhobeg=0.5, rhoend=1e-10, maxiter=500).minimize(
        objective, _origin(), constraints=[constraint]
    )
    assert cheap.parameters.tolist() == pytest.approx([2.5, 2.5], abs=1e-8)
    assert cheap.value == pytest.approx(0.5, abs=1e-8)
    assert cheap.residual == pytest.approx(1.0, abs=1e-8)
    assert cheap.feasible is False
    assert cheap.violations.tolist() == pytest.approx([-1.0], abs=1e-8)
    exact = CobylaOptimizer(
        rhobeg=0.5, rhoend=1e-10, maxiter=500, penalty=2.0
    ).minimize(objective, _origin(), constraints=[constraint])
    assert exact.parameters.tolist() == pytest.approx([2.0, 2.0], abs=1e-8)
    assert exact.value == pytest.approx(2.0, abs=1e-8)
    assert exact.residual == 0.0
    assert exact.feasible is True
    # The point the cheap run returned scores better than the feasible optimum,
    # which is exactly why the flag has to be read beside the value.
    assert cheap.value < exact.value


def test_the_price_doubles_when_an_accepted_step_does_not_reduce_the_violation() -> (
    None
):
    # x + y is worth more per unit than one unit of the x >= 1 violation, so the
    # first accepted step trades one against the other and leaves the residual
    # where it was. That is the signal the price is too low, and the doubled price
    # then buys the boundary on the next iteration.
    result = CobylaOptimizer(
        rhobeg=0.5, rhoend=1e-10, maxiter=500, penalty_ceiling=2.0
    ).minimize(
        lambda v: v[0] + v[1],
        _origin(),
        constraints=[lambda v: v[0] - 1.0, lambda v: v[1] + 5.0],
    )
    assert result.penalty == 2.0
    assert result.parameters.tolist() == pytest.approx([1.0, -5.0], abs=1e-7)
    assert result.value == pytest.approx(-4.0, abs=1e-7)
    assert result.feasible is True
    assert result.residual == 0.0


def test_a_price_that_is_frozen_abandons_feasibility_entirely() -> None:
    # The same problem with the ceiling set to the starting price: the schedule
    # cannot double, so the run keeps trading the constraint away and reports a
    # residual far above the one it started with.
    result = CobylaOptimizer(
        rhobeg=0.5, rhoend=1e-10, maxiter=500, penalty_ceiling=1.0
    ).minimize(
        lambda v: v[0] + v[1],
        _origin(),
        constraints=[lambda v: v[0] - 1.0, lambda v: v[1] + 5.0],
    )
    assert result.penalty == 1.0
    assert result.feasible is False
    assert result.residual > 1.0
    assert result.converged is False
    assert float(result.parameters[0]) != pytest.approx(1.0, abs=1e-6)


def test_a_constraint_set_no_point_can_satisfy_is_reported_and_not_raised() -> None:
    result = CobylaOptimizer(rhobeg=0.5, rhoend=1e-10, maxiter=500).minimize(
        lambda v: v[0] * v[0],
        torch.zeros(1, dtype=torch.float64),
        constraints=[lambda v: v[0] - 1.0, lambda v: -v[0] - 1.0],
    )
    # x >= 1 and x <= -1 cannot both hold, so the run returns the point that
    # violates both least: the largest violation is 1.0 and it says so.
    assert result.residual == pytest.approx(1.0, abs=1e-8)
    assert result.feasible is False
    assert result.violations.tolist() == pytest.approx([-1.0, -1.0], abs=1e-8)
    assert result.converged is True


def test_the_reported_violations_keep_the_caller_s_order_and_count() -> None:
    result = CobylaOptimizer(rhobeg=0.25, rhoend=1e-8, maxiter=300).minimize(
        lambda v: v[0],
        torch.tensor([1.0], dtype=torch.float64),
        constraints=[
            lambda v: v[0] - 1.5,
            lambda v: 10.0 - v[0],
            lambda v: -3.0 - v[0],
        ],
    )
    values = result.violations.tolist()
    assert len(values) == 3
    # The objective pulls the point up, so the first and third constraints are the
    # ones that end violated, and the merit balances the two against each other
    # rather than letting either grow alone.
    assert values[0] < 0.0 and values[2] < 0.0
    assert values[1] > 0.0
    assert values[0] == pytest.approx(values[2], abs=1e-6)
    assert result.residual == pytest.approx(-min(values), abs=1e-12)
    assert result.feasible is False


def test_no_constraints_is_the_same_run_as_an_empty_constraint_sequence() -> None:
    default = CobylaOptimizer(rhobeg=0.5, rhoend=1e-9).minimize(
        _offset_to((0.5, 0.5)), _origin()
    )
    explicit = CobylaOptimizer(rhobeg=0.5, rhoend=1e-9).minimize(
        _offset_to((0.5, 0.5)), _origin(), constraints=[]
    )
    assert torch.equal(default.parameters, explicit.parameters)
    assert default.value == explicit.value
    assert default.evaluations == explicit.evaluations


def test_the_budget_bounds_a_run_that_never_settles() -> None:
    result = CobylaOptimizer(rhobeg=0.5, rhoend=1e-12, maxiter=5).minimize(
        _offset_to((3.0, 3.0)), _origin()
    )
    assert result.iterations == 5
    assert result.converged is False
    assert result.rho > result.rhoend


@pytest.mark.parametrize("maxiter", [1, 2, 3])
def test_each_iteration_reads_exactly_one_point_it_has_not_read(maxiter: int) -> None:
    # One parameter and no constraint, so an iteration builds a model at the base
    # point and a model at the trial point. The trial model's base is the simplex
    # offset the previous model already read, and its own offset is the one new
    # point, so the run reads two points before the first step and one more per
    # iteration after that -- an uncached implementation would read two per model.
    counter = _Counter(_offset_to((7.0,)))
    result = CobylaOptimizer(rhobeg=0.5, rhoend=1e-12, maxiter=maxiter).minimize(
        counter, torch.zeros(1, dtype=torch.float64)
    )
    assert result.iterations == maxiter
    assert counter.calls == maxiter + 2
    assert result.evaluations == maxiter + 2
    assert counter.calls < 2 * (maxiter + 1)


def test_the_reported_cost_is_the_sum_over_the_objective_and_the_constraints() -> None:
    optimizer = CobylaOptimizer(rhobeg=0.5, rhoend=1e-12, maxiter=1)
    with_two = optimizer.minimize(
        lambda v: (v[0] - 7.0) ** 2,
        torch.zeros(1, dtype=torch.float64),
        constraints=[lambda v: v[0], lambda v: -v[0]],
    )
    # Three distinct points, three functions, so each function is called once per
    # point and the run reports nine calls in total.
    assert with_two.evaluations == 9
    optimizer = CobylaOptimizer(rhobeg=0.5, rhoend=1e-12, maxiter=1)
    without = optimizer.minimize(
        lambda v: (v[0] - 7.0) ** 2, torch.zeros(1, dtype=torch.float64)
    )
    assert without.evaluations == 3


def test_the_cost_is_reset_by_every_run() -> None:
    optimizer = CobylaOptimizer(rhobeg=0.5, rhoend=1e-10, maxiter=1)
    first = optimizer.minimize(_offset_to((7.0,)), _origin(1))
    assert optimizer.evaluations == first.evaluations
    second = optimizer.minimize(_offset_to((7.0,)), _origin(1))
    assert optimizer.evaluations == second.evaluations
    assert first.evaluations == second.evaluations


def test_the_objective_never_receives_the_caller_s_tensor() -> None:
    start = torch.zeros(2, dtype=torch.float64)
    observed: list[torch.Tensor] = []

    def objective(parameters: torch.Tensor) -> torch.Tensor:
        observed.append(parameters)
        return (parameters**2).sum()

    CobylaOptimizer(rhobeg=0.5, rhoend=1e-9, maxiter=3).minimize(objective, start)
    assert observed
    assert all(seen is not start for seen in observed)
    assert start.tolist() == [0.0, 0.0]


def test_a_float32_starting_point_stays_float32() -> None:
    result = CobylaOptimizer(rhobeg=0.5, rhoend=1e-6, maxiter=200).minimize(
        _offset_to((1.0, -1.0)), torch.zeros(2, dtype=torch.float32)
    )
    assert result.parameters.dtype is torch.float32
    assert result.parameters.tolist() == pytest.approx([1.0, -1.0], abs=1e-5)
    assert result.violations.dtype is torch.float64


def test_the_returned_point_is_shaped_like_the_starting_point() -> None:
    result = CobylaOptimizer(rhobeg=0.5, rhoend=1e-8, maxiter=300).minimize(
        _offset_to((1.0,) * 6), torch.zeros(2, 3, dtype=torch.float64)
    )
    assert result.parameters.shape == (2, 3)
    assert result.parameters.reshape(-1).tolist() == pytest.approx([1.0] * 6, abs=1e-7)


def test_a_run_with_one_parameter_exercises_the_smallest_subproblem() -> None:
    result = CobylaOptimizer(rhobeg=0.5, rhoend=1e-12, maxiter=400).minimize(
        _offset_to((3.0,)), _origin(1)
    )
    assert float(result.parameters[0]) == pytest.approx(3.0, abs=1e-10)
    assert result.converged is True


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("maxiter", 0),
        ("maxiter", -1),
        ("maxiter", 1.0),
        ("maxiter", True),
        ("penalty", 0.0),
        ("penalty", float("nan")),
        ("penalty_ceiling", 0.0),
        ("penalty_ceiling", True),
        ("constraint_tolerance", 0.0),
        ("constraint_tolerance", float("inf")),
    ],
)
def test_an_option_that_is_not_what_its_name_promises_is_refused(
    option: str, value: object
) -> None:
    with pytest.raises(ValidationError):
        CobylaOptimizer(**{option: value})


@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf"), "1"])
def test_a_starting_radius_that_is_not_a_positive_number_is_refused(
    value: object,
) -> None:
    # A string is not a number, so the unit refuses it rather than letting the
    # comparison against the floor coerce it into one.
    with pytest.raises(ValidationError, match="rhobeg must be a finite positive"):
        CobylaOptimizer(rhobeg=value)


@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), None])
def test_a_radius_floor_that_is_not_a_positive_number_is_refused(value: object) -> None:
    with pytest.raises(ValidationError, match="rhoend must be a finite positive"):
        CobylaOptimizer(rhoend=value)


def test_a_floor_that_is_not_below_the_starting_radius_is_refused() -> None:
    with pytest.raises(ValidationError, match="strictly below rhobeg"):
        CobylaOptimizer(rhobeg=0.5, rhoend=0.5)
    with pytest.raises(ValidationError, match="strictly below rhobeg"):
        CobylaOptimizer(rhobeg=0.5, rhoend=1.0)


def test_a_ceiling_below_the_starting_price_is_refused() -> None:
    with pytest.raises(ValidationError, match="at least penalty"):
        CobylaOptimizer(penalty=2.0, penalty_ceiling=1.0)


def test_the_options_are_read_back_unchanged() -> None:
    optimizer = CobylaOptimizer(
        maxiter=17,
        rhobeg=0.25,
        rhoend=1e-7,
        penalty=3.0,
        penalty_ceiling=12.0,
        constraint_tolerance=1e-5,
    )
    assert optimizer.maxiter == 17
    assert optimizer.rhobeg == 0.25
    assert optimizer.rhoend == 1e-7
    assert optimizer.penalty == 3.0
    assert optimizer.penalty_ceiling == 12.0
    assert optimizer.constraint_tolerance == 1e-5


def test_an_objective_that_is_not_callable_is_refused() -> None:
    with pytest.raises(ValidationError, match="objective must be callable"):
        CobylaOptimizer().minimize(5, _origin())


def test_a_constraint_collection_that_is_not_a_sequence_of_callables_is_refused() -> (
    None
):
    with pytest.raises(ValidationError, match="sequence of callables"):
        CobylaOptimizer().minimize(lambda v: v.sum(), _origin(), constraints="abc")
    with pytest.raises(ValidationError, match="sequence of callables"):
        CobylaOptimizer().minimize(lambda v: v.sum(), _origin(), constraints=3.0)
    with pytest.raises(ValidationError, match="constraint 1 must be callable"):
        CobylaOptimizer().minimize(
            lambda v: v.sum(), _origin(), constraints=[lambda v: v[0], 7]
        )


@pytest.mark.parametrize(
    ("parameters", "message"),
    [
        ([0.0, 0.0], "parameters must be a torch.Tensor"),
        (torch.zeros(2, dtype=torch.int64), "must have a floating dtype"),
        (torch.zeros(0, dtype=torch.float64), "at least one value"),
        (
            torch.tensor([0.0, float("nan")], dtype=torch.float64),
            "parameters must all be finite",
        ),
    ],
)
def test_a_starting_point_that_is_not_a_finite_float_tensor_is_refused(
    parameters: object, message: str
) -> None:
    with pytest.raises(ValidationError, match=message):
        CobylaOptimizer().minimize(lambda v: v.sum(), parameters)


@pytest.mark.parametrize(
    ("function", "message"),
    [
        (lambda v: 1.0, "must return a torch.Tensor"),
        (lambda v: torch.zeros(2, dtype=torch.float64), "exactly one value"),
        (
            lambda v: torch.tensor(float("nan"), dtype=torch.float64),
            "returned a non-finite value",
        ),
        (
            lambda v: torch.tensor(float("inf"), dtype=torch.float64),
            "returned a non-finite value",
        ),
    ],
)
def test_a_function_that_does_not_return_one_finite_scalar_is_refused(
    function, message: str
) -> None:
    with pytest.raises(ValidationError, match=message):
        CobylaOptimizer().minimize(function, _origin())


def test_a_function_that_writes_into_its_argument_is_refused() -> None:
    def objective(parameters: torch.Tensor) -> torch.Tensor:
        return parameters.add_(1.0).sum()

    with pytest.raises(ValidationError, match="modified the tensor it was given"):
        CobylaOptimizer().minimize(objective, _origin())


def test_a_constraint_that_does_not_return_one_finite_scalar_is_refused() -> None:
    with pytest.raises(
        ValidationError, match="constraint 0 must return a torch.Tensor"
    ):
        CobylaOptimizer().minimize(
            lambda v: v.sum(), _origin(), constraints=[lambda v: 1.0]
        )
    with pytest.raises(ValidationError, match="constraint 0 returned a non-finite"):
        CobylaOptimizer().minimize(
            lambda v: v.sum(),
            _origin(),
            constraints=[lambda v: torch.tensor(float("nan"))],
        )


def test_a_constraint_that_writes_into_its_argument_is_refused() -> None:
    def constraint(parameters: torch.Tensor) -> torch.Tensor:
        return parameters.mul_(2.0).sum()

    with pytest.raises(ValidationError, match="constraint 0 modified the tensor"):
        CobylaOptimizer().minimize(
            lambda v: v.sum(), _origin(), constraints=[constraint]
        )


def test_a_refused_run_leaves_nothing_half_described_behind() -> None:
    calls = _Counter(_offset_to((1.0,)))
    optimizer = CobylaOptimizer(rhobeg=0.5, rhoend=1e-9)
    with pytest.raises(ValidationError, match="must return a torch.Tensor"):
        optimizer.minimize(lambda v: calls(v).item(), _origin(1))
    # The objective was called once and refused, and the same optimizer still
    # answers a well-formed run rather than remembering the refused value.
    assert calls.calls == 1
    result = optimizer.minimize(calls, _origin(1))
    assert result.value == pytest.approx(0.0, abs=1e-9)
    assert float(result.parameters[0]) == pytest.approx(1.0, abs=1e-9)


def _valid_result(**overrides: object) -> CobylaResult:
    fields: dict[str, object] = {
        "parameters": torch.zeros(2, dtype=torch.float64),
        "value": 1.0,
        "violations": torch.tensor([0.5], dtype=torch.float64),
        "residual": 0.0,
        "feasible": True,
        "rho": 1e-8,
        "rhoend": 1e-8,
        "penalty": 1.0,
        "constraint_tolerance": 1e-8,
        "iterations": 1,
        "evaluations": 3,
        "converged": True,
    }
    fields.update(overrides)
    return CobylaResult(**fields)  # type: ignore[arg-type]


def test_a_valid_result_carries_the_assumptions_and_limitations_of_the_unit() -> None:
    result = _valid_result()
    assert result.assumptions is COBYLA_ASSUMPTIONS
    assert result.limitations is COBYLA_LIMITATIONS
    assert result.assumptions and result.limitations


def test_a_result_that_is_not_a_tensor_where_one_is_needed_is_refused() -> None:
    with pytest.raises(TypeError, match="parameters must be the tensor"):
        _valid_result(parameters=[0.0, 0.0])
    with pytest.raises(TypeError, match="violations must be the tensor"):
        _valid_result(violations=[0.0])


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"parameters": torch.zeros(0, dtype=torch.float64)}, "at least one value"),
        (
            {"parameters": torch.tensor([float("nan")], dtype=torch.float64)},
            "must all be finite",
        ),
        ({"value": 1}, "value must be finite"),
        ({"value": float("inf")}, "value must be finite"),
        ({"violations": torch.zeros(2, 2, dtype=torch.float64)}, "one value per"),
        (
            {"violations": torch.tensor([float("inf")], dtype=torch.float64)},
            "every constraint value must be finite",
        ),
        ({"rho": 0.0}, "rho must be a finite positive"),
        ({"rhoend": -1.0}, "rhoend must be a finite positive"),
        ({"penalty": 0.0}, "penalty must be a finite positive"),
        ({"constraint_tolerance": 0.0}, "constraint_tolerance must be a finite"),
        ({"iterations": True}, "iterations must be a non-negative integer"),
        ({"iterations": -1}, "iterations must be a non-negative integer"),
        ({"iterations": 1.0}, "iterations must be a non-negative integer"),
        ({"evaluations": -1}, "evaluations must be a non-negative integer"),
        ({"evaluations": True}, "evaluations must be a non-negative integer"),
    ],
)
def test_a_result_field_that_is_not_what_its_name_promises_is_refused(
    overrides: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _valid_result(**overrides)


def test_a_result_refuses_a_run_that_cannot_have_produced_it() -> None:
    with pytest.raises(ValueError, match="applied no iteration"):
        _valid_result(iterations=0)
    with pytest.raises(ValueError, match="one evaluation per simplex point"):
        _valid_result(evaluations=2)
    assert _valid_result(evaluations=3, iterations=1).evaluations == 3


def test_a_result_refuses_a_residual_its_constraint_values_do_not_carry() -> None:
    with pytest.raises(ValueError, match="largest violation the constraint values"):
        _valid_result(violations=torch.tensor([-2.0]), residual=0.0)
    with pytest.raises(ValueError, match="largest violation the constraint values"):
        _valid_result(violations=torch.tensor([0.5]), residual=0.5)
    # The largest violation of a satisfied set is zero, not its smallest value.
    assert _valid_result(violations=torch.tensor([0.5]), residual=0.0).residual == 0.0


def test_a_result_refuses_a_feasibility_flag_its_own_numbers_contradict() -> None:
    # A violation inside the tolerance is feasible, and one outside it is not; the
    # float64 tensor keeps the violation the residual is compared against exact.
    tiny = torch.tensor([-1e-10], dtype=torch.float64)
    assert (
        _valid_result(violations=tiny, residual=1e-10, feasible=True).feasible is True
    )
    with pytest.raises(ValueError, match="feasible must report the residual"):
        _valid_result(violations=torch.tensor([-1.0]), residual=1.0, feasible=True)
    with pytest.raises(ValueError, match="feasible must report the residual"):
        _valid_result(violations=tiny, residual=1e-10, feasible=False)
    assert (
        _valid_result(
            violations=torch.tensor([-1.0]), residual=1.0, feasible=False
        ).feasible
        is False
    )


def test_a_result_refuses_a_convergence_flag_its_radius_contradicts() -> None:
    with pytest.raises(ValueError, match="converged must report the radius"):
        _valid_result(rho=1.0, rhoend=1e-8, converged=True)
    with pytest.raises(ValueError, match="converged must report the radius"):
        _valid_result(rho=1e-8, rhoend=1e-8, converged=False)
    assert _valid_result(rho=1.0, rhoend=1e-8, converged=False).converged is False
    assert _valid_result(rho=1e-8, rhoend=1e-8, converged=True).converged is True


def test_the_largest_violation_of_no_constraint_is_zero() -> None:
    assert _largest_violation([]) == 0.0
    assert _largest_violation([1.0, 2.0]) == 0.0
    assert _largest_violation([1.0, -2.0, -0.5]) == 2.0


def test_the_local_method_limitation_is_the_one_this_unit_measures() -> None:
    # The unit promises a local method over affine models and a measured
    # feasibility, not optimality. A run whose trust region collapses against a
    # constraint that binds reports converged, and the value it reports is the
    # merit's, so reading it as a constrained optimum is the mistake the
    # limitations warn about.
    result = CobylaOptimizer(rhobeg=0.5, rhoend=1e-10, maxiter=500).minimize(
        _offset_to((3.0, 3.0)),
        _origin(),
        constraints=[lambda v: 4.0 - v[0] - v[1]],
    )
    assert result.converged is True
    assert result.feasible is False
    assert result.rho <= result.rhoend
    assert result.limitations is COBYLA_LIMITATIONS
    assert any("converged" in line for line in result.limitations)
    assert any("penalty is a schedule" in line for line in result.limitations)


def test_the_unit_is_reachable_from_the_algorithms_namespace() -> None:
    """The unit is an ``algorithms`` name; a root promotion is a separate change."""
    import flagquantum.algorithms as algorithms
    from flagquantum.algorithms import cobyla as cobyla_module

    assert algorithms.CobylaOptimizer is CobylaOptimizer
    assert algorithms.CobylaResult is CobylaResult
    assert algorithms.CobylaOptimizer.__module__ == "flagquantum.algorithms.cobyla"
    assert cobyla_module._trust_region_step is _trust_region_step
    for name in (
        "COBYLA_ASSUMPTIONS",
        "COBYLA_LIMITATIONS",
        "CobylaOptimizer",
        "CobylaResult",
        "cobyla",
    ):
        assert name in algorithms.__all__, name
        assert getattr(algorithms, name) is not None
    # The module declares exactly these four names, and the root namespace is
    # untouched: the package is where an optimizer lives.
    assert sorted(cobyla_module.__all__) == [
        "COBYLA_ASSUMPTIONS",
        "COBYLA_LIMITATIONS",
        "CobylaOptimizer",
        "CobylaResult",
    ]
    assert not hasattr(fq, "CobylaOptimizer")
