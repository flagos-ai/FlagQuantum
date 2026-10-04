"""Nelder-Mead: reproducibility, cost, the refusals, and the local-minimum limit.

Three of these tests pin measurements rather than behaviour, and each of them is
the evidence for a sentence in the unit's own ``limitations``:

- :func:`test_a_two_well_quartic_converges_into_the_nearer_well` measures the
  local-minimum failure the unit admits to. It reports ``converged=True`` at a
  point ``0.199984362162`` above the global minimum, which is the honest reading
  of the flag: the simplex collapsed, and where it collapsed is a property of the
  starting point.
- :func:`test_the_published_counterexample_start_is_refused` measures that the
  one construction the textbooks use to break this method cannot be expressed
  here at all. McKinnon's example starts from a collinear simplex precisely
  because the method never leaves the line it starts on and then converges to a
  point that is not a minimum; refusing a simplex that does not span the
  parameter directions fails closed on that construction instead of reproducing
  it. The citation is Lagarias, Reeds, Wright, and Wright, *Convergence
  properties of the Nelder-Mead simplex method in low dimensions*, SIAM Journal
  on Optimization 9(1), 1998.
- :func:`test_a_pauli_energy_is_minimized_from_three_starts` measures the unit
  against a real objective: a two-qubit Pauli energy whose minimum is exactly
  ``-2``, reached from three different starting points with no gradient, no
  parameter-shift rule, and no random draw.
"""

from __future__ import annotations

import math

import pytest
import torch

import flagquantum as fq
from flagquantum.algorithms import (
    NELDER_MEAD_ASSUMPTIONS,
    NELDER_MEAD_LIMITATIONS,
    NelderMeadOptimizer,
    NelderMeadResult,
)
from flagquantum.errors import ValidationError

pytestmark = pytest.mark.unit

_TARGET = torch.tensor([1.5, -2.0], dtype=torch.float64)


class _Counter:
    """A scalar objective that records how often it was called."""

    def __init__(self, function):
        self._function = function
        self.calls = 0

    def __call__(self, parameters: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return self._function(parameters)


def _quadratic(parameters: torch.Tensor) -> torch.Tensor:
    """A separable convex objective with its minimizer at the origin."""

    return (parameters**2).sum()


def _rosenbrock(parameters: torch.Tensor) -> torch.Tensor:
    """The curved valley a simplex has to follow one step at a time."""

    first, second = parameters[0], parameters[1]
    return ((1.0 - first) ** 2 + 100.0 * (second - first**2) ** 2).reshape(())


def _two_well(parameters: torch.Tensor) -> torch.Tensor:
    """Two minima, one at ``-1.0123`` and one shallower at ``0.9873``."""

    value = parameters[0]
    return ((value * value - 1.0) ** 2 + 0.1 * value).reshape(())


def _mckinnon(parameters: torch.Tensor) -> torch.Tensor:
    """McKinnon's function, whose minimizer is ``(0, -0.5)`` at ``-0.25``.

    It is piecewise: the negative half of the first coordinate is scaled by
    ``theta * phi`` and the positive half by ``theta``, which is what makes the
    method's own reflection decisions walk in one direction forever.
    """

    first, second = float(parameters[0]), float(parameters[1])
    leading = 6.0 * (2.0 * abs(first) if first <= 0.0 else first)
    return torch.tensor(leading + second + second * second, dtype=torch.float64)


def _energy(values: torch.Tensor) -> torch.Tensor:
    """The negative Pauli energy ``-(<Z0> + <Z1>)`` of a two-parameter circuit."""

    circuit = fq.Circuit(2, dtype=torch.complex128)
    circuit = circuit.ry(0, values[0]).ry(1, values[1]).cx(0, 1)
    outputs = fq.expectation(fq.Z(0) + fq.Z(1))
    return -fq.run(circuit, outputs=outputs).expectation().sum()


def _global_minimum_of_the_two_well() -> tuple[float, float]:
    """Return the global minimum of :func:`_two_well` and where it is.

    ``(x**2 - 1)**2 + 0.1 x`` is a quartic that goes to infinity at both ends,
    so its global minimum is its smallest value at a stationary point, and the
    stationary points are the roots of ``4 x**3 - 4 x + 0.1``. They are
    bracketed on a grid and bisected to machine precision here, which keeps the
    number this module compares against independent of the optimizer the module
    is checking.
    """
    grid = torch.linspace(-3.0, 3.0, 20001, dtype=torch.float64)
    derivative = 4.0 * grid**3 - 4.0 * grid + 0.1
    roots: set[float] = set()
    for index in range(grid.numel() - 1):
        low, high = float(grid[index]), float(grid[index + 1])
        low_value = float(derivative[index])
        if low_value == 0.0:
            roots.add(low)
            continue
        if low_value * float(derivative[index + 1]) >= 0.0:
            continue
        for _ in range(200):
            middle = 0.5 * (low + high)
            if low_value * (4.0 * middle**3 - 4.0 * middle + 0.1) <= 0.0:
                high = middle
            else:
                low, low_value = middle, 4.0 * middle**3 - 4.0 * middle + 0.1
        roots.add(0.5 * (low + high))
    return min(((root * root - 1.0) ** 2 + 0.1 * root, root) for root in roots)


def test_a_run_is_reproducible_bit_for_bit() -> None:
    """Nothing here draws a random number, so a replay is the same run."""
    start = torch.tensor([0.4, 0.4], dtype=torch.float64)
    first = NelderMeadOptimizer(maxiter=200).minimize(_energy, start)
    second = NelderMeadOptimizer(maxiter=200).minimize(_energy, start)
    assert torch.equal(first.parameters, second.parameters)
    assert torch.equal(first.simplex, second.simplex)
    assert first.value == second.value
    assert first.iterations == second.iterations
    assert first.evaluations == second.evaluations


def test_a_quadratic_reaches_its_minimizer() -> None:
    """A convex objective with a reachable minimum is reached exactly."""
    result = NelderMeadOptimizer(maxiter=400, atol=1e-14).minimize(
        lambda point: ((point - _TARGET) ** 2).sum(),
        torch.zeros(2, dtype=torch.float64),
    )
    assert result.value < 1e-12
    assert torch.allclose(result.parameters, _TARGET, atol=1e-6)
    assert result.converged is True


def test_the_curved_valley_is_followed_to_the_minimum_from_five_starts() -> None:
    """The method follows the valley itself, and not just a convex bowl."""
    for start in ([1.2, 1.0], [-1.2, 1.0], [0.0, 0.0], [2.0, 2.0], [-2.0, 2.0]):
        result = NelderMeadOptimizer(maxiter=2000, atol=1e-14).minimize(
            _rosenbrock, torch.tensor(start, dtype=torch.float64)
        )
        assert result.converged is True, start
        assert result.value < 1e-12, start
        assert torch.allclose(
            result.parameters, torch.ones(2, dtype=torch.float64), atol=1e-6
        ), start


def test_every_objective_call_is_counted_and_the_count_is_bracketed() -> None:
    """The cost is a count of calls, and the bracket is what the branches cost.

    Each replacement evaluates the reflection and then at most one more point,
    unless even the contraction fails, in which case the whole simplex is
    re-evaluated: one call per iteration is the floor, and ``n + 1`` per
    iteration is the ceiling.
    """
    for count in (1, 2, 5):
        objective = _Counter(_quadratic)
        start = torch.full((count,), 0.7, dtype=torch.float64)
        optimizer = NelderMeadOptimizer(maxiter=40)
        result = optimizer.minimize(objective, start)
        assert result.evaluations == objective.calls
        assert optimizer.evaluations == objective.calls
        floor = (count + 1) + result.iterations
        ceiling = (count + 1) + (count + 1) * result.iterations
        assert floor <= result.evaluations <= ceiling


def test_the_budget_bounds_a_run_that_never_settles() -> None:
    """An unreachable pair of thresholds is a bounded run and not a failure."""
    objective = _Counter(_rosenbrock)
    optimizer = NelderMeadOptimizer(maxiter=7, tolerance=1e-300, atol=1e-300)
    result = optimizer.minimize(
        objective, torch.tensor([-1.2, 1.0], dtype=torch.float64)
    )
    assert result.iterations == 7
    assert result.converged is False
    assert result.evaluations == objective.calls
    assert result.value_spread > result.atol
    assert result.simplex_spread > result.tolerance


def test_the_default_simplex_is_the_coordinate_offsets_it_documents() -> None:
    """The built simplex offsets each coordinate by a fraction of its own size."""
    optimizer = NelderMeadOptimizer(initial_step=0.25)
    point = torch.tensor([0.0, 100.0], dtype=torch.float64)
    simplex = optimizer.initial_simplex(point)
    assert simplex.shape == (3, 2)
    assert torch.equal(simplex[0], point)
    assert torch.equal(simplex[1], torch.tensor([0.25, 100.0], dtype=torch.float64))
    assert torch.equal(simplex[2], torch.tensor([0.0, 125.0], dtype=torch.float64))


def test_a_caller_supplied_simplex_is_the_simplex_that_is_run() -> None:
    """The first ``n + 1`` evaluations are the starting simplex, whoever built it."""
    seen: list[torch.Tensor] = []

    def objective(point: torch.Tensor) -> torch.Tensor:
        seen.append(point.clone())
        return _quadratic(point)

    start = torch.tensor([4.0, 4.0], dtype=torch.float64)
    supplied = torch.tensor([[4.0, 4.0], [4.5, 4.0], [4.0, 4.5]], dtype=torch.float64)
    NelderMeadOptimizer(maxiter=3).minimize(objective, start, simplex=supplied)
    assert torch.equal(torch.stack(seen[:3]), supplied)

    built: list[torch.Tensor] = []

    def built_objective(point: torch.Tensor) -> torch.Tensor:
        built.append(point.clone())
        return _quadratic(point)

    optimizer = NelderMeadOptimizer(maxiter=3)
    optimizer.minimize(built_objective, start)
    assert torch.equal(torch.stack(built[:3]), optimizer.initial_simplex(start))
    assert not torch.equal(torch.stack(built[:3]), supplied)


def test_the_objective_never_receives_the_caller_s_tensor() -> None:
    """The starting point is copied, and a graph on it is not carried in."""
    seen: list[torch.Tensor] = []

    def objective(point: torch.Tensor) -> torch.Tensor:
        seen.append(point)
        return _quadratic(point)

    start = torch.tensor([1.0, 2.0], dtype=torch.float64, requires_grad=True)
    NelderMeadOptimizer(maxiter=5).minimize(objective, start)
    assert seen
    assert all(vertex.requires_grad is False for vertex in seen)
    assert all(vertex is not start for vertex in seen)
    assert torch.equal(start.detach(), torch.tensor([1.0, 2.0], dtype=torch.float64))


def test_a_float32_starting_point_stays_float32() -> None:
    """The simplex runs in the caller's dtype rather than promoting it."""
    result = NelderMeadOptimizer(maxiter=400).minimize(
        lambda point: ((point - 1.0) ** 2).sum(), torch.zeros(3, dtype=torch.float32)
    )
    assert result.parameters.dtype == torch.float32
    assert result.simplex.dtype == torch.float32
    assert result.value == 0.0


def test_a_two_well_quartic_converges_into_the_nearer_well() -> None:
    """The measured local-minimum limit, and what the flag does not say.

    The global minimum is ``-0.100617376638`` at ``-1.012273131``, computed
    here from the stationary points rather than taken from this unit. Started at
    ``0.5`` the run settles in the shallower well at ``0.987257475`` and reports
    ``converged=True``, because it converged to a minimum and not to *the*
    minimum. Started at ``-0.5`` it finds the global one to 17 digits. The gap
    between them is measured here rather than described.
    """
    global_value, global_point = _global_minimum_of_the_two_well()
    assert global_value == pytest.approx(-0.100617376638, abs=1e-11)
    assert global_point == pytest.approx(-1.012273131, abs=1e-8)
    near = NelderMeadOptimizer(maxiter=400).minimize(
        _two_well, torch.tensor([0.5], dtype=torch.float64)
    )
    far = NelderMeadOptimizer(maxiter=400).minimize(
        _two_well, torch.tensor([-0.5], dtype=torch.float64)
    )
    assert near.converged is True
    assert far.converged is True
    assert near.value == pytest.approx(0.099366985524, abs=1e-11)
    assert near.parameters[0] == pytest.approx(0.9872574747, abs=1e-8)
    assert far.value == pytest.approx(global_value, abs=1e-11)
    assert far.parameters[0] == pytest.approx(global_point, abs=1e-8)
    assert near.value - global_value == pytest.approx(0.199984362162, abs=1e-11)


def test_the_published_counterexample_start_is_refused() -> None:
    """McKinnon's collinear simplex is refused rather than run.

    The method cannot leave the line its first simplex lies on, and the simplex
    does not span the parameter directions, so this unit refuses it by name
    instead of converging to a point that is not a minimum.
    """
    collinear = torch.tensor([[0.0, 0.0], [1.0, 1.0], [0.5, 0.5]], dtype=torch.float64)
    with pytest.raises(ValidationError) as caught:
        NelderMeadOptimizer().minimize(_mckinnon, collinear[0], simplex=collinear)
    assert "span all 2 parameter directions" in str(caught.value)
    assert "span only 1" in str(caught.value)


def test_a_non_degenerate_start_on_mckinnon_reaches_the_true_minimum() -> None:
    """With a simplex that spans the plane, the same function is solved."""
    simplex = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=torch.float64)
    result = NelderMeadOptimizer(maxiter=1000, tolerance=1e-12, atol=1e-16).minimize(
        _mckinnon, simplex[0], simplex=simplex
    )
    assert result.value == pytest.approx(-0.25, abs=1e-9)
    assert result.parameters[0] == pytest.approx(0.0, abs=1e-9)
    # The second coordinate is the flat one: the simplex spread is 6.6e-13
    # while this coordinate sits 5.2e-9 from the minimizer, because a value
    # this flat does not locate its own argmin any better than that.
    assert result.parameters[1] == pytest.approx(-0.5, abs=1e-6)


def test_a_pauli_energy_is_minimized_from_three_starts() -> None:
    """No gradient, no parameter-shift rule, no random draw, exact minimum."""
    for start in ([0.4, 0.4], [1.0, 1.0], [-0.7, 2.0]):
        result = NelderMeadOptimizer(maxiter=200).minimize(
            _energy, torch.tensor(start, dtype=torch.float64)
        )
        assert result.value == pytest.approx(-2.0, abs=1e-12), start
        assert result.converged is True, start
        assert torch.allclose(
            result.parameters, torch.zeros(2, dtype=torch.float64), atol=1e-6
        ), start


@pytest.mark.parametrize(
    ("settings", "message"),
    [
        ({"maxiter": 0}, "maxiter must be a positive integer"),
        ({"maxiter": True}, "maxiter must be a positive integer"),
        ({"initial_step": 0.0}, "initial_step must be a finite positive number"),
        ({"reflection": float("nan")}, "reflection must be a finite positive number"),
        ({"expansion": 1.0}, "expansion must exceed 1"),
        (
            {"reflection": 3.0, "expansion": 2.0},
            "expansion must be at least reflection",
        ),
        ({"contraction": 1.0}, "contraction must lie strictly between 0 and 1"),
        ({"shrink": 1.0}, "shrink must lie strictly between 0 and 1"),
        ({"shrink": 0.0}, "shrink must be a finite positive number"),
        ({"tolerance": -1.0}, "tolerance must be a finite positive number"),
        ({"atol": 0.0}, "atol must be a finite positive number"),
    ],
)
def test_a_coefficient_that_is_not_what_its_name_promises_is_refused(
    settings: dict[str, object], message: str
) -> None:
    with pytest.raises(ValidationError) as caught:
        NelderMeadOptimizer(**settings)  # type: ignore[arg-type]
    assert message in str(caught.value)


@pytest.mark.parametrize(
    ("parameters", "message"),
    [
        ([1.0, 2.0], "parameters must be a torch.Tensor"),
        (
            torch.tensor([1, 2], dtype=torch.int64),
            "parameters must have a floating dtype",
        ),
        (torch.zeros(0, dtype=torch.float64), "at least one value"),
        (
            torch.tensor([1.0, float("inf")], dtype=torch.float64),
            "parameters must all be finite",
        ),
    ],
)
def test_a_starting_point_that_is_not_a_finite_float_tensor_is_refused(
    parameters: object, message: str
) -> None:
    with pytest.raises(ValidationError) as caught:
        NelderMeadOptimizer().minimize(_quadratic, parameters)  # type: ignore[arg-type]
    assert message in str(caught.value)


@pytest.mark.parametrize(
    ("objective", "message"),
    [
        ("not callable", "objective must be callable"),
        (
            lambda point: float(point.sum()),
            "the objective must return a torch.Tensor",
        ),
        (
            lambda point: point.reshape(-1)[:1].repeat(2),
            "the objective must return exactly one value",
        ),
        (
            lambda point: torch.tensor([float("nan")], dtype=torch.float64),
            "the objective returned a non-finite value",
        ),
    ],
)
def test_an_objective_that_does_not_return_one_finite_scalar_is_refused(
    objective: object, message: str
) -> None:
    start = torch.ones(2, dtype=torch.float64)
    with pytest.raises(ValidationError) as caught:
        NelderMeadOptimizer(maxiter=3).minimize(objective, start)  # type: ignore[arg-type]
    assert message in str(caught.value)


def test_an_objective_that_writes_into_its_argument_is_refused() -> None:
    """A stored vertex and its value have to describe the same parameters."""

    def objective(point: torch.Tensor) -> torch.Tensor:
        point.mul_(0.0)
        return _quadratic(point)

    with pytest.raises(ValidationError) as caught:
        NelderMeadOptimizer(maxiter=3).minimize(
            objective, torch.ones(2, dtype=torch.float64)
        )
    assert "the objective modified the tensor it was given" in str(caught.value)


@pytest.mark.parametrize(
    ("simplex", "message"),
    [
        (
            torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=torch.float64),
            "must have shape (3, 2)",
        ),
        (
            torch.tensor([[0, 0], [1, 0], [0, 1]]),
            "a simplex must have a floating dtype",
        ),
        (
            torch.tensor(
                [[0.0, 0.0], [1.0, 0.0], [float("inf"), 1.0]], dtype=torch.float64
            ),
            "every vertex of a simplex must be finite",
        ),
        (
            [[0.0, 0.0], [1.0, 0.0]],
            "needs 3 vertices, and 2 were given",
        ),
    ],
)
def test_a_simplex_that_is_not_what_it_has_to_be_is_refused(
    simplex: object, message: str
) -> None:
    with pytest.raises(ValidationError) as caught:
        NelderMeadOptimizer().minimize(
            _quadratic,
            torch.zeros(2, dtype=torch.float64),
            simplex=simplex,  # type: ignore[arg-type]
        )
    assert message in str(caught.value)


def test_every_vertex_has_to_be_usable_and_the_simplex_has_to_span() -> None:
    """A simplex that is flat in one direction cannot move in it."""
    start = torch.zeros(3, dtype=torch.float64)
    flat = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
        dtype=torch.float64,
    )
    with pytest.raises(ValidationError) as caught:
        NelderMeadOptimizer().minimize(_quadratic, start, simplex=flat)
    assert "span all 3 parameter directions" in str(caught.value)
    assert "span only 2" in str(caught.value)


def test_a_tensor_simplex_has_to_agree_with_the_point_it_is_given() -> None:
    """A result reports one point and one simplex, so the two have to be one run's.

    Two things have to agree, and they fail for different reasons: the first
    vertex is the point the run starts from, and the vertices are moved in the
    parameters' dtype. A sequence of rows carries no dtype of its own, so it is
    built in the parameters' dtype rather than refused, which is what the second
    half of this test pins.
    """
    start = torch.zeros(2, dtype=torch.float64)
    elsewhere = torch.tensor([[1.0, 1.0], [1.5, 1.0], [1.0, 1.5]], dtype=torch.float64)
    with pytest.raises(ValidationError) as caught:
        NelderMeadOptimizer().minimize(_quadratic, start, simplex=elsewhere)
    assert "first vertex must be the starting point" in str(caught.value)

    narrow = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    with pytest.raises(ValidationError) as caught:
        NelderMeadOptimizer().minimize(_quadratic, start, simplex=narrow)
    assert "must carry the parameters' own dtype" in str(caught.value)

    result = NelderMeadOptimizer(maxiter=200, atol=1e-14).minimize(
        _quadratic,
        start,
        simplex=[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
    )
    assert result.simplex.dtype == torch.float64
    assert result.value < 1e-12


def test_a_result_refuses_a_converged_flag_its_spreads_do_not_support() -> None:
    """The flag is derived from the two spreads rather than asserted beside them."""
    parameters = torch.zeros(2, dtype=torch.float64)
    simplex = torch.tensor([[0.0, 0.0], [0.5, 0.0], [0.0, 0.5]], dtype=torch.float64)
    settings = {
        "parameters": parameters,
        "value": 1.0,
        "simplex": simplex,
        "value_spread": 1.0,
        "simplex_spread": 0.5,
        "tolerance": 1e-8,
        "atol": 1e-8,
        "iterations": 3,
        "evaluations": 9,
    }
    with pytest.raises(ValueError) as caught:
        NelderMeadResult(converged=True, **settings)
    assert "converged must report the two spreads" in str(caught.value)
    settled = NelderMeadResult(
        converged=True,
        **{**settings, "value_spread": 0.0, "simplex_spread": 0.0},
    )
    assert settled.converged is True
    assert settled.assumptions is NELDER_MEAD_ASSUMPTIONS
    assert settled.limitations is NELDER_MEAD_LIMITATIONS
    assert any("local method" in item for item in settled.assumptions)
    assert any("not guaranteed" in item for item in settled.limitations)


def test_a_result_refuses_a_simplex_that_is_not_the_run_it_reports() -> None:
    """One result carries one point and one simplex, from the same run."""
    parameters = torch.zeros(2, dtype=torch.float64)
    other = torch.tensor([[1.0, 1.0], [1.5, 1.0], [1.0, 1.5]], dtype=torch.float64)
    with pytest.raises(ValueError) as caught:
        NelderMeadResult(
            parameters=parameters,
            value=1.0,
            simplex=other,
            value_spread=0.0,
            simplex_spread=0.0,
            tolerance=1e-8,
            atol=1e-8,
            iterations=1,
            evaluations=4,
            converged=True,
        )
    assert "the simplex's own best vertex" in str(caught.value)


def test_a_result_refuses_a_run_that_cannot_have_ended_where_it_claims() -> None:
    """The two counts have to be consistent with how a run can end."""
    parameters = torch.zeros(2, dtype=torch.float64)
    simplex = torch.tensor([[0.0, 0.0], [0.5, 0.0], [0.0, 0.5]], dtype=torch.float64)
    settled = {
        "parameters": parameters,
        "value": 1.0,
        "simplex": simplex,
        "value_spread": 0.0,
        "simplex_spread": 0.0,
        "tolerance": 1e-8,
        "atol": 1e-8,
    }
    with pytest.raises(ValueError) as caught:
        NelderMeadResult(iterations=0, evaluations=3, converged=False, **settled)
    assert "applied no simplex replacement" in str(caught.value)
    assert (
        NelderMeadResult(
            iterations=0, evaluations=3, converged=True, **settled
        ).iterations
        == 0
    )

    with pytest.raises(ValueError) as caught:
        NelderMeadResult(
            iterations=1,
            evaluations=2,
            converged=False,
            **{**settled, "value_spread": 1.0, "simplex_spread": 1.0},
        )
    assert "at least one evaluation per vertex" in str(caught.value)
    assert (
        NelderMeadResult(
            iterations=1,
            evaluations=3,
            converged=False,
            **{**settled, "value_spread": 1.0, "simplex_spread": 1.0},
        ).evaluations
        == 3
    )


def test_the_local_method_limitation_is_the_one_this_unit_measures() -> None:
    """The limitations name the failure the two-well test exhibits, and no other."""
    named = " ".join(NELDER_MEAD_LIMITATIONS)
    assert "not guaranteed" in named
    assert "local minimum" in named
    assert "restart" in named
    assert math.isfinite(NelderMeadOptimizer().initial_step)
