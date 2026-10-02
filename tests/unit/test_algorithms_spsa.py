"""Simultaneous perturbation stochastic approximation: cost, refusals, replay."""

from __future__ import annotations

import math

import pytest
import torch

import flagquantum as fq
from flagquantum.algorithms import SPSAOptimizer
from flagquantum.errors import ValidationError

pytestmark = pytest.mark.unit


class _Counter:
    """A scalar objective that records how often it was called."""

    def __init__(self, function):
        self._function = function
        self.calls = 0

    def __call__(self, parameters: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return self._function(parameters)


def _quadratic(parameters: torch.Tensor) -> torch.Tensor:
    return (parameters**2).sum()


def _coupled(parameters: torch.Tensor) -> torch.Tensor:
    """A non-separable objective, so the estimate does depend on the draw.

    A separable objective's symmetric difference is even in every sign, so its
    estimate is the same for every perturbation and cannot show replay at all.
    """

    return torch.cos(parameters).sum() + parameters.prod()


def _seeded(seed: int, **settings: object) -> SPSAOptimizer:
    settings.setdefault("maxiter", 50)
    return SPSAOptimizer(
        generator=torch.Generator().manual_seed(seed),
        **settings,  # type: ignore[arg-type]
    )


def test_two_evaluations_per_step_regardless_of_parameter_count() -> None:
    """The cost that makes SPSA usable does not grow with the parameter count."""
    for count in (1, 8, 64):
        objective = _Counter(_quadratic)
        optimizer = _seeded(1, maxiter=5)
        parameters = torch.full((count,), 0.5, dtype=torch.float64)
        for _ in range(5):
            parameters = optimizer.step(objective, parameters)
        assert optimizer.steps == 5
        assert optimizer.evaluations == 10
        assert objective.calls == 10


def test_step_and_cost_costs_one_more_evaluation_than_step() -> None:
    """The reported cost is read at the pre-update point, not the post-update one."""
    objective = _Counter(_quadratic)
    optimizer = _seeded(2, maxiter=5)
    parameters = torch.tensor([1.0, 2.0], dtype=torch.float64)
    _, cost = optimizer.step_and_cost(objective, parameters)
    assert cost.item() == pytest.approx(5.0)
    assert optimizer.evaluations == 3
    assert optimizer.steps == 1


def test_estimate_is_exact_for_a_one_parameter_quadratic() -> None:
    """With one parameter, a simultaneous perturbation differences exactly.

    ``(p**2)`` takes ``+2 p c d + c**2`` when perturbed up and ``-2 p c d + c**2``
    when perturbed down, so ``(y+ - y-) / (2 c d)`` collapses to the analytic
    ``2p`` for every draw and every perturbation size.  That makes it the one
    objective whose estimate can be checked against a closed form.
    """
    parameters = torch.tensor([1.7], dtype=torch.float64)
    for seed in range(4):
        for perturbation in (1e-4, 0.2, 1.0):
            optimizer = _seeded(seed, perturbation=perturbation)
            estimate = optimizer.estimate_gradient(_quadratic, parameters)
            assert torch.allclose(estimate, 2.0 * parameters, atol=1e-9)
            assert optimizer.steps == 0
            assert optimizer.evaluations == 2


def test_the_estimate_is_unbiased_though_not_exact_over_many_parameters() -> None:
    """The estimate averages to the gradient and is noisy at any single draw.

    This is the honest statement of what SPSA gives: ``E[g_hat] = grad`` because
    the signs are independent, while one draw carries the cross-talk of every
    other coordinate.  A caller who needs the gradient itself at this point
    wants autograd or parameter shift, not this.  The mean is compared against
    its own standard error rather than a fixed tolerance, because the estimator's
    variance grows with the parameter magnitudes.
    """
    parameters = torch.tensor([0.3, -1.7, 2.4], dtype=torch.float64)
    optimizer = _seeded(0, maxiter=4096, perturbation=0.05)
    draws = torch.stack(
        [optimizer.estimate_gradient(_quadratic, parameters) for _ in range(4096)]
    )
    mean = draws.mean(dim=0)
    standard_error = draws.std(dim=0) / math.sqrt(draws.shape[0])
    assert torch.all(torch.abs(mean - 2.0 * parameters) < 4.0 * standard_error)
    assert not torch.allclose(draws[0], 2.0 * parameters, atol=1e-3)


def test_estimate_gradient_does_not_advance_the_counter() -> None:
    """Repeated estimates at one point keep ``c_k`` and redraw the perturbation.

    The estimate is invariant under negating every sign, because that swaps the
    two evaluations while flipping the divisor; with four parameters there are
    eight distinct outcomes, so several draws at one point do not all coincide.
    """
    optimizer = _seeded(3, maxiter=20, perturbation=0.5)
    parameters = torch.tensor([0.7, 1.1, 0.35, -0.9], dtype=torch.float64)
    draws = [
        tuple(optimizer.estimate_gradient(_coupled, parameters).tolist())
        for _ in range(8)
    ]
    assert optimizer.perturbation_size == pytest.approx(0.5)
    assert optimizer.steps == 0
    assert optimizer.evaluations == 16
    assert len(set(draws)) > 1


def test_gain_sequences_follow_the_declared_exponents() -> None:
    """``c_k = c / k**gamma`` and ``a_k = a / (A + k)**alpha``."""
    optimizer = SPSAOptimizer(
        stability=4.0,
        parameter_gain=0.9,
        perturbation=0.3,
        parameter_gain_exponent=0.6,
        perturbation_exponent=0.1,
    )
    parameters = torch.tensor([0.1], dtype=torch.float64)
    for index in (1, 2, 3, 4, 5, 6, 7):
        assert optimizer.perturbation_size == pytest.approx(0.3 / index**0.1)
        assert optimizer.step_size == pytest.approx(0.9 / (4.0 + index) ** 0.6)
        if index != 7:
            optimizer.step(_quadratic, parameters)


def test_first_step_defaults_to_five_percent_for_any_maxiter() -> None:
    """Supplying only ``maxiter`` still fixes ``a_1`` at 0.05."""
    for maxiter in (10, 200, 5000):
        optimizer = SPSAOptimizer(maxiter=maxiter)
        assert optimizer.steps == 0
        assert optimizer.step_size == pytest.approx(0.05)


def test_stability_alone_is_enough() -> None:
    """A caller who knows the stability constant need not state an iteration budget."""
    optimizer = SPSAOptimizer(stability=1.0)
    assert optimizer.maxiter is None
    assert optimizer.step_size == pytest.approx(0.05)


def test_a_seeded_generator_replays_the_trajectory_exactly() -> None:
    """The perturbation is reproducible from the call site, on the parameter device."""

    def trajectory(seed: int) -> list[float]:
        optimizer = _seeded(seed, maxiter=10)
        values = torch.tensor([1.0, 2.0], dtype=torch.float64)
        for _ in range(10):
            values = optimizer.step(_coupled, values)
        return values.tolist()

    first = trajectory(7)
    assert trajectory(7) == first
    assert trajectory(8) != first


def test_an_internal_generator_is_created_for_the_parameter_device() -> None:
    """Omitting the generator is allowed, and it draws where the parameters live."""
    optimizer = SPSAOptimizer(maxiter=5)
    parameters = torch.tensor([0.25], dtype=torch.float64)
    optimizer.estimate_gradient(_quadratic, parameters)
    assert optimizer._generator is not None
    assert str(optimizer._generator.device) == parameters.device.type


def test_a_generator_for_another_device_is_refused() -> None:
    """A generator that cannot draw where the parameters live is named, not ignored."""
    optimizer = _seeded(1, maxiter=5)
    optimizer._generator_device = "meta"
    with pytest.raises(ValidationError, match="draws on 'meta'"):
        optimizer.estimate_gradient(
            _quadratic, torch.tensor([1.0], dtype=torch.float64)
        )


def test_converges_on_a_circuit_energy_with_no_gradient() -> None:
    """The documented workflow: a two-qubit Pauli energy, two evaluations per step."""

    def energy(values: torch.Tensor) -> torch.Tensor:
        circuit = fq.Circuit(2, dtype=torch.complex128)
        circuit = circuit.ry(0, values[0]).ry(1, values[1]).cx(0, 1)
        outputs = fq.expectation(fq.Z(0) + fq.Z(1))
        return -fq.run(circuit, outputs=outputs).expectation().sum()

    optimizer = _seeded(13, maxiter=120, perturbation=0.25)
    values = torch.full((2,), 0.4, dtype=torch.float64)
    initial = float(energy(values))
    for _ in range(120):
        values = optimizer.step(energy, values)
    final = float(energy(values))
    assert final < initial
    assert final < -1.99
    assert optimizer.evaluations == 240


def test_converges_on_a_shot_noisy_circuit_objective() -> None:
    """A sampled objective has no exact gradient, which is what SPSA is for.

    Each evaluation draws a fresh but seeded 4096-shot batch, so the test itself
    replays: the noise is real and the outcome is deterministic.
    """
    shots = 4096
    state = {"evaluation": 0}

    def sampled_energy(values: torch.Tensor) -> torch.Tensor:
        state["evaluation"] += 1
        circuit = fq.Circuit(1, dtype=torch.complex128).ry(0, values[0])
        result = fq.run(
            circuit,
            outputs=fq.samples([0]),
            options=fq.ExecutionOptions(shots=shots, seed=state["evaluation"]),
        )
        return -(1.0 - 2.0 * result.samples.to(torch.float64).mean())

    optimizer = _seeded(17, maxiter=120, perturbation=0.3)
    values = torch.tensor([0.35], dtype=torch.float64)
    initial = float(sampled_energy(values))
    for _ in range(120):
        values = optimizer.step(sampled_energy, values)
    final = float(sampled_energy(values))
    assert initial < -0.9
    assert final < initial
    assert final < -0.98
    assert optimizer.evaluations == 240


@pytest.mark.parametrize(
    ("settings", "message"),
    [
        ({}, "stability constant"),
        ({"maxiter": 0}, "maxiter must be a positive integer"),
        ({"maxiter": -3}, "maxiter must be a positive integer"),
        ({"maxiter": True}, "maxiter must be a positive integer"),
        ({"stability": -1.0}, "stability must be a finite non-negative"),
        ({"stability": math.inf}, "stability must be a finite non-negative"),
        ({"maxiter": 10, "perturbation": 0.0}, "perturbation must be positive"),
        ({"maxiter": 10, "perturbation": -0.1}, "perturbation must be positive"),
        ({"maxiter": 10, "perturbation": math.nan}, "perturbation must be a finite"),
        ({"maxiter": 10, "parameter_gain": 0.0}, "parameter_gain must be a finite"),
        (
            {"maxiter": 10, "parameter_gain_exponent": math.inf},
            "parameter_gain_exponent must be a finite",
        ),
        (
            {"maxiter": 10, "perturbation_exponent": math.nan},
            "perturbation_exponent must be a finite",
        ),
        ({"maxiter": 10, "generator": 7}, "generator must be a torch.Generator"),
    ],
)
def test_constructor_refusals(settings: dict, message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        SPSAOptimizer(**settings)


@pytest.mark.parametrize(
    ("parameters", "message"),
    [
        ([1.0, 2.0], "parameters must be a torch.Tensor"),
        (torch.tensor([1, 2]), "parameters must have a floating dtype"),
        (torch.empty(0, dtype=torch.float64), "at least one value"),
        (
            torch.tensor([1.0, float("inf")], dtype=torch.float64),
            "parameters must all be finite",
        ),
    ],
)
def test_parameter_refusals(parameters: object, message: str) -> None:
    optimizer = _seeded(1, maxiter=5)
    with pytest.raises(ValidationError, match=message):
        optimizer.estimate_gradient(_quadratic, parameters)  # type: ignore[arg-type]


def test_a_non_callable_objective_is_refused() -> None:
    optimizer = _seeded(1, maxiter=5)
    with pytest.raises(ValidationError, match="objective must be callable"):
        optimizer.estimate_gradient(3, torch.tensor([1.0]))  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("objective", "message"),
    [
        (lambda p: 1.0, "must return a torch.Tensor"),
        (lambda p: p * 2.0, "must return exactly one value"),
        (lambda p: p.reshape(-1, 1).sum(dim=1), "must return exactly one value"),
        (lambda p: p.sum() * float("nan"), "returned a non-finite value"),
    ],
)
def test_objective_result_refusals(objective, message: str) -> None:
    optimizer = _seeded(1, maxiter=5)
    with pytest.raises(ValidationError, match=message):
        optimizer.estimate_gradient(objective, torch.tensor([1.0, 2.0]))


def test_an_objective_that_modifies_its_input_is_refused() -> None:
    """An in-place objective corrupts the second evaluation, so it fails closed."""

    def inplace(parameters: torch.Tensor) -> torch.Tensor:
        parameters.mul_(1.5)
        return (parameters**2).sum()

    optimizer = _seeded(1, maxiter=5)
    parameters = torch.tensor([1.0, 2.0], dtype=torch.float64)
    with pytest.raises(ValidationError, match="modified the tensor it was given"):
        optimizer.estimate_gradient(inplace, parameters)


def test_refusals_happen_before_the_objective_runs() -> None:
    """A refused call spends no evaluation and leaves the counters where they were."""
    objective = _Counter(_quadratic)
    optimizer = _seeded(1, maxiter=5)
    with pytest.raises(ValidationError):
        optimizer.estimate_gradient(objective, torch.tensor([1, 2]))
    assert objective.calls == 0
    assert optimizer.evaluations == 0
    assert optimizer.steps == 0


def test_the_optimizer_preserves_the_parameter_dtype_and_shape() -> None:
    for dtype in (torch.float32, torch.float64):
        optimizer = _seeded(1, maxiter=5)
        parameters = torch.tensor([1.0, 2.0], dtype=dtype)
        updated = optimizer.step(_quadratic, parameters)
        assert updated.dtype is dtype
        assert updated.shape == parameters.shape
        assert optimizer.evaluations == 2


def test_the_optimizer_is_reachable_from_the_algorithms_namespace() -> None:
    """The optimizer is an ``algorithms`` name; a root promotion is a separate change."""
    import flagquantum.algorithms as algorithms

    assert algorithms.SPSAOptimizer is SPSAOptimizer
    assert algorithms.SPSAOptimizer.__module__ == "flagquantum.algorithms.spsa"
    assert "SPSAOptimizer" in algorithms.__all__
