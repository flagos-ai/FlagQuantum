"""Perform the entire sampled low-tail optimization and independent route solve."""

import math
from dataclasses import dataclass

import torch

from flagquantum.algorithms import CobylaOptimizer

from .model import RouteModel, exact_path, make_model, route
from .quantum import LAYERS, build_program, probabilities

SHOTS = 2048
QUANTILE = 0.6
MAX_QUERIES = 150


class _QueryBudgetError(Exception):
    """Stop before the optimizer would exceed the source query budget."""


@dataclass(frozen=True)
class Evaluation:
    """Actual finite-shot low-tail mean and the evaluated angles."""

    angles: tuple[float, ...]
    value: float


@dataclass(frozen=True)
class Workflow:
    """The observed optimized and uniform samples plus the independent exact route."""

    model: RouteModel
    evaluations: tuple[Evaluation, ...]
    optimized_draw: torch.Tensor
    uniform_draw: torch.Tensor
    best_word: int
    best_route: tuple[int, ...] | None
    exact: tuple[float, tuple[int, ...]]
    converged: bool
    budget_reached: bool


def low_tail_mean(values: torch.Tensor, quantile: float = QUANTILE) -> torch.Tensor:
    """Average exactly the requested lower fraction, including boundary mass."""
    if values.ndim != 1 or not values.numel() or not 0 < quantile <= 1:
        raise ValueError(
            "Supply a nonempty observation vector and a quantile in (0, 1]"
        )
    sorted_values = values.sort().values
    mass = values.numel() * quantile
    whole = int(mass)
    total = sorted_values[:whole].sum()
    if whole < values.numel():
        total = total + (mass - whole) * sorted_values[whole]
    return torch.div(total, mass)


def _draw(
    model: RouteModel, angles: tuple[float, ...], rng: torch.Generator
) -> torch.Tensor:
    weight = probabilities(build_program(model, angles))
    return torch.multinomial(weight, SHOTS, replacement=True, generator=rng)


def run_workflow() -> Workflow:
    """Use the original pi ramp, all 150 queries, and both final sample runs."""
    model = make_model()
    rng = torch.Generator().manual_seed(42)
    history: list[Evaluation] = []
    ramp = tuple(math.pi * layer / LAYERS for layer in range(1, LAYERS + 1))

    def objective(parameters: torch.Tensor) -> torch.Tensor:
        if len(history) == MAX_QUERIES:
            raise _QueryBudgetError
        angles = tuple(map(float, parameters))
        draw = _draw(model, angles, rng)
        value = low_tail_mean(model.cost[draw])
        history.append(Evaluation(angles, float(value)))
        return value

    budget_reached = False
    try:
        result = CobylaOptimizer(maxiter=MAX_QUERIES, rhobeg=1.0, rhoend=1e-8).minimize(
            objective, torch.tensor((*ramp, *reversed(ramp)), dtype=torch.float64)
        )
        converged = result.converged
    except _QueryBudgetError:
        budget_reached = True
        converged = False
    optimized = _draw(model, history[-1].angles, torch.Generator().manual_seed(7))
    uniform = _draw(model, (0.0,) * (2 * LAYERS), torch.Generator().manual_seed(8))
    words = set(map(int, optimized))
    best = min(words, key=lambda word: (float(model.cost[word]), word))
    return Workflow(
        model,
        tuple(history),
        optimized,
        uniform,
        best,
        route(model, best),
        exact_path(model),
        converged,
        budget_reached,
    )
