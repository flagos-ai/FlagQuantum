"""Optimize both original schedules and compare the observed and classical plans."""

from dataclasses import dataclass
from typing import cast

import torch

import flagquantum as fq
from flagquantum.algorithms import CobylaOptimizer

from . import problem, quantum

SHOTS = 2048


@dataclass(frozen=True)
class Evaluation:
    parameters: tuple[float, ...]
    quantile_cost: float


@dataclass(frozen=True)
class Result:
    problem: problem.Problem
    quantile: float
    trace: tuple[Evaluation, ...]
    parameters: tuple[float, ...]
    counts: tuple[tuple[int, int], ...]
    selected: int
    selected_cost: float
    selected_matrix: tuple[tuple[int, ...], ...]
    selected_valid: bool
    classical_cost: int
    classical_matrix: tuple[tuple[int, ...], ...]
    stopped: bool
    converged: bool


class _ObjectiveBudgetError(Exception):
    """Stop before the source's explicit evaluation budget is exceeded."""


def distribution(
    problem_: problem.Problem, parameters: tuple[float, ...]
) -> torch.Tensor:
    native = quantum.build(problem_, parameters)
    state = cast(torch.Tensor, fq.run(native).statevector()).reshape(-1)
    return state.abs().square()


def lower_quantile(
    problem_: problem.Problem, draws: torch.Tensor, quantile: float
) -> float:
    costs = problem.costs(problem_, draws).sort().values
    mass = quantile * draws.numel()
    full = int(mass)
    partial = mass - full
    total = costs[:full].sum()
    if partial:
        total += partial * costs[full]
    return float(total / mass)


def logical(words: torch.Tensor, width: int) -> torch.Tensor:
    out = torch.zeros_like(words)
    for bit in range(width):
        out |= ((words >> bit) & 1) << (width - 1 - bit)
    return out


def optimize(large: bool) -> Result:
    model = problem.create(large)
    max_queries = 1 if large else 20
    quantile = 0.6 if large else 0.3
    generator = torch.Generator().manual_seed(10)
    trace: list[Evaluation] = []

    def draw(parameters: tuple[float, ...]) -> torch.Tensor:
        physical = torch.multinomial(
            distribution(model, parameters),
            SHOTS,
            replacement=True,
            generator=generator,
        )
        return logical(physical, model.width)

    def evaluate(params: torch.Tensor) -> torch.Tensor:
        if len(trace) >= max_queries:
            raise _ObjectiveBudgetError()
        parameters = tuple(float(v) for v in params)
        value = lower_quantile(model, draw(parameters), quantile)
        trace.append(Evaluation(parameters, value))
        return torch.tensor(value, dtype=torch.float64)

    stopped = converged = False
    try:
        optimized = CobylaOptimizer(maxiter=100, rhobeg=1, rhoend=1e-8).minimize(
            evaluate,
            torch.tensor(quantum.initial_parameters(model), dtype=torch.float64),
        )
        converged = optimized.converged
    except _ObjectiveBudgetError:
        stopped = True
    parameters = min(trace, key=lambda row: row.quantile_cost).parameters
    words, counts = torch.unique(draw(parameters), return_counts=True)
    values = problem.costs(model, words)
    indices = sorted(
        range(words.numel()), key=lambda i: (float(values[i]), int(words[i]))
    )
    observations = tuple((int(words[i]), int(counts[i])) for i in indices)
    selected = observations[0][0]
    matrix = problem.decode(model, selected)
    classical_cost, classical_matrix = problem.classical(model)
    return Result(
        model,
        quantile,
        tuple(trace),
        parameters,
        observations,
        selected,
        problem.original_cost(model, selected),
        matrix,
        problem.valid(model, matrix),
        classical_cost,
        classical_matrix,
        stopped,
        converged,
    )
