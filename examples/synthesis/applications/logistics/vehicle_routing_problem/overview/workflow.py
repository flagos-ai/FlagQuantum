"""Optimize and sample the original sixteen-bit routing instance."""

from dataclasses import dataclass

import torch

from flagquantum.algorithms import CobylaOptimizer

from . import problem, quantum


@dataclass(frozen=True)
class Evaluation:
    parameters: tuple[float, ...]
    sampled_cost: float


@dataclass(frozen=True)
class Result:
    problem: problem.Problem
    trace: tuple[Evaluation, ...]
    parameters: tuple[float, ...]
    counts: tuple[tuple[int, int], ...]
    selected_word: int
    best_cost: float
    budget_reached: bool
    converged: bool

    @property
    def sampled_shots(self) -> int:
        return (len(self.trace) + 1) * problem.SHOTS


class _QueryBudgetError(Exception):
    """Stop before exceeding the original optimizer's observed 26 queries."""


def sample(
    data: problem.Problem,
    parameters: tuple[float, ...],
    generator: torch.Generator,
) -> torch.Tensor:
    probabilities = quantum.probabilities(data, parameters)
    return torch.multinomial(
        probabilities, problem.SHOTS, replacement=True, generator=generator
    )


def optimize(data: problem.Problem) -> Result:
    costs = problem.costs(data)
    cost_in_physical_order = quantum.logical_costs(costs)
    trace: list[Evaluation] = []
    generator = torch.Generator().manual_seed(10)

    def evaluate(parameters: torch.Tensor) -> torch.Tensor:
        if len(trace) == problem.OBJECTIVE_QUERIES:
            raise _QueryBudgetError()
        values = tuple(float(value) for value in parameters)
        outcomes = sample(data, values, generator)
        estimate = float(cost_in_physical_order[outcomes].mean())
        trace.append(Evaluation(values, estimate))
        return torch.tensor(estimate, dtype=torch.float64)

    stopped = False
    converged = False
    try:
        optimum = CobylaOptimizer(maxiter=100, rhobeg=0.7, rhoend=1e-8).minimize(
            evaluate, torch.tensor(quantum.initial_parameters(), dtype=torch.float64)
        )
        converged = optimum.converged
    except _QueryBudgetError:
        stopped = True
    # Preserve the best actually evaluated sample mean.
    parameters = min(trace, key=lambda value: value.sampled_cost).parameters
    final = sample(data, parameters, generator)
    physical_counts = torch.bincount(final, minlength=1 << problem.WIDTH)
    counts = []
    for word, count in enumerate(physical_counts):
        if int(count):
            logical = int(f"{word:016b}"[::-1], 2)
            counts.append((logical, int(count)))
    ordered = tuple(sorted(counts, key=lambda row: (float(costs[row[0]]), row[0])))
    best = ordered[0][0]
    return Result(
        data,
        tuple(trace),
        parameters,
        ordered,
        best,
        float(costs[best]),
        stopped,
        converged,
    )
