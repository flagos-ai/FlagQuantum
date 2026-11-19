"""Run original-size resiliency circuits with actual local finite sampling."""

import math
from dataclasses import dataclass
from typing import cast

import torch

import flagquantum as fq
from examples.synthesis.applications.telecom.resiliency_planning.main import (
    build_program,
)
from flagquantum.algorithms import CobylaOptimizer

from . import problem


def initial_parameters() -> tuple[float, ...]:
    return tuple(
        value
        for layer in range(20)
        for value in (
            math.pi * (2 * layer + 1) / 40,
            math.pi * (1 - (2 * layer + 1) / 40),
        )
    )


def physical_to_logical(words: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(words)
    for bit in range(24):
        out |= ((words >> bit) & 1) << (23 - bit)
    return out


def draw(
    parameters: tuple[float, ...], shots: int, generator: torch.Generator
) -> torch.Tensor:
    program = build_program(parameters)
    state = cast(torch.Tensor, fq.run(program).statevector()).reshape(-1)
    physical = torch.multinomial(
        state.abs().square(), shots, replacement=True, generator=generator
    )
    return physical_to_logical(physical)


class _QueryBudgetError(Exception):
    """Stop before any objective query beyond the explicit local budget."""


@dataclass(frozen=True)
class Result:
    trace: tuple[tuple[tuple[float, ...], float], ...]
    parameters: tuple[float, ...]
    counts: tuple[tuple[int, int], ...]
    feasible: tuple[tuple[int, int], ...]
    objective_shots: int
    final_shots: int
    stopped: bool
    converged: bool


def optimize(
    *, queries: int = 1, objective_shots: int = 20_000, final_shots: int = 20_000
) -> Result:
    if any(
        type(value) is not int or value <= 0
        for value in (queries, objective_shots, final_shots)
    ):
        raise ValueError("Query and sample budgets must be positive integers")
    generator = torch.Generator().manual_seed(10)
    trace: list[tuple[tuple[float, ...], float]] = []

    def evaluate(params: torch.Tensor) -> torch.Tensor:
        if len(trace) >= queries:
            raise _QueryBudgetError()
        parameters = tuple(float(p) for p in params)
        cost = float(
            problem.values(draw(parameters, objective_shots, generator))[0].mean()
        )
        trace.append((parameters, cost))
        return torch.tensor(cost, dtype=torch.float64)

    stopped = converged = False
    try:
        result = CobylaOptimizer(maxiter=100, rhobeg=1.0, rhoend=1e-8).minimize(
            evaluate, torch.tensor(initial_parameters(), dtype=torch.float64)
        )
        converged = result.converged
    except _QueryBudgetError:
        stopped = True
    parameters = min(trace, key=lambda row: row[1])[0]
    words, counts = torch.unique(
        draw(parameters, final_shots, generator), return_counts=True
    )
    order = torch.argsort(counts, descending=True, stable=True)
    ordered = tuple((int(words[i]), int(counts[i])) for i in order)
    _, flow, overlap = problem.values(words)
    good = {
        int(word)
        for word, first, second in zip(words, flow, overlap, strict=True)
        if not int(first) and not int(second)
    }
    return Result(
        tuple(trace),
        parameters,
        ordered,
        tuple((word, count) for word, count in ordered if word in good),
        objective_shots,
        final_shots,
        stopped,
        converged,
    )
