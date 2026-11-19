"""Run the full traffic QAOA with actual finite-shot objectives."""

import math
from dataclasses import dataclass
from typing import cast

import torch

import flagquantum as fq
from examples.synthesis.applications.telecom.network_traffic_optimization.main import (
    build_program,
)
from flagquantum.algorithms import CobylaOptimizer

from . import problem


def initial_parameters() -> tuple[float, ...]:
    return tuple(
        value
        for layer in range(5)
        for value in (
            math.pi * (2 * layer + 1) / 10,
            math.pi * (1 - (2 * layer + 1) / 10),
        )
    )


def physical_to_logical(words: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(words)
    for bit in range(24):
        out |= ((words >> bit) & 1) << (23 - bit)
    return out


def draw(parameters: tuple[float, ...], generator: torch.Generator) -> torch.Tensor:
    program = build_program(parameters)
    state = cast(torch.Tensor, fq.run(program).statevector()).reshape(-1)
    probabilities = state.abs().square()
    physical = torch.multinomial(
        probabilities, problem.SHOTS, replacement=True, generator=generator
    )
    return physical_to_logical(physical)


class _QueryBudgetError(Exception):
    """Stop at the original SciPy transcript's twelve-query minimum."""


@dataclass(frozen=True)
class Result:
    trace: tuple[tuple[tuple[float, ...], float], ...]
    parameters: tuple[float, ...]
    counts: tuple[tuple[int, int], ...]
    feasible: tuple[tuple[int, int], ...]
    classical_word: int
    classical_objective: int
    classical_latency: int
    stopped: bool
    converged: bool


def optimize() -> Result:
    classical_word, objective, latency = problem.classical()
    generator = torch.Generator().manual_seed(10)
    trace: list[tuple[tuple[float, ...], float]] = []

    def evaluate(params: torch.Tensor) -> torch.Tensor:
        if len(trace) >= problem.QUERIES:
            raise _QueryBudgetError()
        parameters = tuple(float(p) for p in params)
        cost = float(problem.values(draw(parameters, generator))[0].mean())
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
    words, counts = torch.unique(draw(parameters, generator), return_counts=True)
    order = torch.argsort(counts, descending=True, stable=True)
    ordered = tuple((int(words[i]), int(counts[i])) for i in order)
    _, flow, overlap = problem.values(words)
    good = {
        int(word)
        for word, one, two in zip(words, flow, overlap, strict=True)
        if not int(one) and not int(two)
    }
    feasible = tuple((word, count) for word, count in ordered if word in good)
    return Result(
        tuple(trace),
        parameters,
        ordered,
        feasible,
        classical_word,
        objective,
        latency,
        stopped,
        converged,
    )
