"""Retain finite-shot objectives and independent optimized samples for both mixers."""

import math
from dataclasses import dataclass

import torch

from examples.synthesis.graph_optimization import Evaluation
from flagquantum import synth
from flagquantum.algorithms import CobylaOptimizer

from .quantum import build_program, distribution

SHOTS = 1000
BUDGET = 30
INITIAL = (0.0, math.pi / 2, math.pi, math.pi, math.pi / 2, 0.0)


class _ObjectiveBudgetError(Exception):
    """Stop before the 31st objective execution."""


def cost(word: int, *, grover: bool) -> int:
    counts = tuple((word >> i) & 1 for i in range(4))
    value = counts[0] + 2 * counts[1] - counts[2] + 12 * counts[3]
    return value if grover else value + (word.bit_count() - 1) ** 2


@dataclass(frozen=True)
class Observation:
    word: int
    count: int
    cost: int
    one_hot: bool


@dataclass(frozen=True)
class Run:
    program: synth.Program
    trace: tuple[Evaluation, ...]
    parameters: tuple[float, ...]
    observations: tuple[Observation, ...]
    budget_reached: bool
    converged: bool


def draw(program: synth.Program, generator: torch.Generator) -> torch.Tensor:
    masses = torch.tensor(distribution(program), dtype=torch.float64)
    return torch.multinomial(masses, SHOTS, replacement=True, generator=generator)


def optimize(*, grover: bool) -> Run:
    stream = torch.Generator().manual_seed(811 if grover else 810)
    scores = torch.tensor(
        [cost(word, grover=grover) for word in range(16)], dtype=torch.float64
    )
    trace: list[Evaluation] = []

    def evaluate(vector: torch.Tensor) -> torch.Tensor:
        if len(trace) >= BUDGET:
            raise _ObjectiveBudgetError
        p = tuple(float(value) for value in vector)
        sampled = draw(build_program(p, grover=grover), stream)
        result = scores[sampled].mean()
        trace.append(Evaluation(p, float(result)))
        return result

    budget_reached = converged = False
    try:
        optimized = CobylaOptimizer(maxiter=100, rhobeg=1.0, rhoend=1e-8).minimize(
            evaluate, torch.tensor(INITIAL, dtype=torch.float64)
        )
        converged = optimized.converged
    except _ObjectiveBudgetError:
        budget_reached = True
    best = min(trace, key=lambda row: row.cost)
    program = build_program(best.parameters, grover=grover)
    selected = draw(program, torch.Generator().manual_seed(1011 if grover else 1010))
    counts = torch.bincount(selected, minlength=16).tolist()
    rows = tuple(
        Observation(word, int(count), cost(word, grover=grover), word.bit_count() == 1)
        for word, count in enumerate(counts)
        if count
    )
    return Run(program, tuple(trace), best.parameters, rows, budget_reached, converged)
