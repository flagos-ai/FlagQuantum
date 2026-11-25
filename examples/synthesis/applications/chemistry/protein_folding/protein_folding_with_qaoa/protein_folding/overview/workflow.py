"""Run the finite-shot optimization and solve all 128 classical assignments."""

import math
from dataclasses import dataclass

import torch

from flagquantum.algorithms import CobylaOptimizer

from .model import costs
from .quantum import LAYERS, build_program, probabilities

SHOTS = 2048
MAX_QUERIES = 70
QUANTILE = 0.7


class _QueryBudgetError(Exception):
    """Stop before a callback would exceed the sample budget."""


@dataclass(frozen=True)
class Evaluation:
    angles: tuple[float, ...]
    lower_tail_mean: float


@dataclass(frozen=True)
class Result:
    energies: tuple[int, ...]
    evaluations: tuple[Evaluation, ...]
    optimized: torch.Tensor
    uniform: torch.Tensor
    observed_word: int
    exact_words: tuple[int, ...]
    converged: bool
    budget_reached: bool


def lower_tail_mean(samples: torch.Tensor) -> torch.Tensor:
    """Compute the lower 70 percent mean with a fractional boundary sample."""
    ordered = samples.sort().values
    mass = QUANTILE * ordered.numel()
    whole = int(mass)
    return (ordered[:whole].sum() + (mass - whole) * ordered[whole]) / mass


def run_workflow() -> Result:
    """Sample every objective query and compare with exact finite enumeration."""
    energies = costs()
    table = torch.tensor(energies, dtype=torch.float64)
    generator = torch.Generator().manual_seed(42)
    evaluations: list[Evaluation] = []
    ramp = tuple(math.pi * layer / LAYERS for layer in range(1, LAYERS + 1))

    def draw(angles: tuple[float, ...], rng: torch.Generator) -> torch.Tensor:
        return torch.multinomial(
            probabilities(build_program(angles)), SHOTS, replacement=True, generator=rng
        )

    def objective(parameters: torch.Tensor) -> torch.Tensor:
        if len(evaluations) == MAX_QUERIES:
            raise _QueryBudgetError
        angles = tuple(map(float, parameters))
        score = lower_tail_mean(table[draw(angles, generator)])
        evaluations.append(Evaluation(angles, float(score)))
        return score

    budget_reached = False
    try:
        result = CobylaOptimizer(maxiter=MAX_QUERIES, rhobeg=1.0, rhoend=1e-8).minimize(
            objective, torch.tensor((*ramp, *reversed(ramp)), dtype=torch.float64)
        )
        converged = result.converged
    except _QueryBudgetError:
        converged, budget_reached = False, True

    optimized = draw(evaluations[-1].angles, torch.Generator().manual_seed(7))
    uniform = draw((0.0,) * (2 * LAYERS), torch.Generator().manual_seed(8))
    observed_word = min(
        set(map(int, optimized)), key=lambda word: (energies[word], word)
    )
    minimum = min(energies)
    return Result(
        energies,
        tuple(evaluations),
        optimized,
        uniform,
        observed_word,
        tuple(word for word, value in enumerate(energies) if value == minimum),
        converged,
        budget_reached,
    )
