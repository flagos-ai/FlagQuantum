"""Run the original-size finite-sample QAOA optimization and final packing draw."""

import math
from dataclasses import dataclass

import torch

from flagquantum import synth
from flagquantum.algorithms import CobylaOptimizer

from .model import PackingModel, Placement, make_model, unpack
from .quantum import LAYERS, probabilities, program_builder

MAX_QUERIES = 60
SHOTS = 10000


class _QueryBudgetError(Exception):
    """Stop before a query would exceed the stated evaluation budget."""


@dataclass(frozen=True)
class Evaluation:
    """The sampled mean and the actual angles used to obtain it."""

    mean_cost: float
    angles: tuple[float, ...]


@dataclass(frozen=True)
class PackingRun:
    """Actual sampled result and the separate classical optimum."""

    model: PackingModel
    evaluations: tuple[Evaluation, ...]
    selected: Evaluation
    sampled: tuple[tuple[int, int], ...]
    word: int
    placements: tuple[Placement, ...]
    exact_cost: float
    converged: bool
    budget_reached: bool
    program: synth.Program


def initial_angles(model: PackingModel) -> tuple[float, ...]:
    """Scale the ten-layer ramp by the largest half of nonconstant term magnitudes."""
    weights = sorted(abs(coefficient) for _, coefficient in model.terms[1:])
    large = weights[len(weights) // 2 :]
    step = len(large) / (2 * math.fsum(large))
    last = LAYERS - 1
    return tuple(
        value
        for layer in range(LAYERS)
        for value in (step * layer / last, step * (last - layer) / last)
    )


def run_workflow() -> PackingRun:
    """Minimize the actual sampled mean and select only from the final observed words."""
    model = make_model()
    build = program_builder(model)
    evaluations: list[Evaluation] = []
    rng = torch.Generator().manual_seed(0)

    def objective(angles: torch.Tensor) -> torch.Tensor:
        if len(evaluations) >= MAX_QUERIES:
            raise _QueryBudgetError
        binding = tuple(map(float, angles))
        draw = torch.multinomial(
            probabilities(build(binding)), SHOTS, replacement=True, generator=rng
        )
        mean = model.cost[draw].mean()
        evaluations.append(Evaluation(float(mean), binding))
        return mean

    budget_reached = False
    try:
        optimized = CobylaOptimizer(
            maxiter=MAX_QUERIES, rhobeg=1.0, rhoend=1e-8
        ).minimize(objective, torch.tensor(initial_angles(model), dtype=torch.float64))
        converged = optimized.converged
    except _QueryBudgetError:
        budget_reached = True
        converged = False
    chosen = min(evaluations, key=lambda row: row.mean_cost)
    program = build(chosen.angles)
    draw = torch.multinomial(
        probabilities(program),
        SHOTS,
        replacement=True,
        generator=torch.Generator().manual_seed(7),
    )
    counts = torch.bincount(draw, minlength=model.cost.numel())
    sampled = tuple(
        (int(word), int(counts[word])) for word in counts.nonzero().flatten()
    )
    word, _ = min(sampled, key=lambda row: (float(model.cost[row[0]]), -row[1], row[0]))
    return PackingRun(
        model,
        tuple(evaluations),
        chosen,
        sampled,
        word,
        unpack(model, word),
        float(model.cost.min()),
        converged,
        budget_reached,
        program,
    )
