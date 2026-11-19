"""Optimize and sample the actual selected named circuit on the full LABS grid."""

import math
from dataclasses import dataclass

import torch

import flagquantum as fq
from flagquantum.algorithms import CobylaOptimizer

from . import model, quantum

BUDGET = 60
SHOTS = 1000
RAMP = tuple(math.pi * layer / model.LAYERS for layer in range(1, model.LAYERS + 1))


class QueryLimitError(Exception):
    """Stop before executing a call beyond the declared objective budget."""


@dataclass(frozen=True)
class Query:
    parameters: tuple[float, ...]
    cost: float


@dataclass(frozen=True)
class Sample:
    word: int
    sequence: tuple[int, ...]
    cost: int
    count: int


@dataclass(frozen=True)
class Result:
    candidates: tuple[quantum.Candidate, ...]
    selected: quantum.Candidate
    trace: tuple[Query, ...]
    chosen: Query
    samples: tuple[Sample, ...]
    exact_minimum: int
    exact_words: tuple[int, ...]
    converged: bool
    budget_stop: bool


def probabilities(
    candidate: quantum.Candidate, parameters: tuple[float, ...]
) -> torch.Tensor:
    outcome = fq.run(quantum.bind(candidate, parameters))
    physical = torch.as_tensor(outcome.statevector()).reshape((2,) * model.WIDTH)
    logical = physical.permute(tuple(reversed(range(model.WIDTH)))).flatten()
    return logical.abs().square()


def draw(
    costs: tuple[int, ...], distribution: torch.Tensor, shots: int, seed: int
) -> tuple[Sample, ...]:
    observations = torch.multinomial(
        distribution,
        shots,
        replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )
    counts = torch.bincount(observations, minlength=len(costs))
    return tuple(
        sorted(
            (
                Sample(
                    word,
                    tuple((word >> bit) & 1 for bit in range(model.WIDTH)),
                    costs[word],
                    int(count),
                )
                for word, count in enumerate(counts)
                if int(count)
            ),
            key=lambda item: (item.cost, item.word),
        )
    )


def run(*, budget: int = BUDGET, shots: int = SHOTS) -> Result:
    """Keep the configured source depth/budget and execute every queried angle."""
    if type(budget) is not int or not 1 <= budget <= BUDGET:
        raise ValueError("The actual objective budget must be in 1..60")
    if type(shots) is not int or not 1 <= shots <= 100_000:
        raise ValueError("The final finite sample count must be in 1..100000")
    costs = tuple(model.source_cost(word) for word in range(1 << model.WIDTH))
    weights = torch.tensor(costs, dtype=torch.float64)
    candidates, selected = quantum.choose()
    history: list[Query] = []

    def objective(parameters: torch.Tensor) -> torch.Tensor:
        if len(history) == budget:
            raise QueryLimitError
        bound = tuple(map(float, parameters))
        distribution = probabilities(selected, bound)
        mean = torch.dot(distribution, weights)
        history.append(Query(bound, float(mean)))
        return mean

    budget_stop = False
    try:
        result = CobylaOptimizer(maxiter=budget, rhobeg=1.0, rhoend=1e-8).minimize(
            objective,
            torch.tensor((*RAMP, *reversed(RAMP)), dtype=torch.float64),
        )
        converged = result.converged
    except QueryLimitError:
        budget_stop = True
        converged = False
    chosen = min(history, key=lambda row: row.cost)
    samples = draw(costs, probabilities(selected, chosen.parameters), shots, 2718)
    minimum = min(costs)
    minimizers = tuple(word for word, energy in enumerate(costs) if energy == minimum)
    return Result(
        candidates,
        selected,
        tuple(history),
        chosen,
        samples,
        minimum,
        minimizers,
        converged,
        budget_stop,
    )
