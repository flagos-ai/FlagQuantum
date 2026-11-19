"""Bounded local QAOA optimization, actual draws and source feasibility metrics."""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch

import flagquantum as fq
from flagquantum import synth
from flagquantum.algorithms import CobylaOptimizer

from .ansatz import feasible, maxcut, penalty
from .model import (
    INITIAL,
    Model,
    Parameters,
    feasible_value,
    optimization_cost,
    value,
    weight,
)


@dataclass(frozen=True)
class Evaluation:
    parameters: Parameters
    cost: float


@dataclass(frozen=True)
class Observation:
    values: tuple[int, ...]
    count: int
    probability: float


@dataclass(frozen=True)
class Statistics:
    feasibility_probability: float
    average_feasible_value: float
    overall_score: float
    actual_shots: int


@dataclass(frozen=True)
class Optimization:
    model: Model
    evaluations: tuple[Evaluation, ...]
    selected: Evaluation
    converged: bool
    termination: str
    statistics_samples: tuple[Observation, ...]
    table_samples: tuple[Observation, ...]


class EvaluationBudgetError(Exception):
    """Stop the local optimizer before an additional quantum objective call."""


def parameters(values: Sequence[float]) -> Parameters:
    if len(values) != 6 or any(not math.isfinite(v) for v in values):
        raise ValueError("use six finite angles")
    return (
        float(values[0]),
        float(values[1]),
        float(values[2]),
        float(values[3]),
        float(values[4]),
        float(values[5]),
    )


def build_program(model: Model, angles: Parameters) -> synth.Program:
    return synth.synthesize(
        {"maxcut": maxcut, "penalty": penalty, "feasible": feasible}[model],
        classical_inputs={"params": list(parameters(angles))},
        memory_limit_bytes=64 * 1024**2,
    )


def distribution(program: synth.Program) -> Mapping[tuple[int, ...], float]:
    return program.probabilities(fq.run(program), registers=tuple(program.registers))


def mean_cost(model: Model, values: Mapping[tuple[int, ...], float]) -> float:
    return sum(
        probability * optimization_cost(model, state)
        for state, probability in values.items()
    )


def sample(
    values: Mapping[tuple[int, ...], float], *, shots: int, seed: int
) -> tuple[Observation, ...]:
    if type(shots) is not int or not 1 <= shots <= 100000:
        raise ValueError("shots must lie in 1..100000")
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("seed must be a nonnegative 63-bit integer")
    if (
        not values
        or any(not math.isfinite(p) or p < 0 for p in values.values())
        or abs(sum(values.values()) - 1) > 1e-10
    ):
        raise ValueError("use finite normalized probabilities")
    keys = tuple(values)
    counts = torch.bincount(
        torch.multinomial(
            torch.tensor(tuple(values.values()), dtype=torch.float64),
            shots,
            replacement=True,
            generator=torch.Generator().manual_seed(seed),
        ),
        minlength=len(keys),
    )
    return tuple(
        Observation(state, int(counts[i]), int(counts[i]) / shots)
        for i, state in enumerate(keys)
        if int(counts[i])
    )


def statistics(observations: tuple[Observation, ...]) -> Statistics:
    shots = sum(row.count for row in observations)
    if not shots:
        raise ValueError("statistics need actual observations")
    selected = tuple(row for row in observations if weight(*row.values[:2]) <= 12)
    feasible_shots = sum(row.count for row in selected)
    total = sum(value(*row.values[:2]) * row.count for row in selected)
    return Statistics(
        feasible_shots / shots,
        total / feasible_shots if feasible_shots else 0.0,
        sum(feasible_value(*row.values[:2]) * row.count for row in observations)
        / shots,
        shots,
    )


def optimize(model: Model, *, seed: int = 1970) -> Optimization:
    budget = 40 if model == "maxcut" else 50
    evaluations: list[Evaluation] = []
    latest_values: Mapping[tuple[int, ...], float] = {}

    def objective(raw: torch.Tensor) -> torch.Tensor:
        nonlocal latest_values
        if len(evaluations) >= budget:
            raise EvaluationBudgetError
        angles = parameters(tuple(float(v) for v in raw))
        values = distribution(build_program(model, angles))
        row = Evaluation(angles, mean_cost(model, values))
        evaluations.append(row)
        latest_values = values
        return torch.tensor(row.cost, dtype=torch.float64)

    try:
        result = CobylaOptimizer(maxiter=200, rhobeg=1.0, rhoend=1e-8).minimize(
            objective, torch.tensor(INITIAL, dtype=torch.float64)
        )
    except EvaluationBudgetError:
        converged, termination = False, "evaluation_budget"
    else:
        converged = result.converged
        termination = "optimizer_returned"
    # The source reads its final trace entry, so preserve the last actual query.
    selected = evaluations[-1]
    values = latest_values
    return Optimization(
        model,
        tuple(evaluations),
        selected,
        converged,
        termination,
        sample(values, shots=1000, seed=seed),
        sample(values, shots=1000 if model == "maxcut" else 10, seed=seed + 1),
    )
