"""Budget native COBYLA by actual three-basis, thousand-shot objective queries."""

import math
from dataclasses import dataclass
from typing import Literal

import torch

from flagquantum.algorithms import CobylaOptimizer

from .measurements import Estimate, Measurement, estimate, sample
from .quantum import distribution

INITIAL_PARAMETERS = tuple(math.pi * i / 5 for i in range(6))


@dataclass(frozen=True)
class Evaluation:
    parameters: tuple[float, ...]
    estimate: Estimate


@dataclass(frozen=True)
class Workflow:
    trace: tuple[Evaluation, ...]
    selected: Evaluation
    termination: Literal["evaluation_budget", "optimizer_returned"]
    converged: bool
    final_y: Measurement
    positions: tuple[Estimate, Estimate]


class _QueryBudgetError(Exception):
    """Stop before an objective would exceed the source evaluation cap."""


def run_workflow(*, evaluation_budget: int = 30, seed: int = 2718) -> Workflow:
    """Every objective executes three native circuits and draws 1000shots per basis.

    Native COBYLA replaces SciPy COBYLA; retain the original evaluation cap and
    select the best actually observed query without trajectory equivalence.
    Independent final Y samples and two position estimates follow optimization.
    """
    if type(evaluation_budget) is not int or not 1 <= evaluation_budget <= 30:
        raise ValueError("evaluation budget must lie in 1..30")
    if type(seed) is not int or not 0 <= seed < 2**63 - 128:
        raise ValueError("seed must leave room for all source measurement batches")
    trace: list[Evaluation] = []

    def evaluate(parameters: torch.Tensor) -> torch.Tensor:
        if len(trace) >= evaluation_budget:
            raise _QueryBudgetError()
        values = tuple(float(value) for value in parameters)
        reading = estimate(values, seed + 3 * len(trace))
        trace.append(Evaluation(values, reading))
        return torch.tensor(reading.cost, dtype=torch.float64)

    termination: Literal["evaluation_budget", "optimizer_returned"] = (
        "optimizer_returned"
    )
    converged = False
    try:
        result = CobylaOptimizer(maxiter=100, rhobeg=1.0, rhoend=1e-8).minimize(
            evaluate, torch.tensor(INITIAL_PARAMETERS, dtype=torch.float64)
        )
        converged = result.converged
    except _QueryBudgetError:
        termination = "evaluation_budget"
    selected = min(trace, key=lambda value: value.estimate.cost)
    final_y = sample(distribution(selected.parameters, "Y"), "Y", seed + 100)
    positions = estimate(selected.parameters, seed + 101), estimate(
        selected.parameters, seed + 104
    )
    return Workflow(tuple(trace), selected, termination, converged, final_y, positions)
