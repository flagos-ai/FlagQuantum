"""Original three-asset portfolio model, bounded native optimization and readout."""

import math
from dataclasses import dataclass
from itertools import product
from typing import TYPE_CHECKING

import torch

import flagquantum as fq
from flagquantum import synth
from flagquantum.algorithms import CobylaOptimizer, CobylaResult

from ..main import DEFAULT_PARAMS
from .operations import build_program

if TYPE_CHECKING:
    from matplotlib.figure import Figure

RETURNS = (3.0, 4.0, -1.0)
COVARIANCE = ((0.9, 0.5, -0.7), (0.5, 0.9, -0.2), (-0.7, -0.2, 0.9))
Weights = tuple[int, int, int]
Assignment = tuple[int, int, int, int]


@dataclass(frozen=True)
class Evaluation:
    parameters: tuple[float, ...]
    cvar: float


@dataclass(frozen=True)
class Sample:
    weights: Weights
    cost: float
    feasible: bool
    count: int
    probability: float


@dataclass(frozen=True)
class Workflow:
    program: synth.Program
    optimizer: CobylaResult
    trace: tuple[Evaluation, ...]
    optimized: tuple[Sample, ...]
    uniform: tuple[Sample, ...]

    @property
    def best_observed(self) -> Sample | None:
        return next((row for row in self.optimized if row.feasible), None)


def objective(weights: Weights) -> float:
    """Retain the complete covariance quadratic form minus expected return."""
    if len(weights) != 3 or any(type(w) is not int or not 0 <= w <= 7 for w in weights):
        raise ValueError("three integer holdings must lie in 0..7")
    risk = sum(
        COVARIANCE[i][j] * weights[i] * weights[j] for i in range(3) for j in range(3)
    )
    returns = sum(r * w for r, w in zip(RETURNS, weights, strict=True))
    return risk - returns


def encoded_cost(assignment: Assignment) -> float:
    """Preserve the source's bounded slack weights 1, 2 and 3 and penalty 10."""
    a, b, c, slack_word = assignment
    if type(slack_word) is not int or not 0 <= slack_word < 8:
        raise ValueError("slack must be a three-bit word")
    slack = sum(
        weight * ((slack_word >> bit) & 1) for bit, weight in enumerate((1, 2, 3))
    )
    return objective((a, b, c)) + 10 * (a + b + c + slack - 6) ** 2


def classical_optima() -> tuple[tuple[Weights, float], ...]:
    """Enumerate all 512 original holdings, independently of any quantum samples."""
    holdings = ((a, b, c) for a, b, c in product(range(8), repeat=3))
    feasible = tuple((w, objective(w)) for w in holdings if sum(w) <= 6)
    minimum = min(cost for _, cost in feasible)
    return tuple(
        (w, cost) for w, cost in feasible if math.isclose(cost, minimum, abs_tol=1e-12)
    )


def distribution(program: synth.Program) -> dict[Assignment, float]:
    """Read all four named registers, including the bounded slack word."""
    readout = program.probabilities(
        fq.run(program), registers=("a0", "a1", "a2", "slack"), threshold=0
    )
    values: dict[Assignment, float] = {}
    for key, probability in readout.items():
        if len(key) != 4:
            raise ValueError("portfolio readout must contain four declared registers")
        assignment = (key[0], key[1], key[2], key[3])
        encoded_cost(assignment)
        values[assignment] = probability
    if abs(sum(values.values()) - 1) > 1e-10 or any(
        not math.isfinite(p) or p < 0 for p in values.values()
    ):
        raise ValueError(
            "portfolio probabilities must be finite, nonnegative and normalized"
        )
    return values


def cvar(probabilities: dict[Assignment, float], quantile: float = 0.7) -> float:
    """Integrate the lowest cost probability mass, including the fractional boundary."""
    if not math.isfinite(quantile) or not 0 < quantile <= 1:
        raise ValueError("quantile must lie in (0, 1]")
    if abs(sum(probabilities.values()) - 1) > 1e-10 or any(
        not math.isfinite(p) or p < 0 for p in probabilities.values()
    ):
        raise ValueError("CVaR requires normalized probabilities")
    remaining = quantile
    total = 0.0
    for cost, probability in sorted(
        (encoded_cost(key), probability) for key, probability in probabilities.items()
    ):
        taken = min(remaining, probability)
        total += taken * cost
        remaining -= taken
        if remaining <= 0:
            return total / quantile
    if remaining > 1e-10:
        raise ValueError("insufficient probability mass for the requested quantile")
    return total / quantile


def sample(
    probabilities: dict[Assignment, float], *, shots: int, seed: int
) -> tuple[Sample, ...]:
    """Draw actual local assignments, marginalize slack and rank feasible holdings."""
    if type(shots) is not int or not 1 <= shots <= 100000:
        raise ValueError("shots must lie in 1..100000")
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("seed must be a nonnegative 63-bit integer")
    cvar(probabilities, 1.0)
    marginal: dict[Weights, float] = {}
    for (a, b, c, _), probability in probabilities.items():
        w = (a, b, c)
        marginal[w] = marginal.get(w, 0.0) + probability
    assignments = tuple(sorted(marginal))
    weights = torch.tensor([marginal[w] for w in assignments], dtype=torch.float64)
    draws = torch.multinomial(
        weights, shots, replacement=True, generator=torch.Generator().manual_seed(seed)
    )
    counts = torch.bincount(draws, minlength=len(assignments))
    rows = (
        Sample(w, objective(w), sum(w) <= 6, int(counts[i]), marginal[w])
        for i, w in enumerate(assignments)
        if int(counts[i]) > 0
    )
    return tuple(
        sorted(rows, key=lambda row: (not row.feasible, row.cost, row.weights))
    )


def run_workflow(*, maxiter: int = 60, shots: int = 1000, seed: int = 1914) -> Workflow:
    """Run native COBYLA against exact local CVaR, then sample its returned program.

    The source leaves its initial schedule, shot count and optimizer details
    implicit. This workflow declares them explicitly; it does not claim the
    hosted optimizer's trajectory or stochastic estimator is reproduced.
    """
    if type(maxiter) is not int or not 1 <= maxiter <= 128:
        raise ValueError("maxiter must lie in 1..128")
    assignments = ((a, b, c, slack) for a, b, c, slack in product(range(8), repeat=4))
    uniform = dict.fromkeys(assignments, 1 / 4096)
    baseline = sample(uniform, shots=shots, seed=seed + 1)
    trace: list[Evaluation] = []

    def evaluate(parameters: torch.Tensor) -> torch.Tensor:
        values = tuple(float(v) for v in parameters)
        cost = cvar(distribution(build_program(values)), 0.7)
        trace.append(Evaluation(values, cost))
        return torch.tensor(cost, dtype=torch.float64)

    optimizer = CobylaOptimizer(maxiter=maxiter, rhobeg=0.07, rhoend=1e-6).minimize(
        evaluate, torch.tensor(DEFAULT_PARAMS, dtype=torch.float64)
    )
    program = build_program(tuple(float(v) for v in optimizer.parameters))
    final = sample(distribution(program), shots=shots, seed=seed)
    return Workflow(program, optimizer, tuple(trace), final, baseline)


def plot_trace(workflow: Workflow) -> "Figure":
    from matplotlib import pyplot as plt

    figure, axis = plt.subplots(figsize=(7, 4), constrained_layout=True)
    axis.plot([row.cvar for row in workflow.trace])
    axis.set(
        xlabel="Objective evaluations",
        ylabel="Encoded cost CVaR (0.7)",
        title="Native portfolio cost search",
    )
    return figure


def plot_costs(workflow: Workflow) -> "Figure":
    from matplotlib import pyplot as plt

    figure, axis = plt.subplots(figsize=(7, 4), constrained_layout=True)
    all_costs = [row.cost for row in (*workflow.optimized, *workflow.uniform)]
    low, high = min(all_costs), max(all_costs)
    if high == low:
        low, high = low - 0.5, high + 0.5
    bins = [low + (high - low) * i / 40 for i in range(41)]
    for label, rows in (
        ("optimized", workflow.optimized),
        ("uniform", workflow.uniform),
    ):
        shots = sum(row.count for row in rows)
        axis.hist(
            [row.cost for row in rows],
            bins=bins,
            weights=[row.count / shots for row in rows],
            alpha=0.6,
            label=label,
        )
    axis.set(
        xlabel="Original risk minus return",
        ylabel="Observed probability",
        title="Sampled portfolio costs (including infeasible holdings)",
    )
    axis.legend()
    return figure
