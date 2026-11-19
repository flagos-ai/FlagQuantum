"""Optimize actual native executions and inspect seeded local samples."""

from collections.abc import Mapping
from dataclasses import dataclass

import torch

import flagquantum as fq
from flagquantum import synth
from flagquantum.algorithms import NelderMeadOptimizer, NelderMeadResult

from .program import build_program


@dataclass(frozen=True)
class Evaluation:
    """One actual synthesize/run/readout call in the optimization trace."""

    gamma: float
    beta: float
    expectation: float


@dataclass(frozen=True)
class Sample:
    """An observed assignment with its unpenalized objective and feasibility."""

    x: int
    y: int
    cost: int
    feasible: bool
    count: int


@dataclass(frozen=True)
class Workflow:
    """A bounded optimizer run and local samples of the final quantum state."""

    program: synth.Program
    optimizer: NelderMeadResult
    trace: tuple[Evaluation, ...]
    samples: tuple[Sample, ...]

    @property
    def best_feasible(self) -> Sample | None:
        """Return the best observed feasible assignment, if any."""
        return next((sample for sample in self.samples if sample.feasible), None)


def _validate_sampling(shots: int, seed: int) -> None:
    if (
        isinstance(shots, bool)
        or not isinstance(shots, int)
        or not 1 <= shots <= 100000
    ):
        raise ValueError("shots must be an integer in [1, 100000]")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63:
        raise ValueError("seed must be a nonnegative 63-bit integer")


def sample_distribution(
    probabilities: Mapping[tuple[int, ...], float], *, shots: int, seed: int
) -> tuple[Sample, ...]:
    """Sample the executed named distribution on a local CPU generator."""
    _validate_sampling(shots, seed)
    if any(
        len(word) != 2 or any(v not in range(4) for v in word) for word in probabilities
    ):
        raise ValueError("distribution must contain two integers in [0, 3]")
    weights = torch.tensor(
        [probabilities.get((x, y), 0.0) for x in range(4) for y in range(4)],
        dtype=torch.float64,
    )
    if (
        not bool(torch.isfinite(weights).all())
        or bool((weights < 0).any())
        or abs(float(weights.sum()) - 1.0) > 1e-10
    ):
        raise ValueError("distribution must have finite nonnegative normalized weights")
    words = torch.multinomial(
        weights, shots, replacement=True, generator=torch.Generator().manual_seed(seed)
    )
    counts = torch.bincount(words, minlength=16)
    rows = (
        Sample(x, y, 3 * x + 2 * y, 3 * x + y >= 2, int(counts[4 * x + y]))
        for x in range(4)
        for y in range(4)
        if int(counts[4 * x + y]) > 0
    )
    return tuple(
        sorted(rows, key=lambda row: (not row.feasible, row.cost, row.x, row.y))
    )


def run_workflow(*, maxiter: int = 32, shots: int = 1000, seed: int = 2718) -> Workflow:
    """Minimize the penalized expectation, then sample the final actual run.

    This one-layer search is bounded; convergence and an optimal sampled answer
    are not guaranteed. The exhaustive classical optimum is only a test oracle.
    """
    if (
        isinstance(maxiter, bool)
        or not isinstance(maxiter, int)
        or not 1 <= maxiter <= 128
    ):
        raise ValueError("maxiter must be an integer in [1, 128]")
    # Validate sampling controls before spending any quantum evaluations.
    _validate_sampling(shots, seed)
    trace: list[Evaluation] = []

    def evaluate(parameters: torch.Tensor) -> torch.Tensor:
        gamma, beta = float(parameters[0]), float(parameters[1])
        program = build_program(gamma, beta)
        probabilities = program.probabilities(fq.run(program), registers=("x", "y"))
        expectation = sum(
            p * (3 * x + 2 * y + 20 * max(0, 2 - 3 * x - y) ** 2)
            for (x, y), p in probabilities.items()
        )
        trace.append(Evaluation(gamma, beta, expectation))
        return torch.tensor(expectation, dtype=torch.float64)

    optimizer = NelderMeadOptimizer(maxiter=maxiter, initial_step=0.07).minimize(
        evaluate, torch.tensor([0.19, 0.37], dtype=torch.float64)
    )
    program = build_program(
        float(optimizer.parameters[0]), float(optimizer.parameters[1])
    )
    probabilities = program.probabilities(fq.run(program), registers=("x", "y"))
    samples = sample_distribution(dict(probabilities), shots=shots, seed=seed)
    return Workflow(program, optimizer, tuple(trace), samples)
