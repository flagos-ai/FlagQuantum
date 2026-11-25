"""Actual native optimization and sampled partitions for the source Notebook data."""

from dataclasses import dataclass

import torch

import flagquantum as fq
from examples.synthesis.graph_optimization import Evaluation, lower_tail_cost
from flagquantum import synth
from flagquantum.algorithms import NelderMeadOptimizer, NelderMeadResult

from .program import INITIAL_PARAMETERS, build_program, source_weights, validate_weights


@dataclass(frozen=True)
class Sample:
    """An actual observed subset with its squared difference and count."""

    word: int
    cost: int
    count: int


@dataclass(frozen=True)
class Workflow:
    """Actual native optimization, samples and a separate bounded exact reference."""

    weights: tuple[int, ...]
    program: synth.Program
    optimizer: NelderMeadResult
    trace: tuple[Evaluation, ...]
    samples: tuple[Sample, ...]
    uniform: tuple[Sample, ...]
    exact_reference_word: int

    @property
    def best_observed(self) -> Sample:
        """Choose the best actually observed assignment without using the reference."""
        return self.samples[0]


def partitions(
    word: int, weights: tuple[int, ...]
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Map named bit i to source item i and return both value groups."""
    validate_weights(weights)
    if isinstance(word, bool) or not isinstance(word, int) or not 0 <= word < 1024:
        raise ValueError("partition assignment must be an integer in [0, 1023]")
    return (
        tuple(value for i, value in enumerate(weights) if word & (1 << i)),
        tuple(value for i, value in enumerate(weights) if not word & (1 << i)),
    )


def costs(weights: tuple[int, ...]) -> tuple[int, ...]:
    """Calculate the independent classical objective on the bounded input domain."""
    validate_weights(weights)
    return tuple(
        sum(value * (2 * ((word >> i) & 1) - 1) for i, value in enumerate(weights)) ** 2
        for word in range(1024)
    )


def distribution(program: synth.Program) -> tuple[float, ...]:
    """Execute and read v as a named integer with item i at bit i."""
    if tuple(program.registers) != ("v",) or len(program.registers["v"]) != 10:
        raise ValueError("the source program must expose one ten-bit v array")
    readout = dict(program.probabilities(fq.run(program), registers=("v",)))
    if any(len(word) != 1 or not 0 <= word[0] < 1024 for word in readout):
        raise ValueError("readout contains an assignment outside the ten-bit domain")
    return tuple(readout.get((word,), 0.0) for word in range(1024))


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
    weights: tuple[int, ...],
    probabilities: tuple[float, ...],
    *,
    shots: int,
    seed: int,
) -> tuple[Sample, ...]:
    """Sample actual local probabilities; no exact answer replaces observations."""
    _validate_sampling(shots, seed)
    table = costs(weights)
    lower_tail_cost(probabilities, table, 1.0)
    draws = torch.multinomial(
        torch.tensor(probabilities, dtype=torch.float64),
        shots,
        replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )
    counts = torch.bincount(draws, minlength=1024)
    return tuple(
        sorted(
            (
                Sample(word, table[word], int(counts[word]))
                for word in range(1024)
                if int(counts[word]) > 0
            ),
            key=lambda row: (row.cost, row.word),
        )
    )


def run_workflow(
    weights: tuple[int, ...] | None = None,
    *,
    maxiter: int = 80,
    shots: int = 1000,
    seed: int = 2718,
) -> Workflow:
    """Optimize native CVaR0.7, sample it, then enumerate a labeled reference."""
    if (
        isinstance(maxiter, bool)
        or not isinstance(maxiter, int)
        or not 1 <= maxiter <= 128
    ):
        raise ValueError("maxiter must be an integer in [1, 128]")
    _validate_sampling(shots, seed)
    values = source_weights() if weights is None else weights
    table = costs(values)
    trace: list[Evaluation] = []

    def evaluate(parameters: torch.Tensor) -> torch.Tensor:
        angles = tuple(float(value) for value in parameters)
        probabilities = distribution(build_program(values, angles))
        cost = lower_tail_cost(probabilities, table, 0.7)
        trace.append(Evaluation(angles, cost))
        return torch.tensor(cost, dtype=torch.float64)

    optimizer = NelderMeadOptimizer(maxiter=maxiter, initial_step=0.07).minimize(
        evaluate, torch.tensor(INITIAL_PARAMETERS, dtype=torch.float64)
    )
    program = build_program(
        values, tuple(float(value) for value in optimizer.parameters)
    )
    samples = sample_distribution(values, distribution(program), shots=shots, seed=seed)
    uniform = sample_distribution(values, (1 / 1024,) * 1024, shots=shots, seed=31415)
    exact_reference_word = min(range(1024), key=lambda word: (table[word], word))
    return Workflow(
        values, program, optimizer, tuple(trace), samples, uniform, exact_reference_word
    )
