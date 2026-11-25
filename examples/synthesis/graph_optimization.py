"""Shared local optimization/readout for the two fixed subset-selection tutorials.

This is example code, not a general modeling API. Tables contain at most 256
assignments. Every optimizer call synthesizes and executes the native program.
"""

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch

import flagquantum as fq
from flagquantum import synth
from flagquantum.algorithms import NelderMeadOptimizer, NelderMeadResult

INITIAL_PARAMETERS = (0.13, -0.27, 0.19, 0.22, 0.31, -0.17)


@dataclass(frozen=True)
class Evaluation:
    """Parameters and lower-tail cost from one actual quantum execution."""

    parameters: tuple[float, ...]
    cost: float


@dataclass(frozen=True)
class Sample:
    """A sampled subset, its feasibility, and actual observation count."""

    word: int
    nodes: tuple[int, ...]
    penalized_cost: float
    feasible: bool
    count: int


@dataclass(frozen=True)
class SubsetWorkflow:
    """Bounded search, local samples, and an independently labeled exact reference."""

    program: synth.Program
    optimizer: NelderMeadResult
    trace: tuple[Evaluation, ...]
    optimized: tuple[Sample, ...]
    uniform: tuple[Sample, ...]
    exact_nodes: tuple[int, ...]

    @property
    def best_feasible(self) -> Sample | None:
        """Select the largest observed feasible subset, without using the oracle."""
        return next((row for row in self.optimized if row.feasible), None)


def lower_tail_cost(
    probabilities: Sequence[float], costs: Sequence[float], quantile: float
) -> float:
    """Integrate normalized cost mass through a possibly fractional quantile."""
    if not math.isfinite(quantile) or not 0 < quantile <= 1:
        raise ValueError("quantile must be finite and in (0, 1]")
    if len(probabilities) != len(costs) or not probabilities:
        raise ValueError("probabilities and costs must have the same positive length")
    if any(not math.isfinite(p) or p < 0 for p in probabilities):
        raise ValueError("probabilities must be finite and nonnegative")
    if abs(sum(probabilities) - 1) > 1e-10:
        raise ValueError("probabilities must sum to one")
    if any(not math.isfinite(cost) for cost in costs):
        raise ValueError("costs must be finite")
    mass, value = 0.0, 0.0
    for word in sorted(range(len(costs)), key=lambda w: (costs[w], w)):
        take = min(probabilities[word], quantile - mass)
        mass += take
        value += take * costs[word]
        if mass >= quantile - 1e-14:
            break
    return value / quantile


def selected_nodes(word: int, width: int) -> tuple[int, ...]:
    """Decode node i from bit i of the named numeric array value."""
    return tuple(i for i in range(width) if word & (1 << i))


def sample_subsets(
    probabilities: tuple[float, ...],
    costs: tuple[float, ...],
    feasible_words: frozenset[int],
    *,
    shots: int,
    seed: int,
) -> tuple[Sample, ...]:
    """Draw actual local samples and sort feasibility before unpenalized size."""
    if (
        isinstance(shots, bool)
        or not isinstance(shots, int)
        or not 1 <= shots <= 100000
    ):
        raise ValueError("shots must be an integer in [1, 100000]")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63:
        raise ValueError("seed must be a nonnegative 63-bit integer")
    lower_tail_cost(probabilities, costs, 1.0)
    width = len(costs).bit_length() - 1
    if not 1 <= width <= 8 or len(costs) != 1 << width:
        raise ValueError("subset tables require 1..8 bits and a power-of-two length")
    if any(
        isinstance(w, bool) or not isinstance(w, int) or not 0 <= w < len(costs)
        for w in feasible_words
    ):
        raise ValueError("feasible assignments must be valid integer table indices")
    draws = torch.multinomial(
        torch.tensor(probabilities, dtype=torch.float64),
        shots,
        replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )
    counts = torch.bincount(draws, minlength=len(costs))
    rows = (
        Sample(
            word,
            selected_nodes(word, width),
            costs[word],
            word in feasible_words,
            int(counts[word]),
        )
        for word in range(len(costs))
        if int(counts[word]) > 0
    )
    return tuple(
        sorted(rows, key=lambda row: (not row.feasible, -len(row.nodes), row.word))
    )


def read_distribution(program: synth.Program, width: int) -> tuple[float, ...]:
    """Execute and read the complete named array distribution in node order."""
    if tuple(program.registers) != ("v",) or len(program.registers["v"]) != width:
        raise ValueError("native program must expose only the declared v node array")
    probabilities = dict(program.probabilities(fq.run(program), registers=("v",)))
    if any(len(word) != 1 or not 0 <= word[0] < 1 << width for word in probabilities):
        raise ValueError("named readout contains an assignment outside the node array")
    return tuple(probabilities.get((word,), 0.0) for word in range(1 << width))


def run_subset_optimization(
    build_program: Callable[[tuple[float, ...]], synth.Program],
    costs: tuple[float, ...],
    feasible_words: frozenset[int],
    *,
    maxiter: int,
    shots: int = 1000,
    seed: int = 2718,
    quantile: float = 0.7,
) -> SubsetWorkflow:
    """Run bounded native optimization, sample it, then enumerate the reference.

    The exact reference is calculated after optimization and sample selection.
    It does not supply parameters, probabilities, or sampled candidate answers.
    """
    if (
        isinstance(maxiter, bool)
        or not isinstance(maxiter, int)
        or not 1 <= maxiter <= 128
    ):
        raise ValueError("maxiter must be an integer in [1, 128]")
    if not feasible_words:
        raise ValueError("the fixed problem must have at least one feasible assignment")
    uniform = tuple(1.0 / len(costs) for _ in costs) if costs else ()
    lower_tail_cost(uniform, costs, quantile)
    uniform_samples = sample_subsets(
        uniform, costs, feasible_words, shots=shots, seed=31415
    )
    # Validate the optimized sample controls before spending quantum executions.
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63:
        raise ValueError("seed must be a nonnegative 63-bit integer")
    width = len(costs).bit_length() - 1
    trace: list[Evaluation] = []

    def evaluate(parameters: torch.Tensor) -> torch.Tensor:
        values = tuple(float(value) for value in parameters)
        probabilities = read_distribution(build_program(values), width)
        cost = lower_tail_cost(probabilities, costs, quantile)
        trace.append(Evaluation(values, cost))
        return torch.tensor(cost, dtype=torch.float64)

    optimizer = NelderMeadOptimizer(maxiter=maxiter, initial_step=0.07).minimize(
        evaluate, torch.tensor(INITIAL_PARAMETERS, dtype=torch.float64)
    )
    program = build_program(tuple(float(value) for value in optimizer.parameters))
    probabilities = read_distribution(program, width)
    optimized = sample_subsets(
        probabilities, costs, feasible_words, shots=shots, seed=seed
    )
    exact_word = min(feasible_words, key=lambda word: (costs[word], word))
    return SubsetWorkflow(
        program,
        optimizer,
        tuple(trace),
        optimized,
        uniform_samples,
        selected_nodes(exact_word, width),
    )
