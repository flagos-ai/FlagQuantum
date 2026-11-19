"""Optimize and sample the original-sized kidney matching objective."""

import math
from dataclasses import dataclass

import torch

import flagquantum as fq
from examples.synthesis.graph_optimization import Evaluation, lower_tail_cost
from flagquantum import synth
from flagquantum.algorithms import NelderMeadOptimizer, NelderMeadResult

from .model import Assignment, assignment
from .quantum import build_program

INITIAL_PARAMETERS = tuple(math.pi * (layer + 1) / 5 for layer in range(5)) + tuple(
    math.pi * (5 - layer) / 5 for layer in range(5)
)


@dataclass(frozen=True)
class Sample:
    """An observed matching, including its feasibility and actual count."""

    assignment: Assignment
    count: int


@dataclass(frozen=True)
class Workflow:
    """Actual optimization, observations and a separate legal-matching reference."""

    program: synth.Program
    optimizer: NelderMeadResult
    trace: tuple[Evaluation, ...]
    samples: tuple[Sample, ...]
    reference: Assignment

    @property
    def best_observed(self) -> Sample:
        """Select by the sampled penalized cost, as in the source final cell."""
        return self.samples[0]

    @property
    def best_observed_feasible(self) -> Sample | None:
        """Keep a missing legal observation explicit instead of substituting a reference."""
        return next((row for row in self.samples if row.assignment.feasible), None)


def read_distribution(program: synth.Program) -> tuple[float, ...]:
    """Read all 512 named assignments without physical line numbers."""
    if (
        tuple(program.registers) != ("matches",)
        or len(program.registers["matches"]) != 9
    ):
        raise ValueError(
            "kidney program must expose only the nine-bit matches register"
        )
    distribution = dict(program.probabilities(fq.run(program), registers=("matches",)))
    return tuple(distribution.get((word,), 0.0) for word in range(512))


def sample_distribution(
    probabilities: tuple[float, ...], *, shots: int = 1000, seed: int = 2718
) -> tuple[Sample, ...]:
    """Draw local samples and sort by actual penalized objective, then word."""
    if type(shots) is not int or not 1 <= shots <= 100000:
        raise ValueError("shots must be an integer in 1..100000")
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("seed must be a nonnegative 63-bit integer")
    if len(probabilities) != 512:
        raise ValueError("the distribution must contain all 512 assignments")
    table = tuple(assignment(word) for word in range(512))
    lower_tail_cost(probabilities, tuple(row.penalized_cost for row in table), 0.7)
    draws = torch.multinomial(
        torch.tensor(probabilities, dtype=torch.float64),
        shots,
        replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )
    counts = torch.bincount(draws, minlength=512)
    return tuple(
        sorted(
            (
                Sample(table[word], int(count))
                for word, count in enumerate(counts)
                if int(count) > 0
            ),
            key=lambda row: (row.assignment.penalized_cost, row.assignment.word),
        )
    )


def run_workflow(
    *, maxiter: int = 100, shots: int = 1000, seed: int = 2718
) -> Workflow:
    """Run five-layer CVaR 0.7 using public native simplex optimization.

    Exact local CVaR replaces shot estimates during optimization; every vertex
    binds a fresh native program. This does not implement runtime binding or
    reproduce the proprietary optimizer trajectory. Final samples are finite.
    """
    if type(maxiter) is not int or not 1 <= maxiter <= 128:
        raise ValueError("maxiter must be an integer in 1..128")
    table = tuple(assignment(word) for word in range(512))
    costs = tuple(row.penalized_cost for row in table)
    trace: list[Evaluation] = []

    def evaluate(parameters: torch.Tensor) -> torch.Tensor:
        values = tuple(float(value) for value in parameters)
        cost = lower_tail_cost(read_distribution(build_program(values)), costs, 0.7)
        trace.append(Evaluation(values, cost))
        return torch.tensor(cost, dtype=torch.float64)

    optimizer = NelderMeadOptimizer(maxiter=maxiter, initial_step=0.07).minimize(
        evaluate, torch.tensor(INITIAL_PARAMETERS, dtype=torch.float64)
    )
    program = build_program(tuple(float(value) for value in optimizer.parameters))
    samples = sample_distribution(read_distribution(program), shots=shots, seed=seed)
    reference = min(
        (row for row in table if row.feasible), key=lambda row: (-row.score, row.word)
    )
    return Workflow(program, optimizer, tuple(trace), samples, reference)
