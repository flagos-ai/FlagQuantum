"""Run original fifty-iteration native mean-cost facility optimization and sampling."""

import math
from dataclasses import dataclass

import torch

import flagquantum as fq
from examples.synthesis.graph_optimization import Evaluation, lower_tail_cost
from flagquantum import synth
from flagquantum.algorithms import NelderMeadOptimizer, NelderMeadResult

from .model import Assignment, assignment, classical_reference
from .quantum import build_program

INITIAL_PARAMETERS = tuple(math.pi * (i + 1) / 5 for i in range(5)) + tuple(
    math.pi * (5 - i) / 5 for i in range(5)
)


@dataclass(frozen=True)
class Sample:
    assignment: Assignment
    count: int


def read_distribution(program: synth.Program) -> tuple[float, ...]:
    if (
        tuple(program.registers) != ("assignments",)
        or len(program.registers["assignments"]) != 12
    ):
        raise ValueError(
            "facility program requires the original twelve-bit assignments array"
        )
    probabilities = program.probabilities(
        fq.run(program), registers=("assignments",), threshold=0
    )
    return tuple(probabilities.get((word,), 0.0) for word in range(4096))


def sample_distribution(
    probabilities: tuple[float, ...], *, shots: int = 1000, seed: int = 2718
) -> tuple[Sample, ...]:
    if type(shots) is not int or not 1 <= shots <= 100000:
        raise ValueError("shots must be an integer in1..100000")
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("seed must be a nonnegative63-bitinteger")
    table = tuple(assignment(word) for word in range(4096))
    if len(probabilities) != 4096:
        raise ValueError("probabilities must contain all4096assignments")
    lower_tail_cost(probabilities, tuple(value.cost for value in table), 1.0)
    draws = torch.multinomial(
        torch.tensor(probabilities, dtype=torch.float64),
        shots,
        replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )
    counts = torch.bincount(draws, minlength=4096)
    return tuple(
        sorted(
            (
                Sample(table[word], int(count))
                for word, count in enumerate(counts)
                if int(count) > 0
            ),
            key=lambda row: (row.assignment.cost, row.assignment.word),
        )
    )


@dataclass(frozen=True)
class Workflow:
    program: synth.Program
    optimizer: NelderMeadResult
    trace: tuple[Evaluation, ...]
    samples: tuple[Sample, ...]
    uniform: tuple[Sample, ...]
    reference: tuple[Assignment, ...]

    @property
    def best_observed(self) -> Sample:
        return self.samples[0]


def run_workflow(*, maxiter: int = 50, shots: int = 1000, seed: int = 2718) -> Workflow:
    """Preserve source sizes; exact local means replace shot-based cost estimates."""
    if type(maxiter) is not int or not 1 <= maxiter <= 128:
        raise ValueError("maxiter must be an integer in1..128")
    if type(shots) is not int or not 1 <= shots <= 100000:
        raise ValueError("shots must be an integer in1..100000")
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("seed must be a nonnegative63-bitinteger")
    table = tuple(assignment(word) for word in range(4096))
    costs = tuple(value.cost for value in table)
    trace: list[Evaluation] = []

    def evaluate(parameters: torch.Tensor) -> torch.Tensor:
        values = tuple(float(value) for value in parameters)
        cost = lower_tail_cost(read_distribution(build_program(values)), costs, 1.0)
        trace.append(Evaluation(values, cost))
        return torch.tensor(cost, dtype=torch.float64)

    optimizer = NelderMeadOptimizer(maxiter=maxiter, initial_step=0.07).minimize(
        evaluate, torch.tensor(INITIAL_PARAMETERS, dtype=torch.float64)
    )
    program = build_program(tuple(float(value) for value in optimizer.parameters))
    samples = sample_distribution(read_distribution(program), shots=shots, seed=seed)
    uniform = sample_distribution(
        read_distribution(build_program((0.0,) * 10)), shots=shots, seed=31415
    )
    return Workflow(
        program, optimizer, tuple(trace), samples, uniform, classical_reference()
    )
