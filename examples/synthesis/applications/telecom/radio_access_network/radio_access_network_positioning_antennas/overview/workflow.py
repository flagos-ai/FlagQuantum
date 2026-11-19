"""Run source mean-cost optimization and actual finite antenna observations."""

import math
from dataclasses import dataclass

import torch

import flagquantum as fq
from examples.synthesis.graph_optimization import Evaluation
from flagquantum import synth
from flagquantum.algorithms import NelderMeadOptimizer, NelderMeadResult

from .model import (
    Assignment,
    assignment,
    fresh_coefficients,
    residual_penalty,
    validate_coefficients,
)
from .quantum import build_program

INITIAL_PARAMETERS = tuple(math.pi * (i + 1) / 4 for i in range(4)) + tuple(
    math.pi * (4 - i) / 4 for i in range(4)
)


@dataclass(frozen=True)
class Sample:
    """A joint source observation; the model slack remains part of its cost."""

    assignment: Assignment
    slack: int
    penalty: int
    count: int

    @property
    def cost(self) -> float:
        return -self.assignment.score + 3 * self.penalty


def cost_table(coefficients: tuple[float, ...]) -> torch.Tensor:
    decisions = tuple(assignment(word, coefficients) for word in range(8192))
    return torch.tensor(
        [
            -value.score + 3 * residual_penalty(value.word, slack)
            for slack in range(8)
            for value in decisions
        ],
        dtype=torch.float64,
    )


def read_distribution(program: synth.Program) -> torch.Tensor:
    if tuple(program.registers) != ("x", "slack") or tuple(
        len(program.registers[name]) for name in ("x", "slack")
    ) != (13, 3):
        raise ValueError(
            "antenna program requires thirteen decision and three slack bits"
        )
    probabilities = torch.zeros(2**16, dtype=torch.float64)
    for (word, slack), value in program.probabilities(
        fq.run(program), registers=("x", "slack"), threshold=0
    ).items():
        probabilities[word | (slack << 13)] = value
    return probabilities


def sample_distribution(
    probabilities: torch.Tensor,
    coefficients: tuple[float, ...],
    *,
    shots: int = 3000,
    seed: int = 2718
) -> tuple[Sample, ...]:
    validate_coefficients(coefficients)
    if type(shots) is not int or not 1 <= shots <= 100000:
        raise ValueError("shots must be an integer in1..100000")
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("seed must be a nonnegative63-bitinteger")
    if (
        probabilities.shape != (2**16,)
        or not bool(torch.isfinite(probabilities).all())
        or bool((probabilities < 0).any())
        or abs(float(probabilities.sum()) - 1) > 1e-10
    ):
        raise ValueError(
            "probabilities must be a normalized finite nonnegative sixteen-bit vector"
        )
    draws = torch.multinomial(
        probabilities,
        shots,
        replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )
    counts = torch.bincount(draws, minlength=2**16)
    rows = []
    for word in torch.nonzero(counts, as_tuple=False).flatten().tolist():
        decision, slack = word & 8191, word >> 13
        rows.append(
            Sample(
                assignment(decision, coefficients),
                slack,
                residual_penalty(decision, slack),
                int(counts[word]),
            )
        )
    return tuple(
        sorted(rows, key=lambda row: (row.cost, row.assignment.word, row.slack))
    )


@dataclass(frozen=True)
class Workflow:
    """Fresh input, actual optimization trace and source-ordered observations."""

    coefficients: tuple[float, ...]
    program: synth.Program
    optimizer: NelderMeadResult
    trace: tuple[Evaluation, ...]
    samples: tuple[Sample, ...]


def run_workflow(*, maxiter: int = 60, shots: int = 3000, seed: int = 2718) -> Workflow:
    """Freshly regenerate source data; retain all four layers and exact mean cost."""
    if type(maxiter) is not int or not 1 <= maxiter <= 128:
        raise ValueError("maxiter must be an integer in1..128")
    # Validate controls before spending any native quantum evaluations.
    if type(shots) is not int or not 1 <= shots <= 100000:
        raise ValueError("shots must be an integer in1..100000")
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("seed must be a nonnegative63-bitinteger")
    coefficients = fresh_coefficients()
    costs = cost_table(coefficients)
    trace: list[Evaluation] = []

    def evaluate(parameters: torch.Tensor) -> torch.Tensor:
        values = tuple(float(value) for value in parameters)
        cost = float(
            torch.dot(read_distribution(build_program(values, coefficients)), costs)
        )
        trace.append(Evaluation(values, cost))
        return torch.tensor(cost, dtype=torch.float64)

    optimizer = NelderMeadOptimizer(maxiter=maxiter, initial_step=0.07).minimize(
        evaluate, torch.tensor(INITIAL_PARAMETERS, dtype=torch.float64)
    )
    program = build_program(
        tuple(float(value) for value in optimizer.parameters), coefficients
    )
    samples = sample_distribution(
        read_distribution(program), coefficients, shots=shots, seed=seed
    )
    return Workflow(coefficients, program, optimizer, tuple(trace), samples)
