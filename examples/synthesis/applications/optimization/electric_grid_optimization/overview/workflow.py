"""Exact mean-cost search on all eighteen bits with separate finite samples."""

import math
from dataclasses import dataclass

import torch

import flagquantum as fq
from examples.synthesis.graph_optimization import Evaluation
from flagquantum import synth
from flagquantum.algorithms import NelderMeadOptimizer, NelderMeadResult

from .model import Assignment, assignment, classical_reference, residual_penalty
from .quantum import build_program

REGISTERS = ("a", "b", "c", "sa", "sb", "sc")
INITIAL_PARAMETERS = tuple(math.pi * (i + 1) / 4 for i in range(4)) + tuple(
    math.pi * (4 - i) / 4 for i in range(4)
)


@dataclass(frozen=True)
class Sample:
    """Actual joint observation including slack, cost, and decision feasibility."""

    joint_word: int
    assignment: Assignment
    slack: tuple[int, int, int]
    penalty: int
    count: int

    @property
    def penalized_cost(self) -> float:
        return self.assignment.raw_cost + 10 * self.penalty


def decode_joint(word: int) -> tuple[Assignment, tuple[int, int, int]]:
    if type(word) is not int or not 0 <= word < 2**18:
        raise ValueError("joint word must be an integer in 0..262143")
    return assignment(word & 4095), (
        (word >> 12) & 3,
        (word >> 14) & 3,
        (word >> 16) & 3,
    )


def cost_table() -> torch.Tensor:
    """Build fresh exact source residual costs, including every slack choice."""
    decisions = tuple(assignment(word) for word in range(4096))
    return torch.tensor(
        [
            value.raw_cost + 10 * residual_penalty(value, (sa, sb, sc))
            for sc in range(4)
            for sb in range(4)
            for sa in range(4)
            for value in decisions
        ],
        dtype=torch.float64,
    )


def read_distribution(program: synth.Program) -> torch.Tensor:
    """Read named arrays and assemble a logical joint word without physical wires."""
    if tuple(program.registers) != REGISTERS or tuple(
        len(program.registers[name]) for name in REGISTERS
    ) != (4, 4, 4, 2, 2, 2):
        raise ValueError(
            "electric-grid program must expose four-bit decisions and two-bit slack arrays"
        )
    table = torch.zeros(2**18, dtype=torch.float64)
    for key, probability in program.probabilities(
        fq.run(program), registers=REGISTERS, max_bits=18, max_keys=2**18, threshold=0
    ).items():
        a, b, c, sa, sb, sc = key
        table[a | (b << 4) | (c << 8) | (sa << 12) | (sb << 14) | (sc << 16)] = (
            probability
        )
    return table


def sample_distribution(
    probabilities: torch.Tensor, *, shots: int = 1000, seed: int = 2718
) -> tuple[Sample, ...]:
    """Do not strip slack until its source penalized cost has been retained."""
    if type(shots) is not int or not 1 <= shots <= 100000:
        raise ValueError("shots must be an integer in 1..100000")
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("seed must be a nonnegative 63-bit integer")
    if (
        probabilities.shape != (2**18,)
        or not bool(torch.isfinite(probabilities).all())
        or bool((probabilities < 0).any())
        or abs(float(probabilities.sum()) - 1) > 1e-10
    ):
        raise ValueError(
            "probabilities must be a normalized finite nonnegative eighteen-bit vector"
        )
    draws = torch.multinomial(
        probabilities,
        shots,
        replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )
    counts = torch.bincount(draws, minlength=2**18)
    samples = []
    for word in torch.nonzero(counts, as_tuple=False).flatten().tolist():
        value, slack = decode_joint(word)
        samples.append(
            Sample(
                word, value, slack, residual_penalty(value, slack), int(counts[word])
            )
        )
    return tuple(sorted(samples, key=lambda row: (row.penalized_cost, row.joint_word)))


@dataclass(frozen=True)
class Workflow:
    """Actual search, optimized/uniform observations and a separately solved reference."""

    program: synth.Program
    optimizer: NelderMeadResult
    trace: tuple[Evaluation, ...]
    samples: tuple[Sample, ...]
    uniform: tuple[Sample, ...]
    reference: tuple[Assignment, ...]

    @property
    def best_observed(self) -> Sample:
        return self.samples[0]


def run_workflow(
    *, maxiter: int = 100, shots: int = 1000, seed: int = 2718
) -> Workflow:
    """Retain source100iterations and mean objective; resynthesize each angle tuple."""
    if type(maxiter) is not int or not 1 <= maxiter <= 128:
        raise ValueError("maxiter must be an integer in 1..128")
    uniform = sample_distribution(
        read_distribution(build_program((0.0,) * 8)), shots=shots, seed=31415
    )
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("seed must be a nonnegative 63-bit integer")
    costs = cost_table()
    trace: list[Evaluation] = []

    def evaluate(parameters: torch.Tensor) -> torch.Tensor:
        values = tuple(float(value) for value in parameters)
        cost = float(torch.dot(read_distribution(build_program(values)), costs))
        trace.append(Evaluation(values, cost))
        return torch.tensor(cost, dtype=torch.float64)

    optimizer = NelderMeadOptimizer(maxiter=maxiter, initial_step=0.07).minimize(
        evaluate, torch.tensor(INITIAL_PARAMETERS, dtype=torch.float64)
    )
    program = build_program(tuple(float(value) for value in optimizer.parameters))
    samples = sample_distribution(read_distribution(program), shots=shots, seed=seed)
    return Workflow(
        program, optimizer, tuple(trace), samples, uniform, classical_reference()
    )
