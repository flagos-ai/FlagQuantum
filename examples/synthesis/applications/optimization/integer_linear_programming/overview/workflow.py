"""Bounded original-domain integer optimization using actual native readout."""

import itertools
import math
from dataclasses import dataclass

import torch

import flagquantum as fq
from examples.synthesis.applications.optimization.integer_linear_programming.main import (
    main,
)
from examples.synthesis.graph_optimization import (
    INITIAL_PARAMETERS,
    Evaluation,
    lower_tail_cost,
)
from flagquantum import synth
from flagquantum.algorithms import NelderMeadOptimizer, NelderMeadResult

Assignment = tuple[int, int, int, int, int]
NAMES = ("x0", "x1", "x2", "s0", "s1")
DOMAIN: tuple[Assignment, ...] = tuple(
    (a, b, c, s0, s1)
    for a, b, c, s0, s1 in itertools.product(
        range(4), range(4), range(4), range(2), range(2)
    )
)
SOURCE_A = ((1, 1, 1), (2, 2, 2), (3, 3, 3))
SOURCE_B = (1, 2, 3)
SOURCE_C = (1, 2, 3)


@dataclass(frozen=True)
class Sample:
    """An observed joint assignment, original model checks and retained quantum cost."""

    assignment: Assignment
    objective: int
    lhs: tuple[int, int, int]
    feasible: bool
    retained_quantum_cost: int
    count: int


@dataclass(frozen=True)
class Workflow:
    """An actual optimization and sampling run with a separate bounded reference."""

    program: synth.Program
    optimizer: NelderMeadResult
    trace: tuple[Evaluation, ...]
    samples: tuple[Sample, ...]
    uniform: tuple[Sample, ...]
    exact_reference: tuple[int, int, int]

    @property
    def best_feasible(self) -> Sample | None:
        """Return the best actually observed feasible row, if present."""
        return next((row for row in self.samples if row.feasible), None)


def model_row(assignment: Assignment, count: int = 0) -> Sample:
    """Check all three original inequalities separately from the frozen quantum cost."""
    if len(assignment) != 5 or any(
        isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < bound
        for value, bound in zip(assignment, (4, 4, 4, 2, 2), strict=True)
    ):
        raise ValueError("assignment requires three integers in [0, 3] and two bits")
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError("count must be a nonnegative integer")
    a, b, c, s0, s1 = assignment
    values = (a, b, c)
    objective = sum(w * v for w, v in zip(SOURCE_C, values, strict=True))
    lhs = (a + b + c, 2 * a + 2 * b + 2 * c, 3 * a + 3 * b + 3 * c)
    feasible = all(value <= bound for value, bound in zip(lhs, SOURCE_B, strict=True))
    cost = (
        10 * (a * b + a * c + b * c)
        - objective
        + 10 * (s0 + a + b + c - 1) ** 2
        + 10 * (s1 + a + b + c - 1) ** 2
    )
    return Sample(assignment, objective, lhs, feasible, cost, count)


def build_program(parameters: tuple[float, ...] = INITIAL_PARAMETERS) -> synth.Program:
    """Reuse the frozen eight-bit quantum encoding with explicit local limits."""
    if len(parameters) != 6 or any(not math.isfinite(value) for value in parameters):
        raise ValueError("three layers require six finite radian parameters")
    return synth.synthesize(
        main,
        classical_inputs={"params": parameters},
        max_qubits=8,
        max_workspace=0,
        max_operations=5000,
        memory_limit_bytes=4 * 1024**2,
    )


def read_distribution(program: synth.Program) -> tuple[float, ...]:
    """Read the complete named joint domain, including both modeled slack outputs."""
    if tuple(program.registers) != NAMES or tuple(
        len(program.registers[name]) for name in NAMES
    ) != (2, 2, 2, 1, 1):
        raise ValueError("integer optimization requires x0/x1/x2 and s0/s1 registers")
    readout = dict(program.probabilities(fq.run(program), registers=NAMES))
    if not set(readout).issubset(DOMAIN):
        raise ValueError("named readout lies outside the original joint domain")
    probabilities = tuple(readout.get(key, 0.0) for key in DOMAIN)
    lower_tail_cost(
        probabilities,
        tuple(model_row(key).retained_quantum_cost for key in DOMAIN),
        1.0,
    )
    return probabilities


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
    probabilities: tuple[float, ...], *, shots: int, seed: int
) -> tuple[Sample, ...]:
    """Draw actual joint samples and rank original feasibility and objective first."""
    _validate_sampling(shots, seed)
    lower_tail_cost(
        probabilities,
        tuple(model_row(key).retained_quantum_cost for key in DOMAIN),
        1.0,
    )
    draws = torch.multinomial(
        torch.tensor(probabilities, dtype=torch.float64),
        shots,
        replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )
    counts = torch.bincount(draws, minlength=256)
    return tuple(
        sorted(
            (
                model_row(key, int(counts[index]))
                for index, key in enumerate(DOMAIN)
                if int(counts[index]) > 0
            ),
            key=lambda row: (
                not row.feasible,
                -row.objective,
                row.retained_quantum_cost,
                row.assignment,
            ),
        )
    )


def run_workflow(*, maxiter: int = 90, shots: int = 1000, seed: int = 2718) -> Workflow:
    """Optimize the retained joint cost, sample, then compute an independent reference."""
    if (
        isinstance(maxiter, bool)
        or not isinstance(maxiter, int)
        or not 1 <= maxiter <= 128
    ):
        raise ValueError("maxiter must be an integer in [1, 128]")
    _validate_sampling(shots, seed)
    costs = tuple(model_row(key).retained_quantum_cost for key in DOMAIN)
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
    uniform = sample_distribution((1 / 256,) * 256, shots=shots, seed=31415)
    feasible = (
        (a, b, c)
        for a, b, c in itertools.product(range(4), repeat=3)
        if model_row((a, b, c, 0, 0)).feasible
    )
    exact = min(
        feasible,
        key=lambda values: (
            -sum(w * v for w, v in zip(SOURCE_C, values, strict=True)),
            values,
        ),
    )
    return Workflow(program, optimizer, tuple(trace), samples, uniform, exact)
