"""Native optimization/readout for both sections of the original QAOA tutorial."""

import math
from dataclasses import dataclass
from typing import Literal

import torch

import flagquantum as fq
from examples.synthesis.algorithms.search_and_optimization.qaoa.main import (
    main as knapsack,
)
from examples.synthesis.algorithms.search_and_optimization.qaoa.qaoa_max_cut.main import (
    main as maxcut,
)
from flagquantum import synth
from flagquantum.algorithms import NelderMeadOptimizer, NelderMeadResult

Case = Literal["maxcut", "knapsack"]
EDGES = ((0, 1), (0, 2), (1, 2), (1, 3), (2, 4), (3, 4))


@dataclass(frozen=True)
class Evaluation:
    """One actual native execution and its measured expected cost."""

    parameters: tuple[float, ...]
    cost: float


@dataclass(frozen=True)
class Sample:
    """An observed named assignment with its cost and feasibility."""

    assignment: tuple[int, ...]
    cost: float
    feasible: bool
    count: int


@dataclass(frozen=True)
class Workflow:
    """Bounded optimizer trace and actual seeded samples of its final state."""

    case: Case
    program: synth.Program
    optimizer: NelderMeadResult
    trace: tuple[Evaluation, ...]
    samples: tuple[Sample, ...]

    @property
    def best_feasible(self) -> Sample | None:
        """Return the best observed feasible assignment, without consulting an oracle."""
        return next((row for row in self.samples if row.feasible), None)


def assignments(case: Case) -> tuple[tuple[int, ...], ...]:
    """Enumerate the fixed five-bit input domain in named readout order."""
    if case == "maxcut":
        return tuple((word,) for word in range(32))
    if case == "knapsack":
        return tuple((a, b) for a in range(8) for b in range(4))
    raise ValueError("case must be maxcut or knapsack")


def classical_cost(case: Case, assignment: tuple[int, ...]) -> tuple[float, bool]:
    """Apply the source objective and feasibility to a named assignment."""
    if any(
        isinstance(v, bool) or not isinstance(v, int) for v in assignment
    ) or assignment not in assignments(case):
        raise ValueError("assignment must be in the declared five-bit domain")
    if case == "maxcut":
        (word,) = assignment
        cuts = sum(bool(word & (1 << a)) != bool(word & (1 << b)) for a, b in EDGES)
        return -cuts / len(EDGES), True
    if case == "knapsack":
        a, b = assignment
        feasible = 2 * a + 3 * b <= 12
        return float(-3 * a - 5 * b) if feasible else 0.0, feasible
    raise ValueError("case must be maxcut or knapsack")


def initial_parameters(case: Case) -> tuple[float, ...]:
    """Retain each source section's linear schedule, including its pi scaling."""
    assignments(case)
    layers = 4 if case == "maxcut" else 3
    scale = 1.0 if case == "maxcut" else math.pi
    return tuple(
        scale * float(value)
        for value in torch.cat(
            (
                torch.linspace(0, 1, layers, dtype=torch.float64),
                torch.linspace(1, 0, layers, dtype=torch.float64),
            )
        )
    )


def build_program(case: Case, parameters: tuple[float, ...]) -> synth.Program:
    """Reuse the retained native quantum functions with explicit local budgets."""
    expected = len(initial_parameters(case))
    if len(parameters) != expected or any(not math.isfinite(v) for v in parameters):
        raise ValueError(f"{case} requires {expected} finite radian parameters")
    return synth.synthesize(
        maxcut if case == "maxcut" else knapsack,
        classical_inputs={"params": parameters},
        max_qubits=5,
        max_workspace=0,
        max_operations=2000,
        memory_limit_bytes=4 * 1024**2,
    )


def distribution(case: Case, program: synth.Program) -> tuple[float, ...]:
    """Execute and decode named integers; array node i remains bit i."""
    domain = assignments(case)
    names = ("v",) if case == "maxcut" else ("a", "b")
    if tuple(program.registers) != names:
        raise ValueError("program registers do not match the selected tutorial section")
    readout = dict(program.probabilities(fq.run(program), registers=names))
    if set(readout) - set(domain):
        raise ValueError("readout contains an assignment outside the declared domain")
    values = tuple(readout.get(key, 0.0) for key in domain)
    if (
        any(not math.isfinite(value) or value < 0 for value in values)
        or abs(sum(values) - 1.0) > 1e-10
    ):
        raise ValueError("readout must contain normalized finite probabilities")
    return values


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
    case: Case, probabilities: tuple[float, ...], *, shots: int, seed: int
) -> tuple[Sample, ...]:
    """Sample the actual local probabilities and rank feasible observed answers."""
    _validate_sampling(shots, seed)
    domain = assignments(case)
    if (
        len(probabilities) != len(domain)
        or any(not math.isfinite(p) or p < 0 for p in probabilities)
        or abs(sum(probabilities) - 1.0) > 1e-10
    ):
        raise ValueError("sampling requires 32 finite normalized probabilities")
    draws = torch.multinomial(
        torch.tensor(probabilities, dtype=torch.float64),
        shots,
        replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )
    counts = torch.bincount(draws, minlength=32)
    samples = (
        Sample(key, *classical_cost(case, key), int(counts[i]))
        for i, key in enumerate(domain)
        if int(counts[i]) > 0
    )
    return tuple(
        sorted(samples, key=lambda row: (not row.feasible, row.cost, row.assignment))
    )


def run_workflow(
    case: Case, *, maxiter: int = 60, shots: int = 1000, seed: int = 2718
) -> Workflow:
    """Optimize executed expected costs, then sample the actual final state.

    Public Nelder-Mead replaces source COBYLA, and an exact local expectation
    replaces its shot estimator. Neither replacement implies trajectory parity.
    """
    if (
        isinstance(maxiter, bool)
        or not isinstance(maxiter, int)
        or not 1 <= maxiter <= 128
    ):
        raise ValueError("maxiter must be an integer in [1, 128]")
    _validate_sampling(shots, seed)
    costs = tuple(classical_cost(case, key)[0] for key in assignments(case))
    trace: list[Evaluation] = []

    def evaluate(parameters: torch.Tensor) -> torch.Tensor:
        values = tuple(float(value) for value in parameters)
        probabilities = distribution(case, build_program(case, values))
        cost = sum(p * c for p, c in zip(probabilities, costs, strict=True))
        trace.append(Evaluation(values, cost))
        return torch.tensor(cost, dtype=torch.float64)

    optimizer = NelderMeadOptimizer(maxiter=maxiter, initial_step=0.07).minimize(
        evaluate, torch.tensor(initial_parameters(case), dtype=torch.float64)
    )
    program = build_program(case, tuple(float(value) for value in optimizer.parameters))
    samples = sample_distribution(
        case, distribution(case, program), shots=shots, seed=seed
    )
    return Workflow(case, program, optimizer, tuple(trace), samples)
