"""Original five-node three-color minimum coloring with actual native readout."""

import math
from dataclasses import dataclass

import torch

import flagquantum as fq
from examples.synthesis.applications.optimization.min_graph_coloring.main import (
    DEFAULT_PARAMS,
    main,
)
from examples.synthesis.graph_optimization import (
    Evaluation,
    lower_tail_cost,
)
from flagquantum import synth
from flagquantum.algorithms import NelderMeadOptimizer, NelderMeadResult

Assignment = tuple[int]
NAMES = ("colors",)
DOMAIN: tuple[Assignment, ...] = tuple((word,) for word in range(1 << 15))
EDGES = ((0, 1), (0, 3), (0, 4), (1, 3), (2, 3))


@dataclass(frozen=True)
class Sample:
    """An observed joint assignment, original model checks and retained quantum cost."""

    assignment: Assignment
    objective: int
    node_colors: tuple[tuple[int, ...], ...]
    occupancy_penalty: int
    directed_conflicts: int
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
    exact_reference: Assignment

    @property
    def best_feasible(self) -> Sample | None:
        """Return the best actually observed feasible row, if present."""
        return next((row for row in self.samples if row.feasible), None)


def model_row(assignment: Assignment, count: int = 0) -> Sample:
    """Check original exactly-one-color and adjacency constraints separately."""
    if len(assignment) != 1 or any(
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 0 <= value < 1 << 15
        for value in assignment
    ):
        raise ValueError("assignment requires one fifteen-bit unsigned integer word")
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError("count must be a nonnegative integer")
    word = assignment[0]
    node_colors = tuple(
        tuple(c for c in range(3) if word & (1 << (5 * c + v))) for v in range(5)
    )
    occupancy = sum((len(colors) - 1) ** 2 for colors in node_colors)
    directed = 2 * sum(
        bool(word & (1 << (5 * c + i))) and bool(word & (1 << (5 * c + j)))
        for c in range(3)
        for i, j in EDGES
    )
    used = sum(bool(word & (31 << (5 * c))) for c in range(3))
    return Sample(
        assignment,
        used,
        node_colors,
        occupancy,
        directed,
        occupancy == directed == 0,
        used + 10 * occupancy + 10 * directed**2,
        count,
    )


def build_program(parameters: tuple[float, ...] = DEFAULT_PARAMS) -> synth.Program:
    """Reuse the frozen fifteen-bit quantum encoding with explicit local limits."""
    if len(parameters) != 12 or any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        for value in parameters
    ):
        raise ValueError("six layers require twelve finite radian parameters")
    return synth.synthesize(
        main,
        classical_inputs={"params": parameters},
        max_qubits=15,
        max_workspace=0,
        max_operations=40000,
        memory_limit_bytes=32 * 1024**2,
    )


def read_distribution(program: synth.Program) -> tuple[float, ...]:
    """Execute and read the modeled color array over the complete joint domain."""
    if tuple(program.registers) != NAMES or tuple(
        len(program.registers[name]) for name in NAMES
    ) != (15,):
        raise ValueError("minimum coloring requires one fifteen-bit colors register")
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
    counts = torch.bincount(draws, minlength=32768)
    return tuple(
        sorted(
            (
                model_row(key, int(counts[index]))
                for index, key in enumerate(DOMAIN)
                if int(counts[index]) > 0
            ),
            key=lambda row: (
                not row.feasible,
                row.objective,
                row.retained_quantum_cost,
                row.assignment,
            ),
        )
    )


def run_workflow(
    *, maxiter: int = 100, shots: int = 1000, seed: int = 2718
) -> Workflow:
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
        evaluate, torch.tensor(DEFAULT_PARAMS, dtype=torch.float64)
    )
    program = build_program(tuple(float(value) for value in optimizer.parameters))
    samples = sample_distribution(read_distribution(program), shots=shots, seed=seed)
    uniform = sample_distribution((1 / 32768,) * 32768, shots=shots, seed=31415)
    exact = min(
        (key for key in DOMAIN if model_row(key).feasible),
        key=lambda key: (model_row(key).objective, key),
    )
    return Workflow(program, optimizer, tuple(trace), samples, uniform, exact)
