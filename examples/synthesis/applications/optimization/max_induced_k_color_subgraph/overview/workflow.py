"""Original six-node induced two-color subgraph with actual native readout."""

import math
from dataclasses import dataclass

import torch

import flagquantum as fq
from examples.synthesis.applications.optimization.max_induced_k_color_subgraph.main import (
    DEFAULT_PARAMS,
    main,
)
from examples.synthesis.graph_optimization import (
    Evaluation,
    lower_tail_cost,
)
from flagquantum import synth
from flagquantum.algorithms import NelderMeadOptimizer, NelderMeadResult

Assignment = tuple[int, int]
NAMES = ("a", "b")
DOMAIN: tuple[Assignment, ...] = tuple((a, b) for a in range(64) for b in range(64))
EDGES = (
    (0, 1),
    (0, 2),
    (0, 4),
    (1, 2),
    (1, 3),
    (1, 5),
    (2, 3),
    (2, 4),
    (2, 5),
    (3, 4),
    (4, 5),
)


@dataclass(frozen=True)
class Sample:
    """An observed joint assignment, original model checks and retained quantum cost."""

    assignment: Assignment
    objective: int
    colored_nodes: tuple[int, ...]
    double_colored: int
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
    """Check original coloring constraints separately from the retained quantum cost."""
    if len(assignment) != 2 or any(
        isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < 64
        for value in assignment
    ):
        raise ValueError("assignment requires two six-bit unsigned integer words")
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError("count must be a nonnegative integer")
    a, b = assignment
    nodes = tuple(i for i in range(6) if (a | b) & (1 << i))
    double = (a & b).bit_count()
    directed = 2 * sum(
        bool(word & (1 << i)) and bool(word & (1 << j))
        for word in (a, b)
        for i, j in EDGES
    )
    return Sample(
        assignment,
        len(nodes),
        nodes,
        double,
        directed,
        double == directed == 0,
        -len(nodes) + double + directed**2,
        count,
    )


def build_program(parameters: tuple[float, ...] = DEFAULT_PARAMS) -> synth.Program:
    """Reuse the frozen twelve-bit quantum encoding with explicit local limits."""
    if len(parameters) != 16 or any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        for value in parameters
    ):
        raise ValueError("eight layers require sixteen finite radian parameters")
    return synth.synthesize(
        main,
        classical_inputs={"params": parameters},
        max_qubits=12,
        max_workspace=0,
        max_operations=40000,
        memory_limit_bytes=16 * 1024**2,
    )


def read_distribution(program: synth.Program) -> tuple[float, ...]:
    """Execute and read both modeled color arrays over the complete joint domain."""
    if tuple(program.registers) != NAMES or tuple(
        len(program.registers[name]) for name in NAMES
    ) != (6, 6):
        raise ValueError("induced coloring requires six-bit a/b registers")
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
    counts = torch.bincount(draws, minlength=4096)
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


def run_workflow(*, maxiter: int = 50, shots: int = 1000, seed: int = 2718) -> Workflow:
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
    uniform = sample_distribution((1 / 4096,) * 4096, shots=shots, seed=31415)
    exact = min(
        (key for key in DOMAIN if model_row(key).feasible),
        key=lambda key: (-model_row(key).objective, key),
    )
    return Workflow(program, optimizer, tuple(trace), samples, uniform, exact)
