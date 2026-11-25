"""Bounded original maximum-k-vertex-cover search with actual native readout."""

import math
from dataclasses import dataclass

import torch

from examples.synthesis.applications.optimization.max_k_vertex_cover.main import main
from examples.synthesis.graph_optimization import (
    INITIAL_PARAMETERS,
    Evaluation,
    lower_tail_cost,
    read_distribution,
    selected_nodes,
)
from flagquantum import synth
from flagquantum.algorithms import NelderMeadOptimizer, NelderMeadResult


@dataclass(frozen=True)
class Sample:
    """An actual observation with separate objective, penalty and feasibility."""

    word: int
    uncovered: int
    penalized_cost: int
    feasible: bool
    count: int


@dataclass(frozen=True)
class Workflow:
    """Actual optimization and observations with a separately bounded reference."""

    program: synth.Program
    optimizer: NelderMeadResult
    trace: tuple[Evaluation, ...]
    samples: tuple[Sample, ...]
    uniform: tuple[Sample, ...]
    exact_reference_word: int

    @property
    def best_feasible(self) -> Sample | None:
        """Return the best actually observed feasible row, if one was observed."""
        return next((row for row in self.samples if row.feasible), None)


def source_edges() -> tuple[tuple[int, int], ...]:
    """Reproduce the source ten-node graph with edge probability 0.5 and seed 13."""
    import networkx as nx

    graph = nx.erdos_renyi_graph(10, 0.5, seed=13)
    return tuple((int(a), int(b)) for a, b in graph.edges())


def _assignment(word: int, edges: tuple[tuple[int, int], ...]) -> tuple[int, int, bool]:
    """Calculate uncovered edges, the original penalty and five-node feasibility."""
    if isinstance(word, bool) or not isinstance(word, int) or not 0 <= word < 1024:
        raise ValueError("assignment must be an integer in [0, 1023]")
    nodes = frozenset(selected_nodes(word, 10))
    uncovered = sum(a not in nodes and b not in nodes for a, b in edges)
    return uncovered, uncovered + 10 * (len(nodes) - 5) ** 2, len(nodes) == 5


def model_table() -> tuple[tuple[int, int, bool], ...]:
    """Evaluate every assignment using one construction of the source graph."""
    edges = source_edges()
    return tuple(_assignment(word, edges) for word in range(1024))


def build_program(parameters: tuple[float, ...] = INITIAL_PARAMETERS) -> synth.Program:
    """Reuse the frozen native program with finite angles and explicit local limits."""
    if len(parameters) != 6 or any(not math.isfinite(value) for value in parameters):
        raise ValueError("three layers require six finite radian parameters")
    return synth.synthesize(
        main,
        classical_inputs={"params": parameters},
        max_qubits=10,
        max_workspace=0,
        max_operations=5000,
        memory_limit_bytes=4 * 1024**2,
    )


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
    """Draw actual CPU observations and rank feasibility before objective and word."""
    _validate_sampling(shots, seed)
    table = model_table()
    lower_tail_cost(probabilities, tuple(row[1] for row in table), 1.0)
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
                Sample(word, *table[word], int(counts[word]))
                for word in range(1024)
                if int(counts[word]) > 0
            ),
            key=lambda row: (not row.feasible, row.penalized_cost, row.word),
        )
    )


def run_workflow(*, maxiter: int = 90, shots: int = 1000, seed: int = 2718) -> Workflow:
    """Optimize actual native CVaR at quantile 0.7, sample, then enumerate a separate reference."""
    if (
        isinstance(maxiter, bool)
        or not isinstance(maxiter, int)
        or not 1 <= maxiter <= 128
    ):
        raise ValueError("maxiter must be an integer in [1, 128]")
    _validate_sampling(shots, seed)
    table = model_table()
    costs = tuple(row[1] for row in table)
    trace: list[Evaluation] = []

    def evaluate(parameters: torch.Tensor) -> torch.Tensor:
        values = tuple(float(value) for value in parameters)
        probabilities = read_distribution(build_program(values), 10)
        cost = lower_tail_cost(probabilities, costs, 0.7)
        trace.append(Evaluation(values, cost))
        return torch.tensor(cost, dtype=torch.float64)

    optimizer = NelderMeadOptimizer(maxiter=maxiter, initial_step=0.07).minimize(
        evaluate, torch.tensor(INITIAL_PARAMETERS, dtype=torch.float64)
    )
    program = build_program(tuple(float(value) for value in optimizer.parameters))
    samples = sample_distribution(
        read_distribution(program, 10), shots=shots, seed=seed
    )
    uniform = sample_distribution((1 / 1024,) * 1024, shots=shots, seed=31415)
    exact_word = min(
        (word for word in range(1024) if table[word][2]),
        key=lambda word: (table[word][0], word),
    )
    return Workflow(program, optimizer, tuple(trace), samples, uniform, exact_word)
