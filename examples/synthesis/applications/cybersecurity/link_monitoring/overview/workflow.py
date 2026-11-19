"""Fixed-node minimum vertex cover, local cost search and observed solutions."""

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import networkx as nx
import torch

import flagquantum as fq
from examples.synthesis.graph_optimization import lower_tail_cost
from flagquantum import synth
from flagquantum.algorithms import NelderMeadOptimizer, NelderMeadResult

from .quantum import main

if TYPE_CHECKING:
    from matplotlib.figure import Figure

EDGES = (
    (1, 2),
    (1, 3),
    (1, 4),
    (1, 5),
    (2, 3),
    (2, 4),
    (3, 7),
    (3, 12),
    (4, 6),
    (5, 11),
    (6, 8),
    (8, 9),
    (8, 10),
)
INITIAL_PARAMETERS = (0.13, -0.27)


@dataclass(frozen=True)
class Evaluation:
    parameters: tuple[float, ...]
    cost: float


@dataclass(frozen=True)
class Sample:
    word: int
    nodes: tuple[int, ...]
    encoded_cost: float
    feasible: bool
    count: int


@dataclass(frozen=True)
class Workflow:
    program: synth.Program
    optimizer: NelderMeadResult
    trace: tuple[Evaluation, ...]
    samples: tuple[Sample, ...]

    @property
    def best_feasible(self) -> Sample | None:
        return next((row for row in self.samples if row.feasible), None)


def source_graph() -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(range(1, 13))
    graph.add_edges_from(EDGES)
    return graph


def selected_nodes(word: int) -> tuple[int, ...]:
    if type(word) is not int or not 0 <= word < 4096:
        raise ValueError("a link-monitoring word must lie in 0..4095")
    return tuple(i + 1 for i in range(12) if word & (1 << i))


def model_tables() -> tuple[tuple[float, ...], frozenset[int]]:
    costs = []
    feasible = set()
    for word in range(4096):
        nodes = selected_nodes(word)
        uncovered = sum(u not in nodes and v not in nodes for u, v in EDGES)
        costs.append(float(len(nodes) + 2 * uncovered))
        if 1 in nodes and uncovered == 0:
            feasible.add(word)
    return tuple(costs), frozenset(feasible)


def classical_optima() -> tuple[tuple[int, ...], ...]:
    """Enumerate feasible fixed-node covers as independent comparison labels."""
    _, feasible = model_tables()
    size = min(word.bit_count() for word in feasible)
    return tuple(
        selected_nodes(word) for word in sorted(feasible) if word.bit_count() == size
    )


def build_program(parameters: tuple[float, ...] = INITIAL_PARAMETERS) -> synth.Program:
    if len(parameters) != 2 or any(
        isinstance(v, bool) or not math.isfinite(v) for v in parameters
    ):
        raise ValueError("one QAOA layer requires two finite radian parameters")
    return synth.synthesize(
        main,
        classical_inputs={"parameters": parameters},
        max_qubits=12,
        max_workspace=0,
        max_operations=4096,
        memory_limit_bytes=64 * 1024**2,
    )


def read_distribution(program: synth.Program) -> tuple[float, ...]:
    if (
        tuple(program.registers) != ("target",)
        or len(program.registers["target"]) != 12
    ):
        raise ValueError("link-monitoring readout requires the twelve-bit target array")
    values = dict(
        program.probabilities(fq.run(program), registers=("target",), threshold=0)
    )
    if any(len(key) != 1 or not 0 <= key[0] < 4096 for key in values):
        raise ValueError("link-monitoring readout is outside its declared array")
    probabilities = tuple(values.get((word,), 0.0) for word in range(4096))
    lower_tail_cost(probabilities, (0.0,) * 4096, 1.0)
    if sum(p for word, p in enumerate(probabilities) if not word & 1) > 1e-22:
        raise ValueError("resource node one must remain fixed to one")
    return probabilities


def sample(
    probabilities: tuple[float, ...], *, shots: int, seed: int
) -> tuple[Sample, ...]:
    if type(shots) is not int or not 1 <= shots <= 100000:
        raise ValueError("shots must lie in 1..100000")
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("seed must be a nonnegative 63-bit integer")
    costs, feasible = model_tables()
    lower_tail_cost(probabilities, costs, 1.0)
    if sum(p for word, p in enumerate(probabilities) if not word & 1) > 1e-22:
        raise ValueError("resource node one must remain fixed to one")
    counts = torch.bincount(
        torch.multinomial(
            torch.tensor(probabilities, dtype=torch.float64),
            shots,
            replacement=True,
            generator=torch.Generator().manual_seed(seed),
        ),
        minlength=4096,
    )
    rows = (
        Sample(
            word, selected_nodes(word), costs[word], word in feasible, int(counts[word])
        )
        for word in range(4096)
        if int(counts[word]) > 0
    )
    return tuple(
        sorted(rows, key=lambda row: (not row.feasible, len(row.nodes), row.word))
    )


def run_workflow(*, maxiter: int = 60, shots: int = 1000, seed: int = 1960) -> Workflow:
    """Execute native Nelder-Mead and CVaR 0.9, then draw actual local samples.

    The source leaves its optimizer, initial angles and shots implicit. Native
    optimizer, two explicit angles and 1,000 seeded draws are declared adaptations;
    no provider trajectory equivalence. The fixed-node constraint is preserved.
    """
    if type(maxiter) is not int or not 1 <= maxiter <= 128:
        raise ValueError("maxiter must lie in 1..128")
    if type(shots) is not int or not 1 <= shots <= 100000:
        raise ValueError("shots must lie in 1..100000")
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("seed must be a nonnegative 63-bit integer")
    costs, _ = model_tables()
    trace: list[Evaluation] = []

    def evaluate(parameters: torch.Tensor) -> torch.Tensor:
        values = tuple(float(v) for v in parameters)
        value = lower_tail_cost(read_distribution(build_program(values)), costs, 0.9)
        trace.append(Evaluation(values, value))
        return torch.tensor(value, dtype=torch.float64)

    optimizer = NelderMeadOptimizer(maxiter=maxiter, initial_step=0.07).minimize(
        evaluate, torch.tensor(INITIAL_PARAMETERS, dtype=torch.float64)
    )
    program = build_program(tuple(float(v) for v in optimizer.parameters))
    rows = sample(read_distribution(program), shots=shots, seed=seed)
    return Workflow(program, optimizer, tuple(trace), rows)


def plot_graph(nodes: tuple[int, ...] = ()) -> "Figure":
    if len(set(nodes)) != len(nodes) or any(
        type(v) is not int or not 1 <= v <= 12 for v in nodes
    ):
        raise ValueError("selected resources must be distinct nodes in 1..12")
    from matplotlib import pyplot as plt

    graph = source_graph()
    figure, axis = plt.subplots(figsize=(7, 5), constrained_layout=True)
    nx.draw_networkx(
        graph,
        pos=nx.spring_layout(graph, seed=0),
        node_color=["tomato" if n in nodes else "skyblue" for n in graph],
        edge_color=[
            "gold" if u in nodes or v in nodes else "gray" for u, v in graph.edges()
        ],
        ax=axis,
    )
    axis.set_title(
        "Observed cover (node 1 fixed)" if nodes else "Original wireless sensor network"
    )
    axis.axis("off")
    return figure


def plot_trace(workflow: Workflow) -> "Figure":
    from matplotlib import pyplot as plt

    figure, axis = plt.subplots(figsize=(7, 4), constrained_layout=True)
    axis.plot([row.cost for row in workflow.trace])
    axis.set(
        xlabel="Objective evaluations",
        ylabel="Encoded cost CVaR (0.9)",
        title="Fixed-node link-monitoring cost search",
    )
    return figure


def plot_costs(workflow: Workflow) -> "Figure":
    from matplotlib import pyplot as plt

    figure, axis = plt.subplots(figsize=(7, 4), constrained_layout=True)
    axis.hist(
        [row.encoded_cost for row in workflow.samples],
        bins=20,
        weights=[
            row.count / sum(s.count for s in workflow.samples)
            for row in workflow.samples
        ],
    )
    axis.set(
        xlabel="Joint encoded cover cost",
        ylabel="Observed probability",
        title="Actual link-monitoring samples",
    )
    return figure
