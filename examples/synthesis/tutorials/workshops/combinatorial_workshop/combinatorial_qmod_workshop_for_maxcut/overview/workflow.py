"""Original MaxCut Workshop graph, bounded native optimization and observations."""

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import networkx as nx
import torch

import flagquantum as fq
from examples.synthesis.tutorials.workshops.combinatorial_workshop.combinatorial_qmod_workshop_for_maxcut.comparisons.circuit import (
    reference_circuit,
)
from examples.synthesis.tutorials.workshops.combinatorial_workshop.combinatorial_qmod_workshop_for_maxcut.main import (
    main,
)
from flagquantum import synth
from flagquantum.algorithms import CobylaOptimizer
from flagquantum.compiler import ResourceEstimate

if TYPE_CHECKING:
    from matplotlib.figure import Figure

EDGES = ((0, 1), (0, 2), (1, 2), (1, 3), (2, 4), (3, 4))


@dataclass(frozen=True)
class Evaluation:
    parameters: tuple[float, ...]
    cost: float


@dataclass(frozen=True)
class Sample:
    word: int
    groups: tuple[int, ...]
    cost: int
    count: int
    probability: float


@dataclass(frozen=True)
class Workflow:
    program: synth.Program
    trace: tuple[Evaluation, ...]
    selected: Evaluation
    samples: tuple[Sample, ...]
    termination: Literal["evaluation_budget", "optimizer_returned"]
    converged: bool

    @property
    def best_observed(self) -> Sample:
        return self.samples[0]


@dataclass(frozen=True)
class StateCheck:
    parameters: tuple[float, ...]
    maximum_state_error: float
    dirty_workspace_probability: float
    resources: ResourceEstimate


def initial_parameters() -> tuple[float, ...]:
    """Preserve the source's 0..1-radian gamma/beta schedule without pi scaling."""
    return tuple(value for i in range(4) for value in (i / 3, 1 - i / 3))


def source_graph() -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(range(5))
    graph.add_edges_from(EDGES)
    return graph


def group_bits(word: int) -> tuple[int, ...]:
    if type(word) is not int or not 0 <= word < 32:
        raise ValueError("a five-node assignment must be a word in 0..31")
    return tuple((word >> i) & 1 for i in range(5))


def cut_cost(word: int) -> int:
    groups = group_bits(word)
    return -sum(groups[a] != groups[b] for a, b in EDGES)


def classical_optima() -> tuple[int, ...]:
    """Enumerate comparison labels without supplying any sampled answer."""
    minimum = min(cut_cost(word) for word in range(32))
    return tuple(word for word in range(32) if cut_cost(word) == minimum)


def build_program(parameters: tuple[float, ...]) -> synth.Program:
    if len(parameters) != 8 or any(
        isinstance(v, bool) or not math.isfinite(v) for v in parameters
    ):
        raise ValueError("four MaxCut layers require eight finite radian parameters")
    return synth.synthesize(
        main,
        classical_inputs={"params": parameters},
        max_qubits=5,
        max_workspace=0,
        max_operations=2048,
        memory_limit_bytes=4 * 1024**2,
    )


def distribution(program: synth.Program) -> tuple[float, ...]:
    if tuple(program.registers) != ("v",) or len(program.registers["v"]) != 5:
        raise ValueError("MaxCut readout requires the five-node v array")
    values = dict(program.probabilities(fq.run(program), registers=("v",), threshold=0))
    if any(len(key) != 1 or not 0 <= key[0] < 32 for key in values):
        raise ValueError("MaxCut readout contains an out-of-bounds assignment")
    probabilities = tuple(values.get((word,), 0.0) for word in range(32))
    mean_cost(probabilities)
    return probabilities


def mean_cost(probabilities: tuple[float, ...]) -> float:
    if (
        len(probabilities) != 32
        or any(not math.isfinite(p) or p < 0 for p in probabilities)
        or abs(sum(probabilities) - 1) > 1e-10
    ):
        raise ValueError("require 32 finite nonnegative normalized probabilities")
    return sum(
        probability * cut_cost(word) for word, probability in enumerate(probabilities)
    )


def sample(
    probabilities: tuple[float, ...], *, shots: int, seed: int
) -> tuple[Sample, ...]:
    if type(shots) is not int or not 1 <= shots <= 100000:
        raise ValueError("shots must lie in 1..100000")
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("seed must be a nonnegative 63-bit integer")
    mean_cost(probabilities)
    counts = torch.bincount(
        torch.multinomial(
            torch.tensor(probabilities, dtype=torch.float64),
            shots,
            replacement=True,
            generator=torch.Generator().manual_seed(seed),
        ),
        minlength=32,
    )
    rows = (
        Sample(
            word,
            group_bits(word),
            cut_cost(word),
            int(counts[word]),
            probabilities[word],
        )
        for word in range(32)
        if int(counts[word]) > 0
    )
    return tuple(sorted(rows, key=lambda row: (row.cost, row.word)))


class _EvaluationBudgetError(Exception):
    """Stop before executing beyond the source's objective-evaluation budget."""


def run_workflow(
    *, evaluation_budget: int = 40, shots: int = 1000, seed: int = 1970
) -> Workflow:
    """Preserve actual source objective calls, not native trust-region iterations.

    Native COBYLA uses exact local mean probabilities instead of provider noisy
    estimates. Select the best actually evaluated parameters when the callback
    cap stops the solver, recording nonconvergence without fabricating a result.
    Final readout and independent verification are outside the objective cap.
    """
    if type(evaluation_budget) is not int or not 1 <= evaluation_budget <= 200:
        raise ValueError("evaluation budget must lie in 1..200")
    if type(shots) is not int or not 1 <= shots <= 100000:
        raise ValueError("shots must lie in 1..100000")
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("seed must be a nonnegative 63-bit integer")
    trace: list[Evaluation] = []

    def evaluate(parameters: torch.Tensor) -> torch.Tensor:
        if len(trace) >= evaluation_budget:
            raise _EvaluationBudgetError()
        values = tuple(float(v) for v in parameters)
        value = mean_cost(distribution(build_program(values)))
        trace.append(Evaluation(values, value))
        return torch.tensor(value, dtype=torch.float64)

    termination: Literal["evaluation_budget", "optimizer_returned"] = (
        "optimizer_returned"
    )
    converged = False
    try:
        result = CobylaOptimizer(maxiter=200, rhobeg=1.0, rhoend=1e-8).minimize(
            evaluate, torch.tensor(initial_parameters(), dtype=torch.float64)
        )
        converged = result.converged
    except _EvaluationBudgetError:
        termination = "evaluation_budget"
    selected = min(trace, key=lambda row: row.cost)
    program = build_program(selected.parameters)
    observed = sample(distribution(program), shots=shots, seed=seed)
    return Workflow(program, tuple(trace), selected, observed, termination, converged)


def verify(parameters: tuple[float, ...]) -> StateCheck:
    program = build_program(parameters)
    actual = fq.run(program).statevector()
    expected = fq.run(reference_circuit(parameters)).statevector()
    assert isinstance(actual, torch.Tensor) and isinstance(expected, torch.Tensor)
    logical = program.registers["v"]
    width = actual.numel().bit_length() - 1
    hidden = tuple(w for w in range(width) if w not in logical)
    state = actual.reshape((2,) * width).permute((*logical, *hidden)).reshape(32, -1)
    error = float((state[:, 0] - expected.flatten()).abs().max())
    dirty = float(state[:, 1:].abs().square().sum())
    torch.testing.assert_close(state[:, 0], expected.flatten(), atol=3e-11, rtol=0)
    assert dirty < 1e-22
    resources = program.report["resources"]
    assert isinstance(resources, ResourceEstimate) and resources.n_qubits == 5
    return StateCheck(parameters, error, dirty, resources)


def plot_graph(word: int | None = None) -> "Figure":
    from matplotlib import pyplot as plt

    groups = group_bits(word) if word is not None else None
    graph = source_graph()
    figure, axis = plt.subplots(figsize=(6, 4), constrained_layout=True)
    colors = (
        ["tomato" if bit == 0 else "lightgreen" for bit in groups]
        if groups
        else "skyblue"
    )
    nx.draw_networkx(
        graph,
        pos=nx.planar_layout(graph),
        node_color=colors,
        alpha=0.8,
        node_size=500,
        ax=axis,
    )
    axis.set_title(
        "Best observed MaxCut" if groups else "Original five-node MaxCut graph"
    )
    axis.axis("off")
    return figure


def plot_trace(workflow: Workflow) -> "Figure":
    from matplotlib import pyplot as plt

    figure, axis = plt.subplots(figsize=(7, 4), constrained_layout=True)
    axis.plot(range(1, len(workflow.trace) + 1), [row.cost for row in workflow.trace])
    axis.set(
        xlabel="Objective evaluations",
        ylabel="Exact mean negative cut",
        title="Native MaxCut Workshop search",
    )
    return figure
