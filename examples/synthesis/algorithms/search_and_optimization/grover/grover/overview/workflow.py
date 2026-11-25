"""Native adaptive Grover searches for both source SAT formulas and graph cuts."""

import itertools
import math
from dataclasses import dataclass
from typing import Literal

import networkx as nx
import torch
from matplotlib import colormaps
from matplotlib.figure import Figure

import flagquantum as fq
from examples.synthesis.algorithms.search_and_optimization.grover.grover.main import (
    EDGES,
)
from examples.synthesis.algorithms.search_and_optimization.grover.grover.main import (
    main as graph_search,
)
from examples.synthesis.algorithms.search_and_optimization.grover.three_sat_grover.main import (
    main as large_search,
)
from flagquantum import synth
from flagquantum.compiler import ResourceEstimate
from flagquantum.synth import (
    Output,
    QArray,
    QBit,
    allocate,
    control,
    hadamard_transform,
    phase,
    qfunc,
)

Problem = Literal["small_3sat", "large_3sat", "max_cut"]
SMALL_CLAUSES = (
    (1, 2, 3),
    (-1, 2, 3),
    (-1, -2, -3),
    (-1, -2, 3),
    (1, 2, -3),
    (-1, 2, -3),
)
LARGE_CLAUSES = (
    (2, 3, 4),
    (-1, 2, 3),
    (-1, 2, -3),
    (-1, -2, 3),
    (1, -2, -3),
    (1, -2, 3),
    (-1, -2, -4),
    (-1, -2, 4),
    (-2, -3, -4),
    (2, -3, 4),
    (1, -3, 4),
    (1, -2, -4),
    (-1, -2, -3),
)


@dataclass(frozen=True)
class Outcome:
    """One observed assignment with its counts and classical predicate value."""

    bits: tuple[int, ...]
    count: int
    probability: float
    marked: bool


@dataclass(frozen=True)
class Attempt:
    """One actual native execution and its finite-shot observations."""

    loop_index: int
    repetitions: int
    seed: int
    outcomes: tuple[Outcome, ...]
    resources: ResourceEstimate


@dataclass(frozen=True)
class Search:
    """Adaptive search retaining every executed attempt and its stop decision."""

    problem: Problem
    threshold: float
    attempts: tuple[Attempt, ...]
    succeeded: bool


@qfunc
def small_search(r: int, q: Output[QArray[QBit]]) -> None:
    allocate(3, q)
    hadamard_transform(q)
    for _ in range(r):
        control(
            (q[0] | q[1] | q[2])
            & (~q[0] | q[1] | q[2])
            & (~q[0] | ~q[1] | ~q[2])
            & (~q[0] | ~q[1] | q[2])
            & (q[0] | q[1] | ~q[2])
            & (~q[0] | q[1] | ~q[2]),
            lambda: phase(1, math.pi),
        )
        hadamard_transform(q)
        phase((1 - q[0]) * (1 - q[1]) * (1 - q[2]), math.pi)
        phase(1, math.pi)
        hadamard_transform(q)


def width(problem: Problem) -> int:
    """Return the original problem width, rejecting unknown problem names."""
    if problem not in ("small_3sat", "large_3sat", "max_cut"):
        raise ValueError("use small_3sat, large_3sat or max_cut")
    return {"small_3sat": 3, "large_3sat": 4, "max_cut": 5}[problem]


def predicate(problem: Problem, bits: tuple[int, ...]) -> bool:
    """Evaluate the SAT clauses or the graph's cut-at-least-four predicate."""
    if len(bits) != width(problem) or any(
        type(b) is not int or b not in (0, 1) for b in bits
    ):
        raise ValueError("use one integer binary value per logical variable")
    if problem == "max_cut":
        return sum(bits[a] != bits[b] for a, b in EDGES) >= 4
    clauses = SMALL_CLAUSES if problem == "small_3sat" else LARGE_CLAUSES
    return all(
        any(bool(bits[abs(literal) - 1]) == (literal > 0) for literal in clause)
        for clause in clauses
    )


def truth_table(problem: Problem) -> tuple[tuple[tuple[int, ...], bool], ...]:
    """Enumerate source-order logical variables with the first variable outermost."""
    return tuple(
        (bits, predicate(problem, bits))
        for bits in itertools.product((0, 1), repeat=width(problem))
    )


def build_program(problem: Problem, repetitions: int) -> synth.Program:
    """Bind a bounded repetition count and require zero clean workspace.

    Examples:
        >>> program = build_program("small_3sat", 1)
        >>> program.report["resources"].n_qubits
        3
    """
    n = width(problem)
    if type(repetitions) is not int or not 0 <= repetitions <= 3:
        raise ValueError("use an integer repetition count in 0..3")
    entry = {
        "small_3sat": small_search,
        "large_3sat": large_search,
        "max_cut": graph_search,
    }[problem]
    return synth.synthesize(
        entry,
        classical_inputs={"r": repetitions},
        max_qubits=n,
        max_workspace=0,
        memory_limit_bytes=1024**2,
    )


def observed_outcomes(problem: Problem, counts: torch.Tensor) -> tuple[Outcome, ...]:
    """Sort actually observed words by count, breaking ties by packed word."""
    n = width(problem)
    if (
        counts.shape != (1 << n,)
        or counts.dtype != torch.int64
        or bool((counts < 0).any())
        or int(counts.sum()) == 0
    ):
        raise ValueError("use a nonempty int64 count vector for the complete basis")
    shots = int(counts.sum())
    rows = []
    for word, count in enumerate(counts.tolist()):
        if count:
            bits = tuple(word >> i & 1 for i in range(n))
            rows.append(Outcome(bits, count, count / shots, predicate(problem, bits)))
    return tuple(
        sorted(
            rows,
            key=lambda row: (-row.count, sum(b << i for i, b in enumerate(row.bits))),
        )
    )


def exceeds_threshold(outcomes: tuple[Outcome, ...], threshold: float) -> bool:
    """Stop only when one marked row strictly exceeds the source threshold."""
    if (
        isinstance(threshold, bool)
        or not math.isfinite(threshold)
        or not 0 <= threshold <= 1
    ):
        raise ValueError("use a finite probability threshold in 0..1")
    return any(row.marked and row.probability > threshold for row in outcomes)


def run_search(
    problem: Problem,
    *,
    shots: int = 1000,
    threshold: float | None = None,
    max_iter: int = 4,
    seed: int = 1100,
) -> Search:
    """Run the source's bounded adaptive loop with explicit local shot seeds.

    Duplicate repetition counts consume a loop iteration without a quantum
    execution. Failure returns the final actual attempt, rather than claiming
    that a likely marked subspace guarantees an individual-row threshold.
    """
    width(problem)
    if type(shots) is not int or not 1 <= shots <= 100000:
        raise ValueError("use an integer shot count in 1..100000")
    if type(max_iter) is not int or not 1 <= max_iter <= 4:
        raise ValueError("use an integer loop budget in 1..4")
    if type(seed) is not int or not 0 <= seed <= 2**32 - 5:
        raise ValueError("use an integer seed in 0..2**32-5")
    limit = (0.1 if problem == "max_cut" else 0.45) if threshold is None else threshold
    exceeds_threshold((), limit)
    attempts = []
    previous = None
    for i in range(max_iter):
        repetitions = math.ceil(math.pi / 4 * math.sqrt(2**i))
        if repetitions == previous:
            continue
        program = build_program(problem, repetitions)
        values = dict(program.probabilities(fq.run(program), registers=("q",)))
        probabilities = torch.tensor(
            [values.get((word,), 0.0) for word in range(1 << width(problem))],
            dtype=torch.float64,
        )
        draws = torch.multinomial(
            probabilities,
            shots,
            replacement=True,
            generator=torch.Generator().manual_seed(seed + i),
        )
        outcomes = observed_outcomes(
            problem, torch.bincount(draws, minlength=len(probabilities))
        )
        resources = program.report["resources"]
        assert isinstance(resources, ResourceEstimate)
        attempts.append(Attempt(i, repetitions, seed + i, outcomes, resources))
        if exceeds_threshold(outcomes, limit):
            return Search(problem, limit, tuple(attempts), True)
        previous = repetitions
    return Search(problem, limit, tuple(attempts), False)


def run_workflow() -> tuple[Search, ...]:
    """Run all three source problems with 1000 actual quantum samples per attempt."""
    problems: tuple[Problem, ...] = ("small_3sat", "large_3sat", "max_cut")
    return tuple(
        run_search(problem, seed=1100 + 100 * i) for i, problem in enumerate(problems)
    )


def plot_graph(search: Search) -> Figure:
    """Show the source planar graph and the most observed sampled partition."""
    if (
        search.problem != "max_cut"
        or not search.attempts
        or not search.attempts[-1].outcomes
    ):
        raise ValueError("use a completed max_cut search with observed outcomes")
    bits = search.attempts[-1].outcomes[0].bits
    predicate("max_cut", bits)
    graph = nx.Graph()
    graph.add_nodes_from(range(5))
    graph.add_edges_from(EDGES)
    positions = nx.planar_layout(graph)
    figure = Figure(figsize=(10, 4), constrained_layout=True)
    axes = figure.subplots(1, 2)
    nx.draw_networkx(
        graph, pos=positions, ax=axes[0], with_labels=True, alpha=0.8, node_size=500
    )
    axes[0].set_title("Original six-edge graph")
    nx.draw_networkx(
        graph,
        pos=positions,
        ax=axes[1],
        with_labels=True,
        alpha=0.8,
        node_size=500,
        node_color=list(bits),
        width=[float(bits[a] != bits[b]) + 0.5 for a, b in graph.edges],
        cmap=colormaps["rainbow"],
        vmin=0,
        vmax=1,
    )
    cut = sum(bits[a] != bits[b] for a, b in EDGES)
    axes[1].set_title(f"Most observed partition: cut {cut}, marked {cut >= 4}")
    for axis in axes:
        axis.set_axis_off()
    return figure
