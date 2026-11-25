"""Retain the nine-node graph, one Grover iteration and sixteen sampled rows."""

from collections import Counter
from random import Random
from typing import TYPE_CHECKING, TypeAlias

import flagquantum as fq
from examples.synthesis.tutorials.basics.grover_graph_coloring.main import main
from flagquantum import synth

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

Node: TypeAlias = str | int
COLORS = ("red", "blue", "orange", "green")
VARIABLES = ("A", "B", "C", "D", "E")
EDGES: tuple[tuple[Node, Node], ...] = (
    (0, "A"),
    (0, "B"),
    (0, "C"),
    ("A", "B"),
    ("A", 0),
    ("A", 1),
    ("B", "C"),
    ("B", "E"),
    ("B", 2),
    ("C", "D"),
    ("C", "E"),
    ("D", 1),
    ("D", 2),
    ("E", 3),
)
POSITIONS: dict[Node, tuple[int, int]] = {
    "A": (0, 2),
    "B": (-1, 1),
    "C": (0, 1),
    "D": (1, 1),
    "E": (0, 0),
    0: (-1, 2),
    1: (1, 2),
    2: (-1, 0),
    3: (1, 0),
}


def valid(values: tuple[int, ...]) -> bool:
    a, b, c, d, e = values
    return (
        a not in {0, 1}
        and a != b
        and b not in {0, 2}
        and b != c
        and b != e
        and c != 0
        and c != d
        and c != e
        and d not in {1, 2}
        and e != 3
    )


def run_workflow() -> tuple[synth.Program, Counter[tuple[int, ...]]]:
    program = synth.synthesize(
        main,
        classical_inputs={"r": 1},
        max_qubits=24,
        max_workspace=8,
        memory_limit_bytes=64 * 1024**2,
    )
    probabilities = dict(
        program.probabilities(fq.run(program), registers=("a", "b", "c", "d", "e"))
    )
    # The source leaves the cloud shot count unspecified. Bind 1000 local shots.
    counts = Counter(
        Random(17).choices(
            list(probabilities), weights=list(probabilities.values()), k=1000
        )
    )
    for values, count in counts.most_common(16):
        print(values, "count=", count, "valid=", valid(values))
    print("Resources:", program.report["resources"])
    return program, counts


def draw_graph(axes: "Axes", values: tuple[int, ...] | None = None) -> None:
    # The source's undirected graph deduplicates the repeated A--0 edge.
    drawn: set[frozenset[Node]] = set()
    for first, second in EDGES:
        edge = frozenset((first, second))
        if edge in drawn:
            continue
        drawn.add(edge)
        x1, y1 = POSITIONS[first]
        x2, y2 = POSITIONS[second]
        axes.plot((x1, x2), (y1, y2), color="black", linewidth=1, zorder=1)
    nodes: tuple[Node, ...] = (*VARIABLES, 0, 1, 2, 3)
    for index, node in enumerate(nodes):
        x, y = POSITIONS[node]
        color = (
            COLORS[node]
            if isinstance(node, int)
            else ("black" if values is None else COLORS[values[index]])
        )
        axes.scatter([x], [y], s=400, color=color, zorder=2)
        axes.text(x, y, str(node), ha="center", va="center", color="white", zorder=3)
    axes.margins(0.15)
    axes.set_axis_off()


def plot_graph() -> "Figure":
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(figsize=(2, 2))
    draw_graph(axes)
    plt.show()
    return figure


def plot_samples(counts: Counter[tuple[int, ...]]) -> "Figure":
    import matplotlib.pyplot as plt

    rows = counts.most_common(16)
    if len(rows) != 16:
        raise ValueError("Need sixteen distinct sampled rows to fill the source grid.")
    figure, panels = plt.subplots(4, 4, figsize=(12, 12))
    for index, (axes, (values, count)) in enumerate(
        zip(panels.flat, rows, strict=True)
    ):
        draw_graph(axes, values)
        axes.set_title(f"Row {index+1}: count={count}, valid={valid(values)}")
    figure.tight_layout()
    plt.show()
    return figure


def run_example() -> None:
    plot_graph()
    _, counts = run_workflow()
    plot_samples(counts)


if __name__ == "__main__":
    run_example()
