"""Run the four-node walk and distinguish local samples from archived hardware."""

import json
from collections import Counter
from pathlib import Path
from random import Random
from typing import TYPE_CHECKING, TypedDict, cast

import networkx as nx
import torch

import flagquantum as fq
from examples.synthesis.tutorials.basics.quantumwalk_complex_network.main import (
    build_program,
)
from flagquantum import synth

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


class ArchivedResults(TypedDict):
    ideal: list[list[float]]
    hardware_agnostic_counts: list[list[int]]
    hardware_aware_counts: list[list[int]]


def make_graph(nodes: int = 4) -> nx.Graph:
    """Retain the source's four-node and eight-node graph generation settings."""
    if type(nodes) is not int or nodes not in (4, 8):
        raise ValueError("Use the source graph sizes 4 or 8.")
    return nx.connected_watts_strogatz_graph(
        n=nodes, k=nodes // 2, p=0.2, tries=100, seed=312
    )


def neighbor_probabilities(graph: nx.Graph) -> torch.Tensor:
    nodes = len(graph)
    if nodes not in (4, 8) or set(graph.nodes) != set(range(nodes)):
        raise ValueError("Need consecutive node labels on a four- or eight-node graph.")
    probabilities = torch.zeros((nodes, nodes), dtype=torch.float64)
    for node in range(nodes):
        neighbors = tuple(graph.neighbors(node))
        if not neighbors:
            raise ValueError("The walk requires a nonempty neighbor set for each node.")
        probabilities[node, list(neighbors)] = 1 / len(neighbors)
    return probabilities


def run_workflow() -> tuple[synth.Program, Counter[int]]:
    graph = make_graph()
    expected = torch.tensor(
        ((0, 1 / 3, 1 / 3, 1 / 3), (0.5, 0, 0.5, 0), (0.5, 0.5, 0, 0), (1, 0, 0, 0)),
        dtype=torch.float64,
    )
    # The existing native helper binds these four neighbor vectors. Fail if a
    # dependency changes graph generation rather than simulating another graph.
    if not torch.equal(neighbor_probabilities(graph), expected):
        raise ValueError("Generated graph differs from the bound native walk graph.")
    print("Number of nodes:", len(graph), "position qubits:", 2)
    program = build_program(steps=3)
    probabilities = dict(program.probabilities(fq.run(program), registers=("x",)))
    # The source leaves the cloud shot count unspecified; choose 1000 local
    # samples with an explicit seed. Each key is a scalar position node.
    outcomes = Random(17).choices(
        [key[0] for key in probabilities], weights=list(probabilities.values()), k=1000
    )
    counts = Counter(outcomes)
    print("Local position counts:", sorted(counts.items()))
    print("Resources:", program.report["resources"])
    return program, counts


def draw_graph(axes: "Axes", nodes: int) -> None:
    graph = make_graph(nodes)
    nx.draw(
        graph,
        pos=nx.spring_layout(graph, seed=312),
        ax=axes,
        with_labels=True,
        node_color="darkblue",
        edge_color="black",
        font_color="white",
        font_size=10,
    )


def plot_graph() -> "Figure":
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(figsize=(3, 3))
    draw_graph(axes, 4)
    figure.tight_layout()
    plt.show()
    return figure


def plot_samples(counts: Counter[int]) -> "Figure":
    import matplotlib.pyplot as plt

    if not counts or any(type(k) is not int or k not in range(4) for k in counts):
        raise ValueError("Need sampled position nodes in 0..3.")
    if (
        any(type(v) is not int or v < 0 for v in counts.values())
        or sum(counts.values()) <= 0
    ):
        raise ValueError(
            "Sample counts must be nonnegative integers with positive total."
        )
    figure, axes = plt.subplots()
    axes.bar(range(4), [counts[node] / sum(counts.values()) for node in range(4)])
    axes.set(
        xlabel="Position node x", ylabel="Local sampled probability", xticks=range(4)
    )
    figure.tight_layout()
    plt.show()
    return figure


def archived_results(nodes: int) -> ArchivedResults:
    if type(nodes) is not int or nodes not in (4, 8):
        raise ValueError("Use archived graph sizes 4 or 8.")
    data = cast(
        dict[str, ArchivedResults],
        json.loads(Path(__file__).with_name("archived_results.json").read_text()),
    )
    return data[str(nodes)]


def plot_archived_results() -> "Figure":
    """Plot preserved Torino observations; do not submit a new hardware job."""
    import matplotlib.pyplot as plt

    data = archived_results(8)
    figure, panels = plt.subplots(1, 5, figsize=(18, 4))
    draw_graph(panels[0], 8)
    panels[0].set_title("Source eight-node graph")
    for index in range(4):
        axes = panels[index + 1]
        axes.bar(
            [x - 0.2 for x in range(8)],
            data["ideal"][index],
            width=0.2,
            color="red",
            edgecolor="black",
            label="Archived ideal",
        )
        axes.bar(
            range(8),
            [x / 10000 for x in data["hardware_agnostic_counts"][index]],
            width=0.2,
            color="lightcyan",
            edgecolor="darkblue",
            label="Torino: HW agnostic",
        )
        axes.bar(
            [x + 0.2 for x in range(8)],
            [x / 10000 for x in data["hardware_aware_counts"][index]],
            width=0.2,
            color="cyan",
            edgecolor="darkblue",
            label="Torino: HW aware",
        )
        axes.set(
            title=f"t={index+1}", xlabel="x", ylabel="P", ylim=(0, 0.3), xticks=range(8)
        )
        axes.legend(fontsize=8, frameon=False)
    for index, axes in enumerate(panels):
        axes.text(-0.15, 1.08, f"({chr(97+index)})", transform=axes.transAxes)
    figure.suptitle("Historical source data; no new hardware execution")
    figure.tight_layout()
    plt.show()
    return figure


def run_example() -> None:
    plot_graph()
    _, counts = run_workflow()
    plot_samples(counts)
    plot_archived_results()


if __name__ == "__main__":
    run_example()
