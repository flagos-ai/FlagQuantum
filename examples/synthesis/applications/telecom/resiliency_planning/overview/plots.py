"""Plot the original topology and genuinely observed routing assignments."""

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx

from . import problem, workflow


def draw_network(word: int | None, title: str, destination: Path) -> None:
    graph = nx.Graph()
    graph.add_nodes_from(problem.NODES)
    graph.add_edges_from(problem.EDGES)
    fig, axis = plt.subplots(figsize=(8, 4.5))
    nx.draw_networkx_nodes(
        graph, problem.POSITIONS, node_color="#e6edf6", node_size=1100, ax=axis
    )
    nx.draw_networkx_labels(graph, problem.POSITIONS, ax=axis)
    for index, edge in enumerate(problem.EDGES):
        owners = (
            []
            if word is None
            else [demand for demand in range(2) if word >> (12 * demand + index) & 1]
        )
        color = ("#c33535", "#285fa4")[owners[-1]] if owners else "#b9bfc7"
        nx.draw_networkx_edges(
            graph,
            problem.POSITIONS,
            edgelist=[edge],
            edge_color=color,
            width=3 if owners else 1,
            ax=axis,
        )
    nx.draw_networkx_edge_labels(
        graph,
        problem.POSITIONS,
        edge_labels=dict(zip(problem.EDGES, problem.LATENCY, strict=True)),
        ax=axis,
    )
    axis.set_title(title)
    axis.set_axis_off()
    fig.tight_layout()
    fig.savefig(destination, dpi=140)
    plt.close(fig)


def save(result: workflow.Result, destination: Path) -> tuple[str, ...]:
    destination.mkdir(parents=True, exist_ok=True)
    names = ["topology.png", "illustrative-routes.png", "objective.png"]
    draw_network(
        None, "Original five-node network and edge latencies", destination / names[0]
    )
    illustrative = sum(1 << index for index in (1, 7, 10, 17, 18, 23, 22, 20))
    draw_network(
        illustrative,
        "Source illustrative assignment, independent of sampling",
        destination / names[1],
    )
    fig, axis = plt.subplots(figsize=(8, 4))
    axis.plot(
        range(1, len(result.trace) + 1),
        [value for _, value in result.trace],
        "o-",
        color="#285fa4",
    )
    axis.set_xlabel("Actual finite-shot objective query")
    axis.set_ylabel("Mean original cost")
    axis.set_title("Native COBYLA observations")
    fig.tight_layout()
    fig.savefig(destination / names[2], dpi=140)
    plt.close(fig)
    top = result.counts[:100]
    for rank, (word, count) in enumerate(top):
        if not problem.valid(word):
            continue
        name = f"sampled-route-{rank}.png"
        draw_network(
            word,
            f"Actual sample rank {rank}, {count}/{result.final_shots} draws",
            destination / name,
        )
        names.append(name)
    return tuple(names)
