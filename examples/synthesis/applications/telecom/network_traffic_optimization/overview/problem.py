"""Original six-node, eight-edge, three-demand traffic model."""

from itertools import product

import torch

NODES = ("A", "B", "C", "D", "E", "F")
EDGES = (
    ("A", "B"),
    ("A", "C"),
    ("B", "E"),
    ("B", "D"),
    ("C", "E"),
    ("C", "D"),
    ("D", "F"),
    ("E", "F"),
)
LATENCY = (2, 3, 2, 1, 1, 2, 2, 1)
# The original MILP c vector retains the insertion order before graph.edges().
MILP_LATENCY = (2, 2, 3, 1, 1, 2, 2, 1)
DEMANDS = (("A", "E"), ("B", "F"), ("C", "F"))
WIDTH, LAYERS, SHOTS, QUERIES = 24, 5, 10_000, 12
POSITIONS = {
    "A": (0.0, 0.6),
    "B": (1.0, 1.1),
    "C": (1.0, 0.1),
    "D": (2.0, 0.6),
    "E": (2.0, 1.1),
    "F": (3.0, 0.6),
}


def values(words: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    bits = ((words[:, None] >> torch.arange(WIDTH)) & 1).reshape(-1, 3, 8)
    objective = (bits * torch.tensor(LATENCY, dtype=torch.float64)).sum(dim=(1, 2)) / 15
    residual = torch.zeros(words.numel(), dtype=torch.int64)
    for demand, (source, target) in enumerate(DEMANDS):
        for node in NODES:
            flow = torch.full(
                (words.numel(),),
                int(node == source) - int(node == target),
                dtype=torch.int64,
            )
            for edge, (first, second) in enumerate(EDGES):
                if node == second:
                    flow += bits[:, demand, edge]
                if node == first:
                    flow -= bits[:, demand, edge]
            residual += flow.square()
    overlap = (bits.sum(dim=1) >= 2).sum(dim=1)
    return objective + 0.2 * (residual + overlap).to(torch.float64), residual, overlap


def cost(word: int) -> float:
    return float(values(torch.tensor([word], dtype=torch.int64))[0][0])


def valid(word: int) -> bool:
    _, flow, overlap = values(torch.tensor([word], dtype=torch.int64))
    return not int(flow[0]) and not int(overlap[0])


def chosen_edges(word: int, demand: int) -> tuple[tuple[str, str], ...]:
    return tuple(
        edge for index, edge in enumerate(EDGES) if word >> (8 * demand + index) & 1
    )


def classical() -> tuple[int, int, int]:
    """Enumerate the original integral flow constraints exactly."""
    choices = []
    for source, sink in DEMANDS:
        masks = []
        for mask in range(256):
            good = True
            for node in NODES:
                flow = int(node == source) - int(node == sink)
                flow += sum(
                    (mask >> i) & 1
                    for i, (_, target) in enumerate(EDGES)
                    if target == node
                )
                flow -= sum(
                    (mask >> i) & 1
                    for i, (first, _) in enumerate(EDGES)
                    if first == node
                )
                good &= flow == 0
            if good:
                masks.append(mask)
        choices.append(masks)
    feasible = []
    for assignments in product(*choices):
        if any(
            assignments[i] & assignments[j] for i in range(3) for j in range(i + 1, 3)
        ):
            continue
        word = sum(mask << (8 * i) for i, mask in enumerate(assignments))
        milp = sum(
            MILP_LATENCY[edge]
            for mask in assignments
            for edge in range(8)
            if mask >> edge & 1
        )
        latency = sum(
            LATENCY[edge]
            for mask in assignments
            for edge in range(8)
            if mask >> edge & 1
        )
        feasible.append((milp, latency, word))
    minimum = min(row[0] for row in feasible)
    # Retain the source MILP objective, including its independent coefficient order.
    selected = min(row for row in feasible if row[0] == minimum)
    return selected[2], selected[0], selected[1]
