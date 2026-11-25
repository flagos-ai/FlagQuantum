"""Original two-demand network resiliency objective and validity rules."""

from itertools import product

import torch

NODES = ("P1", "P2", "S1", "S2", "S3")
EDGES = (
    ("P1", "S1"),
    ("P1", "S2"),
    ("S2", "S1"),
    ("P2", "S1"),
    ("S3", "S1"),
    ("P2", "S3"),
    ("S3", "S2"),
)
DEMANDS = (("S2", "P1"), ("S2", "P2"))
LATENCY = (1, 1, 1, 1, 1, 1, 2)
WIDTH, LAYERS = 24, 20
POSITIONS = {
    "P1": (2.0, 1.0),
    "P2": (1.0, 1.0),
    "S1": (1.5, 1.0),
    "S2": (1.75, 2.0),
    "S3": (1.25, 2.0),
}


def values(words: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    bits = ((words[:, None] >> torch.arange(WIDTH)) & 1).reshape(-1, 2, 12)
    latency = (bits[:, :, :7] * torch.tensor(LATENCY, dtype=torch.float64)).sum(
        dim=(1, 2)
    )
    flow = torch.zeros(words.numel(), dtype=torch.int64)
    for demand, (source, target) in enumerate(DEMANDS):
        for index, node in enumerate(NODES):
            residual = int(node in (source, target)) - 2 * bits[:, demand, 7 + index]
            for edge, endpoints in enumerate(EDGES):
                if node in endpoints:
                    residual = residual + bits[:, demand, edge]
            flow += residual.square()
        for node in (source, target):
            flow += 1 - bits[:, demand, 7 + NODES.index(node)]
    overlap = sum(
        (
            bits[:, 0, 7 + i] * bits[:, 1, 7 + i]
            for i, node in enumerate(NODES)
            if node != "S2"
        ),
        torch.zeros(words.numel(), dtype=torch.int64),
    )
    correlation = bits[:, 0, 1] * bits[:, 1, 2] + bits[:, 0, 2] * bits[:, 1, 1]
    return (
        0.025 * latency
        + 0.25 * overlap.to(torch.float64)
        + correlation.to(torch.float64)
        + 0.125 * flow.to(torch.float64),
        flow,
        overlap,
    )


def cost(word: int) -> float:
    return float(values(torch.tensor([word], dtype=torch.int64))[0][0])


def valid(word: int) -> bool:
    _, flow, overlap = values(torch.tensor([word], dtype=torch.int64))
    return not int(flow[0]) and not int(overlap[0])


def feasible_words() -> tuple[int, ...]:
    """Enumerate each original 12-bit demand and combine exact valid assignments."""
    rows = torch.arange(4096, dtype=torch.int64)
    choices: list[list[int]] = []
    for demand, (source, target) in enumerate(DEMANDS):
        bits = (rows[:, None] >> torch.arange(12)) & 1
        good = torch.ones(4096, dtype=torch.bool)
        for index, node in enumerate(NODES):
            residual = int(node in (source, target)) - 2 * bits[:, 7 + index]
            for edge, endpoints in enumerate(EDGES):
                if node in endpoints:
                    residual = residual + bits[:, edge]
            good &= residual == 0
        for node in (source, target):
            good &= bits[:, 7 + NODES.index(node)] == 1
        choices.append([int(row) for row in rows[good]])
    return tuple(
        first | second << 12
        for first, second in product(*choices)
        if valid(first | second << 12)
    )
