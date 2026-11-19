"""Generate the fresh four-city directed path objective."""

from dataclasses import dataclass
from itertools import pairwise, permutations

import torch

NUM_CITIES = 4
NONEDGE = 5
EDGES = ((0, 1, 1), (1, 2, 1), (2, 1, 1), (2, 3, 1), (0, 2, 2), (1, 3, 2))


@dataclass(frozen=True)
class RouteModel:
    """The independent cost and assignment residual on every 16-bit word."""

    distances: tuple[float, ...]
    cost: torch.Tensor
    residual: torch.Tensor


def make_model() -> RouteModel:
    """Use the three open-path transitions and enforce one city per slot."""
    edge_weights = {(left, right): weight for left, right, weight in EDGES}
    distances = tuple(
        float(edge_weights.get((left, right), NONEDGE))
        for left in range(NUM_CITIES)
        for right in range(NUM_CITIES)
    )
    words = torch.arange(1 << 16, dtype=torch.int64)
    assignments = ((words[:, None] >> torch.arange(16)) & 1).reshape(-1, 4, 4)
    residual = (assignments.sum(dim=1) - 1).square().sum(dim=1) + (
        assignments.sum(dim=2) - 1
    ).square().sum(dim=1)
    values = assignments.to(torch.float64)
    travel = torch.einsum(
        "ncs,cd,nds->n",
        values[:, :, :3],
        torch.tensor(distances, dtype=torch.float64).reshape(4, 4),
        values[:, :, 1:],
    )
    return RouteModel(distances, travel + residual, residual)


def visits(word: int) -> tuple[tuple[int, ...], ...]:
    """Restore the original rows of cities and columns of slots."""
    if not 0 <= word < 1 << 16:
        raise ValueError("Assignment word is outside the 16-bit model")
    return tuple(
        tuple((word >> (4 * city + slot)) & 1 for slot in range(4)) for city in range(4)
    )


def route(model: RouteModel, word: int) -> tuple[int, ...] | None:
    """Read a travel path only when the observed assignment is valid."""
    if int(model.residual[word]) != 0:
        return None
    table = visits(word)
    return tuple(
        next(city for city in range(4) if table[city][slot]) for slot in range(4)
    )


def exact_path(model: RouteModel) -> tuple[float, tuple[int, ...]]:
    """Enumerate exactly the 24 feasible paths, including nonedge penalties."""
    return min(
        (
            sum(model.distances[4 * left + right] for left, right in pairwise(path)),
            path,
        )
        for path in permutations(range(4))
    )
