"""Regenerate the original rectangle packing geometry and its Ising penalties."""

from dataclasses import dataclass
from itertools import combinations
from typing import TypeAlias

import torch

WIDTH = 4
HEIGHT = 2
RECTANGLES = ((1, 1), (2, 2), (2, 1))
PENALTY = 100
Placement: TypeAlias = tuple[int, int, int]
Term: TypeAlias = tuple[tuple[int, ...], float]


@dataclass(frozen=True)
class PackingModel:
    """Placement variables ordered from least to most significant bit."""

    placements: tuple[Placement, ...]
    terms: tuple[Term, ...]
    cost: torch.Tensor


def make_model() -> PackingModel:
    """Eliminate fixed out-of-container variables and retain every pair penalty."""
    placements = tuple(
        (rectangle, x, y)
        for rectangle, (width, height) in enumerate(RECTANGLES)
        for x in range(WIDTH - width + 1)
        for y in range(HEIGHT - height + 1)
    )[::-1]
    size = len(placements)
    cells = []
    for rectangle, x, y in placements:
        width, height = RECTANGLES[rectangle]
        cells.append({(x + a, y + b) for a in range(width) for b in range(height)})
    fields = [0.5] * size
    constant = -size / 2
    couplings: list[Term] = []
    words = torch.arange(1 << size, dtype=torch.int64)
    bits = (words[:, None] >> torch.arange(size, dtype=torch.int64)) & 1
    cost = -bits.sum(dim=1).to(torch.float64)
    for i, j in combinations(range(size), 2):
        multiplicity = int(placements[i][0] == placements[j][0]) + len(
            cells[i] & cells[j]
        )
        if not multiplicity:
            continue
        penalty = PENALTY * multiplicity
        coefficient = penalty / 4
        fields[i] -= coefficient
        fields[j] -= coefficient
        constant += coefficient
        couplings.append(((i, j), coefficient))
        cost += penalty * bits[:, i] * bits[:, j]
    terms = (
        ((), constant),
        *(((i,), value) for i, value in enumerate(fields)),
        *couplings,
    )
    return PackingModel(placements, terms, cost)


def unpack(model: PackingModel, word: int) -> tuple[Placement, ...]:
    """Return all active placements without dropping conflicting observations."""
    if not 0 <= word < 1 << len(model.placements):
        raise ValueError("Packing word is outside the model width")
    return tuple(
        placement for bit, placement in enumerate(model.placements) if word >> bit & 1
    )


def constraint_cost(positions: tuple[Placement, ...]) -> int:
    """Evaluate the original constraints directly by counting occupied cells."""
    copies = [0] * len(RECTANGLES)
    grid = [[0] * WIDTH for _ in range(HEIGHT)]
    for rectangle, x, y in positions:
        copies[rectangle] += 1
        width, height = RECTANGLES[rectangle]
        for row in range(y, y + height):
            for column in range(x, x + width):
                grid[row][column] += 1
    clashes = sum(count * (count - 1) // 2 for count in copies) + sum(
        count * (count - 1) // 2 for row in grid for count in row
    )
    return PENALTY * clashes - len(positions)
