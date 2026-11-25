"""Regenerate the original six-cycle incidence matrix and decoding table."""

import math
from dataclasses import dataclass
from itertools import combinations

import torch

NODES = (3, 4, 1, 5, 2, 0)
EDGES = ((3, 4), (3, 2), (4, 1), (1, 5), (5, 0), (2, 0))
SIGNS = (1,) * 6
UNARY_SHOTS = 2048
DQI_SHOTS = 10000


@dataclass(frozen=True)
class DQIModel:
    """Classical inputs and all 64 word objective values."""

    columns: tuple[int, ...]
    matrix: tuple[int, ...]
    syndrome_table: tuple[tuple[int, int], ...]
    weights: tuple[float, ...]
    objective: tuple[int, ...]


def make_model() -> DQIModel:
    """Enumerate weight-0/1/2 errors and generate the optimal symmetric amplitudes."""
    columns = tuple(
        (1 << NODES.index(left)) | (1 << NODES.index(right)) for left, right in EDGES
    )
    matrix = tuple(column >> row & 1 for row in range(6) for column in columns)
    entries = []
    for weight in range(3):
        for edges in combinations(range(6), weight):
            syndrome = 0
            word = 0
            for edge in edges:
                syndrome ^= columns[edge]
                word |= 1 << edge
            entries.append((syndrome, word))
    if len({syndrome for syndrome, _ in entries}) != len(entries):
        raise ValueError("Weight-two decoder entries must be unique")
    coupling_1, coupling_2 = math.sqrt(6), math.sqrt(10)
    model = torch.tensor(
        [[0, coupling_1, 0], [coupling_1, 0, coupling_2], [0, coupling_2, 0]],
        dtype=torch.float64,
    )
    _, vectors = torch.linalg.eigh(model)
    vector = vectors[:, -1]
    if float(vector[0]) < 0:
        vector = -vector
    weights = (*map(float, vector), 0.0)
    scores = tuple(
        sum(1 if (word & column).bit_count() % 2 else -1 for column in columns)
        for word in range(64)
    )
    return DQIModel(columns, matrix, tuple(entries), weights, scores)


def dicke_amplitudes(weight: int) -> tuple[float, ...]:
    """Return the full six-bit uniform amplitude vector at one Hamming weight."""
    if not 0 <= weight <= 6:
        raise ValueError("The six-bit Dicke weight must be in 0..6")
    value = 1 / math.sqrt(math.comb(6, weight))
    return tuple(value if word.bit_count() == weight else 0.0 for word in range(64))


def majority_errors(model: DQIModel) -> int:
    """Count weight-limited errors the source illustrative majority rule misses."""
    failures = 0
    for syndrome, expected in model.syndrome_table:
        decoded = sum(
            1 << bit
            for bit, column in enumerate(model.columns)
            if syndrome & column == column
        )
        failures += decoded != expected
    return failures
