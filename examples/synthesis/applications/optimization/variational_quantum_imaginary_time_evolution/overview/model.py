"""Regenerate the four-node source model, generators, and exact cost table."""

import random
from dataclasses import dataclass
from itertools import combinations

import numpy as np
from numpy.typing import NDArray

RealArray = NDArray[np.float64]
Pairs = tuple[tuple[int, int], ...]
Edge = tuple[int, int, float]
NUM_NODES = 4


@dataclass(frozen=True)
class CutModel:
    """Oriented generators, word-level spin eigenvalues, and signed costs."""

    edges: tuple[Edge, ...]
    generators: Pairs
    spins: RealArray
    parities: RealArray
    energy: RealArray


def source_model() -> CutModel:
    """Generate a complete graph for the source Newman-Watts case n=k=4."""
    generator = random.Random(0)
    pairs = tuple(combinations(range(NUM_NODES), 2))
    edges = tuple((u, v, 1.0 - generator.random()) for u, v in pairs)
    weights = tuple(
        sum(weight for u, v, weight in edges if node in (u, v))
        for node in range(NUM_NODES)
    )
    ordered = sorted(pairs, key=lambda pair: sorted(weights[node] for node in pair))
    orientations = tuple(
        (u, v) if weights[u] < weights[v] else (v, u) for u, v in ordered
    )
    spins = np.array(
        [
            [1 - 2 * (word >> node & 1) for node in range(NUM_NODES)]
            for word in range(16)
        ],
        dtype=np.float64,
    )
    parities = np.asarray(
        np.column_stack(
            [*(spins[:, u] * spins[:, v] for u, v, _ in edges), np.ones(16)]
        ),
        dtype=np.float64,
    )
    coefficients = np.array(
        [*(weight / 2 for _, _, weight in edges), -sum(w for _, _, w in edges) / 2],
        dtype=np.float64,
    )
    return CutModel(edges, orientations, spins, parities, parities @ coefficients)
