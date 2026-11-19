"""Regenerate the seven-vertex benchmark and its complete diagonal cost model."""

from dataclasses import dataclass
from itertools import combinations
from random import Random

import flagquantum as fq
from flagquantum.observables import Observable

WIDTH = 7
LAYERS = 10
SHOTS = 10_000
PENALTY = 2


@dataclass(frozen=True)
class Model:
    edges: tuple[tuple[int, int], ...]
    conflicts: tuple[tuple[int, int], ...]
    energies: tuple[int, ...]
    hamiltonian: Observable
    coefficients: tuple[tuple[int, float], ...]


def decode(word: int) -> tuple[int, ...]:
    """Map native low-order lanes to the source's reversed node order."""
    return tuple(node for node in range(WIDTH) if word & (1 << (WIDTH - 1 - node)))


def build_model() -> Model:
    generator = Random(79)
    pairs = tuple(combinations(range(WIDTH), 2))
    edges = tuple(pair for pair in pairs if generator.random() < 0.6)
    conflicts = tuple(pair for pair in pairs if pair not in edges)
    selected = tuple(frozenset(decode(word)) for word in range(1 << WIDTH))
    energies = tuple(
        PENALTY
        * (2 * sum(left in row and right in row for left, right in conflicts)) ** 2
        - len(row)
        for row in selected
    )
    coefficients = tuple(
        (mask, coefficient)
        for mask in range(1 << WIDTH)
        if (
            coefficient := sum(
                energy * (-1) ** (word & mask).bit_count()
                for word, energy in enumerate(energies)
            )
            / (1 << WIDTH)
        )
    )
    operator = 0.0 * fq.I()
    for mask, coefficient in coefficients:
        monomial = fq.I()
        for bit in range(WIDTH):
            if mask & (1 << bit):
                monomial = monomial @ fq.Z(bit)
        operator += coefficient * monomial
    return Model(edges, conflicts, energies, operator, coefficients)


def initial_parameters(model: Model) -> tuple[float, ...]:
    magnitudes = sorted(abs(value) for mask, value in model.coefficients if mask)
    largest = magnitudes[len(magnitudes) // 2 :]
    time_step = len(largest) / (2 * sum(largest))
    return tuple(
        parameter
        for index in range(LAYERS)
        for parameter in (
            time_step * index / (LAYERS - 1),
            time_step * (LAYERS - 1 - index) / (LAYERS - 1),
        )
    )
