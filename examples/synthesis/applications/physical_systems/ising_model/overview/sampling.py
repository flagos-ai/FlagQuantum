"""Finite local samples for the bounded six-spin source Ising workflow."""

import math
from dataclasses import dataclass

import torch

from .workflow import energy


@dataclass(frozen=True)
class Sample:
    """One decoded source-energy row, including zero-count outcomes."""

    word: int
    spins: tuple[int, ...]
    energy: float
    count: int
    probability: float


def sample(rows: tuple[tuple[int, float], ...], seed: int) -> tuple[Sample, ...]:
    """Draw 2048 local outcomes; preserve all 64 energy rows including zero counts."""
    if (
        len(rows) != 64
        or {word for word, _ in rows} != set(range(64))
        or any(type(word) is not int for word, _ in rows)
        or any(
            type(p) not in (int, float) or not math.isfinite(p) or p < 0
            for _, p in rows
        )
        or abs(sum(p for _, p in rows) - 1) > 1e-10
        or type(seed) is not int
        or not 0 <= seed < 2**63
    ):
        raise ValueError(
            "Use all 64 distinct configurations, normalized probabilities and a nonnegative integer seed."
        )
    probabilities = torch.tensor([p for _, p in rows], dtype=torch.float64)
    draws = torch.multinomial(
        probabilities,
        2048,
        replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )
    counts = torch.bincount(draws, minlength=64)
    return tuple(
        Sample(
            word,
            tuple(2 * ((word >> i) & 1) - 1 for i in range(6)),
            energy(word),
            int(count),
            int(count) / 2048,
        )
        for (word, _), count in zip(rows, counts, strict=True)
    )
