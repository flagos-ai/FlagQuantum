"""Actual finite counts and the source literal MSB-first marginal formula."""

import math
from dataclasses import dataclass

import torch

from .quantum import Basis, distribution

SHOTS = 1000
TARGET = (0.6, 1.0, 0.2)
BASES: tuple[Basis, ...] = ("X", "Y", "Z")


@dataclass(frozen=True)
class Measurement:
    basis: Basis
    counts: tuple[int, ...]

    @property
    def marginals(self) -> tuple[float, float]:
        a, b, c, d = self.counts
        return (a + b - c - d) / SHOTS, (a + c - b - d) / SHOTS


def sample(probabilities: tuple[float, ...], basis: Basis, seed: int) -> Measurement:
    if (
        len(probabilities) != 4
        or any(not math.isfinite(p) or p < 0 for p in probabilities)
        or abs(sum(probabilities) - 1) > 1e-10
    ):
        raise ValueError("measurement requires four normalized finite probabilities")
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("seed must be a nonnegative 63-bit integer")
    if basis not in ("X", "Y", "Z"):
        raise ValueError("measurement basis must be X, Y or Z")
    draws = torch.multinomial(
        torch.tensor(probabilities, dtype=torch.float64),
        SHOTS,
        replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )
    counts = tuple(int(value) for value in torch.bincount(draws, minlength=4))
    return Measurement(basis, counts)


@dataclass(frozen=True)
class Estimate:
    measurements: tuple[Measurement, ...]
    position: tuple[float, ...]

    @property
    def cost(self) -> float:
        return sum(
            (target - value) ** 2
            for target, value in zip(TARGET, self.position, strict=True)
        )


def estimate(parameters: tuple[float, ...], seed: int) -> Estimate:
    readings = tuple(
        sample(distribution(parameters, basis), basis, seed + i)
        for i, basis in enumerate(BASES)
    )
    position = tuple(sum(reading.marginals) for reading in readings)
    return Estimate(readings, position)
