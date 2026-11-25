"""Construct the bounded inhomogeneous linear-system encoding for this example."""

import math
from dataclasses import dataclass

import torch

ORDER = 7
GRID = tuple(index / 10 for index in range(11))
INITIAL = (1.0, 1.0)
OFFSET = (0.0, 0.0)
GENERATOR = ((0.0, 1.0), (-1.0, 0.0))


@dataclass(frozen=True)
class Encoding:
    """The source's Taylor weights, selector and controlled preparation."""

    weights: tuple[float, ...]
    scale: float
    householder: tuple[tuple[float, ...], ...]


def taylor_encoding(time: float) -> Encoding:
    """Compute the eight fresh coefficients and source Householder completion."""
    if type(time) not in (int, float) or not math.isfinite(time) or not 0 <= time <= 1:
        raise ValueError("Time must be finite and lie in the source interval [0, 1]")
    norm = math.hypot(*INITIAL)
    weights = tuple(
        norm * time**order / math.factorial(order) for order in range(ORDER + 1)
    )
    scale = math.fsum(weights)
    amplitudes = torch.tensor(
        tuple(math.sqrt(weight / scale) for weight in weights), dtype=torch.float64
    )
    householder = torch.eye(len(weights), dtype=torch.float64)
    if time:
        direction = amplitudes.clone()
        direction[0] -= 1
        householder -= (
            2 * torch.outer(direction, direction) / torch.dot(direction, direction)
        )
    return Encoding(
        weights, scale, tuple(tuple(map(float, row)) for row in householder)
    )


def exact_solution(time: float) -> tuple[float, float]:
    """Solve the original initial-value problem in closed form."""
    return math.cos(time) + math.sin(time), math.cos(time) - math.sin(time)
