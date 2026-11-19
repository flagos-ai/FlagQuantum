"""Fresh seeded antenna coefficients and the original maximization model."""

import math
from dataclasses import dataclass

GROUPS = ((1, 3, 4), (0, 5), (6, 9))


def fresh_coefficients() -> tuple[float, ...]:
    """Reproduce source np.random.seed(0);rand(1,13) without changing global RNG."""
    import numpy as np

    return tuple(float(value) for value in np.random.RandomState(0).rand(1, 13)[0])


def validate_coefficients(coefficients: tuple[float, ...]) -> None:
    if len(coefficients) != 13 or any(
        type(value) not in (int, float) or not math.isfinite(value)
        for value in coefficients
    ):
        raise ValueError("thirteen finite consumption coefficients are required")


@dataclass(frozen=True)
class Assignment:
    """Thirteen original decisions, selected locations, actual score and feasibility."""

    word: int
    selected: tuple[int, ...]
    score: float
    feasible: bool


def assignment(word: int, coefficients: tuple[float, ...]) -> Assignment:
    if type(word) is not int or not 0 <= word < 8192:
        raise ValueError("decision word must be an integer in0..8191")
    validate_coefficients(coefficients)
    selected = tuple(i for i in range(13) if word & (1 << i))
    score = sum(coefficients[i] for i in selected)
    feasible = len(selected) <= 7 and all(
        sum((word >> i) & 1 for i in group) == 1 for group in GROUPS
    )
    return Assignment(word, selected, score, feasible)


def residual_penalty(word: int, slack: int) -> int:
    if (
        type(word) is not int
        or not 0 <= word < 8192
        or type(slack) is not int
        or not 0 <= slack < 8
    ):
        raise ValueError("thirteen-bit decision and three-bit slack words are required")
    return (word.bit_count() + slack - 7) ** 2 + sum(
        (sum((word >> i) & 1 for i in group) - 1) ** 2 for group in GROUPS
    )
