"""The original three-donor, three-patient compatibility model."""

from dataclasses import dataclass

WEIGHTS = (0.9, 0.7, 0.6, 0.8, 0.75, 0.65, 0.85, 0.8, 0.7)


@dataclass(frozen=True)
class Assignment:
    """Compatibility, conflict penalty and feasibility stay distinct."""

    word: int
    rows: tuple[tuple[int, ...], ...]
    score: float
    conflict_pairs: int
    penalized_cost: float
    feasible: bool


def assignment(word: int) -> Assignment:
    """Evaluate all row and column collisions without consulting quantum output."""
    if type(word) is not int or not 0 <= word < 512:
        raise ValueError("matching word must be an integer in 0..511")
    rows = tuple(
        tuple((word >> (3 * donor + patient)) & 1 for patient in range(3))
        for donor in range(3)
    )
    score = sum(
        WEIGHTS[3 * donor + patient] * rows[donor][patient]
        for donor in range(3)
        for patient in range(3)
    )
    conflicts = sum(
        rows[donor][first] * rows[donor][second]
        + rows[first][donor] * rows[second][donor]
        for donor in range(3)
        for first in range(3)
        for second in range(first + 1, 3)
    )
    return Assignment(
        word, rows, score, conflicts, -score + 2 * conflicts, conflicts == 0
    )
