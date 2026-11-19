"""Fresh full nonlinear three-facility, four-customer assignment model."""

from dataclasses import dataclass

WEIGHTS = (0.02, 0.14, 0.62, 0.11, 0.99, 0.22, 0.91, 0.09, 0.4, 0.76, 0.95, 0.61)
FACILITIES = ("Japan", "USA", "France")
CUSTOMERS = ("A", "B", "C", "D")


@dataclass(frozen=True)
class Assignment:
    """Actual decision matrix and separate delivery, penalty and feasibility."""

    word: int
    rows: tuple[tuple[int, ...], ...]
    opened: tuple[int, ...]
    raw_cost: float
    penalty: int
    feasible: bool

    @property
    def cost(self) -> float:
        return self.raw_cost + 10 * self.penalty


def assignment(word: int) -> Assignment:
    if type(word) is not int or not 0 <= word < 4096:
        raise ValueError("decision word must be an integer in0..4095")
    rows = tuple(tuple((word >> (4 * i + j)) & 1 for j in range(4)) for i in range(3))
    opened = tuple(i for i, row in enumerate(rows) if any(row))
    columns = tuple(sum(row[j] for row in rows) for j in range(4))
    raw = sum(WEIGHTS[4 * i + j] * rows[i][j] for i in range(3) for j in range(4))
    penalty = sum((value - 1) ** 2 for value in columns) + (len(opened) - 2) ** 2
    return Assignment(
        word, rows, opened, raw, penalty, columns == (1, 1, 1, 1) and len(opened) == 2
    )


def classical_reference() -> tuple[Assignment, ...]:
    """Solve the complete constrained model and retain ties after quantum sampling."""
    legal = tuple(value for word in range(4096) if (value := assignment(word)).feasible)
    minimum = min(value.raw_cost for value in legal)
    return tuple(value for value in legal if abs(value.raw_cost - minimum) < 1e-12)
