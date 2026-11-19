"""Fresh source-sized supply model with explicit bounded slack semantics."""

from dataclasses import dataclass

COSTS = ((0.5, 1.0, 1.0, 2.1), (1.0, 0.6, 1.4, 1.0), (1.0, 1.4, 0.4, 2.3))


@dataclass(frozen=True)
class Assignment:
    """Twelve decision bits; feasibility never depends on a sampled slack choice."""

    word: int
    rows: tuple[tuple[int, ...], ...]
    raw_cost: float
    feasible: bool


def assignment(word: int) -> Assignment:
    """Decode bit4*source+consumer from the named decision word."""
    if type(word) is not int or not 0 <= word < 4096:
        raise ValueError("decision word must be an integer in 0..4095")
    rows = tuple(
        tuple((word >> (4 * source + consumer)) & 1 for consumer in range(4))
        for source in range(3)
    )
    raw_cost = sum(
        COSTS[source][consumer] * rows[source][consumer]
        for source in range(3)
        for consumer in range(4)
    )
    feasible = all(sum(row) <= 2 for row in rows) and all(
        sum(rows[source][consumer] for source in range(3)) == 1 for consumer in range(4)
    )
    return Assignment(word, rows, raw_cost, feasible)


def residual_penalty(value: Assignment, slack: tuple[int, int, int]) -> int:
    """Slack register bits both have weight one, so valid numeric words0..3 encode0,1,1,2."""
    if len(slack) != 3 or any(
        type(word) is not int or not 0 <= word < 4 for word in slack
    ):
        raise ValueError("three two-bit slack words are required")
    balance = sum(
        (sum(value.rows[source][consumer] for source in range(3)) - 1) ** 2
        for consumer in range(4)
    )
    supply = sum(
        (word.bit_count() + sum(row) - 2) ** 2
        for row, word in zip(value.rows, slack, strict=True)
    )
    return balance + supply


def classical_reference() -> tuple[Assignment, ...]:
    """Solve all original decision constraints exactly; retain every tied optimum."""
    feasible = tuple(row for word in range(4096) if (row := assignment(word)).feasible)
    minimum = min(row.raw_cost for row in feasible)
    return tuple(row for row in feasible if abs(row.raw_cost - minimum) <= 1e-12)
