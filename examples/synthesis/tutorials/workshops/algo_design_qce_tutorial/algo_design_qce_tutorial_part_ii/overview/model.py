"""Independent classical costs; phase and optimizer penalties stay distinct."""

import math
from typing import Literal

Model = Literal["maxcut", "penalty", "feasible"]
Parameters = tuple[float, float, float, float, float, float]
INITIAL: Parameters = (0, math.pi / 2, math.pi, math.pi, math.pi / 2, 0)
BINDINGS: tuple[Parameters, ...] = (
    (0.17, -0.31, 0.23, 0.41, -0.19, 0.29),
    (-0.13, 0.37, 0.29, -0.43, 0.31, 0.11),
)


def value(a: int, b: int) -> int:
    return 3 * a + 5 * b


def weight(a: int, b: int) -> int:
    return 2 * a + 3 * b


def feasible_value(a: int, b: int) -> int:
    return value(a, b) if weight(a, b) <= 12 else 0


def phase_cost(model: Model, values: tuple[int, ...]) -> int:
    if model == "maxcut":
        word = values[0]
        a, b, c = word & 1, (word >> 1) & 1, (word >> 2) & 1
        return -((a ^ b) + (a ^ c))
    a, b = values[:2]
    if model == "penalty":
        return -value(a, b) + 2 * (weight(a, b) + values[2] - 12) ** 2
    return -feasible_value(a, b)


def optimization_cost(model: Model, values: tuple[int, ...]) -> int:
    if model == "maxcut":
        return phase_cost(model, values)
    a, b = values[:2]
    return -value(a, b) + 4 * max(weight(a, b) - 12, 0)
