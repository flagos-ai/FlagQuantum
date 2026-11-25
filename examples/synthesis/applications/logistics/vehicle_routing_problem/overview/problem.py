"""Fresh six-location, two-vehicle routing instance and source costs."""

import itertools
import math
from dataclasses import dataclass

import numpy as np
import torch

WIDTH = 16
LAYERS = 12
SHOTS = 30_000
OBJECTIVE_QUERIES = 26


@dataclass(frozen=True)
class Problem:
    positions: tuple[tuple[float, float], ...]
    distances: tuple[tuple[float, ...], ...]
    maximum: float
    minimum: float
    scale: float


def fresh() -> Problem:
    coordinates = np.random.RandomState(10).rand(6, 2) * 10
    positions = tuple((float(x), float(y)) for x, y in coordinates)
    raw = tuple(
        tuple(math.dist(first, second) for second in positions) for first in positions
    )
    maximum, minimum = max(map(max, raw)), min(
        value for row in raw for value in row if value
    )
    scale = 6 * (maximum - minimum)
    return Problem(
        positions,
        tuple(tuple(value / scale for value in row) for row in raw),
        maximum,
        minimum,
        scale,
    )


def bits(word: int) -> tuple[int, ...]:
    return tuple((word >> index) & 1 for index in range(WIDTH))


def indicator(x: tuple[int, ...], city: int, position: int, vehicle: int) -> int:
    if city < 2 or position in (0, 3):
        return int(city == vehicle and position in (0, 3))
    return x[(city - 2) * 4 + (position - 1) * 2 + vehicle]


def valid(word: int) -> bool:
    x = bits(word)
    return all(sum(x[4 * city : 4 * city + 4]) == 1 for city in range(4)) and all(
        sum(x[4 * city + slot] for city in range(4)) == 1 for slot in range(4)
    )


def scalar_cost(problem: Problem, word: int) -> float:
    x = bits(word)
    travel = sum(
        problem.distances[first][second]
        * indicator(x, first, position, vehicle)
        * indicator(x, second, position + 1, vehicle)
        for position in range(3)
        for first in range(6)
        for second in range(6)
        for vehicle in range(2)
    )
    city_penalty = sum((sum(x[4 * city : 4 * city + 4]) - 1) ** 2 for city in range(4))
    slot_penalty = sum(
        (sum(x[4 * city + slot] for city in range(4)) - 1) ** 2 for slot in range(4)
    )
    return 0.15 * travel + 0.9 * (city_penalty + slot_penalty)


def costs(problem: Problem) -> torch.Tensor:
    """Evaluate all 65,536 original assignments without a dense operator."""
    words = torch.arange(1 << WIDTH, dtype=torch.int64)
    x = ((words[:, None] >> torch.arange(WIDTH)) & 1).to(torch.float64)
    penalty = torch.zeros(words.numel(), dtype=torch.float64)
    for city in range(4):
        penalty += (x[:, 4 * city : 4 * city + 4].sum(dim=1) - 1).square()
    for slot in range(4):
        penalty += (x[:, slot::4].sum(dim=1) - 1).square()
    travel = torch.zeros_like(penalty)
    for city in range(4):
        for vehicle in range(2):
            first, second = 4 * city + vehicle, 4 * city + 2 + vehicle
            travel += problem.distances[vehicle][city + 2] * x[:, first]
            travel += problem.distances[city + 2][vehicle] * x[:, second]
            for other in range(4):
                travel += (
                    problem.distances[city + 2][other + 2]
                    * x[:, first]
                    * x[:, 4 * other + 2 + vehicle]
                )
    return 0.15 * travel + 0.9 * penalty


def route(word: int, vehicle: int) -> tuple[int, ...]:
    x = bits(word)
    return tuple(
        city
        for position in range(4)
        for city in range(6)
        if indicator(x, city, position, vehicle)
    )


def classical_reference(problem: Problem) -> tuple[float, tuple[int, ...]]:
    """Enumerate the source's 24 feasible assignments independently."""
    feasible = []
    for ordering in itertools.permutations(range(4)):
        word = sum(1 << (4 * city + slot) for slot, city in enumerate(ordering))
        feasible.append((scalar_cost(problem, word), word))
    best = min(value for value, _ in feasible)
    return best, tuple(word for value, word in feasible if abs(value - best) < 1e-14)
