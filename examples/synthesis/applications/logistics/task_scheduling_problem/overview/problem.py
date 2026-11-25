"""Generate both original task graphs and their exact penalty polynomials."""

import itertools
from dataclasses import dataclass

import torch

Polynomial = dict[tuple[int, ...], float]


@dataclass(frozen=True)
class Problem:
    name: str
    tasks: int
    edges: tuple[tuple[int, int], ...]
    capacity: tuple[int, ...]
    workload: tuple[int, ...]
    active: tuple[tuple[int, int], ...]
    slack: tuple[tuple[int, ...], ...]
    polynomial: tuple[tuple[tuple[int, ...], float], ...]
    width: int

    def slack_weights(self, time: int) -> tuple[int, ...]:
        size = len(self.slack[time])
        return tuple(1 << i for i in range(size - 1)) + (
            self.capacity[time] - (1 << (size - 1)) + 1,
        )


def _distance(edges: tuple[tuple[int, int], ...], tasks: int) -> list[list[int]]:
    result = [[0 if i == j else 999 for j in range(tasks)] for i in range(tasks)]
    for first, second in edges:
        result[first][second] = 1
    for via in range(tasks):
        for first in range(tasks):
            for second in range(tasks):
                result[first][second] = min(
                    result[first][second], result[first][via] + result[via][second]
                )
    return result


def create(large: bool = False) -> Problem:
    edges: tuple[tuple[int, int], ...]
    capacity: tuple[int, ...]
    workload: tuple[int, ...]
    if large:
        name = "large"
        edges = ((0, 1), (0, 3), (0, 2), (2, 4), (3, 4), (1, 5), (4, 5))
        capacity = (1, 3, 4, 3, 1)
        workload = (1, 3, 2, 2, 1, 1)
    else:
        name = "small"
        edges = ((0, 1), (1, 3), (2, 3))
        capacity = (3, 3, 3)
        workload = (1, 1, 1, 1)
    tasks = len(workload)
    distances = _distance(edges, tasks)
    roots = tuple(i for i in range(tasks) if not any(j == i for _, j in edges))
    ends = tuple(i for i in range(tasks) if not any(j == i for j, _ in edges))
    active = tuple(
        (time, task)
        for time in range(len(capacity))
        for task in range(tasks)
        if min(distances[root][task] for root in roots)
        <= time
        < len(capacity) - min(distances[task][end] for end in ends)
    )
    index = {key: i for i, key in enumerate(active)}
    offset = len(active)
    slack = []
    for value in capacity:
        width = value.bit_length()
        slack.append(tuple(range(offset, offset + width)))
        offset += width
    coefficients: Polynomial = {}

    def add(support: tuple[int, ...], coefficient: float) -> None:
        key = tuple(sorted(set(support)))
        coefficients[key] = coefficients.get(key, 0) + coefficient

    def square(terms: tuple[tuple[int, float], ...], constant: float) -> None:
        add((), 20 * constant * constant)
        for bit, coefficient in terms:
            add((bit,), 20 * (coefficient * coefficient + 2 * constant * coefficient))
        for i, (first, a) in enumerate(terms):
            for second, b in terms[i + 1 :]:
                add((first, second), 40 * a * b)

    for task in range(tasks):
        terms = tuple((bit, 1.0) for bit, (_, t) in enumerate(active) if task == t)
        square(terms, -1)
    for time, limit in enumerate(capacity):
        terms = tuple(
            (bit, float(workload[task]))
            for bit, (t, task) in enumerate(active)
            if t == time
        )
        size = len(slack[time])
        slack_weights = tuple(1 << i for i in range(size - 1)) + (
            limit - (1 << (size - 1)) + 1,
        )
        terms += tuple(
            (bit, float(weight))
            for bit, weight in zip(slack[time], slack_weights, strict=True)
        )
        square(terms, -limit)
    for first, second in edges:
        for t1 in range(len(capacity)):
            for t2 in range(t1 + 1):
                if (t1, first) in index and (t2, second) in index:
                    add((index[t1, first], index[t2, second]), 20.0)
    for time, task in active:
        if task == ends[0]:
            add((index[time, task],), float(time + 1))
    return Problem(
        name,
        tasks,
        edges,
        capacity,
        workload,
        active,
        tuple(slack),
        tuple(
            (support, value) for support, value in sorted(coefficients.items()) if value
        ),
        offset,
    )


def decode(problem: Problem, word: int) -> tuple[tuple[int, ...], ...]:
    return tuple(
        tuple(
            int(
                (time, task) in problem.active
                and (word >> problem.active.index((time, task))) & 1
            )
            for task in range(problem.tasks)
        )
        for time in range(len(problem.capacity))
    )


def original_cost(problem: Problem, word: int) -> float:
    matrix = decode(problem, word)
    objective = sum((t + 1) * row[-1] for t, row in enumerate(matrix))
    penalty = sum(
        (sum(row[task] for row in matrix) - 1) ** 2 for task in range(problem.tasks)
    )
    for time, row in enumerate(matrix):
        used = sum(x * w for x, w in zip(row, problem.workload, strict=True))
        slack = sum(
            ((word >> bit) & 1) * weight
            for bit, weight in zip(
                problem.slack[time], problem.slack_weights(time), strict=True
            )
        )
        penalty += (used + slack - problem.capacity[time]) ** 2
    penalty += sum(
        matrix[t1][first] * matrix[t2][second]
        for first, second in problem.edges
        for t1 in range(len(matrix))
        for t2 in range(t1 + 1)
    )
    return float(objective + 20 * penalty)


def valid(problem: Problem, matrix: tuple[tuple[int, ...], ...]) -> bool:
    if not all(sum(row[t] for row in matrix) == 1 for t in range(problem.tasks)):
        return False
    if any(
        sum(a * b for a, b in zip(row, problem.workload, strict=True)) > cap
        for row, cap in zip(matrix, problem.capacity, strict=True)
    ):
        return False
    times = tuple(
        next(i for i, row in enumerate(matrix) if row[task])
        for task in range(problem.tasks)
    )
    return all(times[first] < times[second] for first, second in problem.edges)


def classical(problem: Problem) -> tuple[int, tuple[tuple[int, ...], ...]]:
    feasible = []
    for times in itertools.product(range(len(problem.capacity)), repeat=problem.tasks):
        if any(times[first] >= times[second] for first, second in problem.edges):
            continue
        matrix = tuple(
            tuple(int(times[task] == t) for task in range(problem.tasks))
            for t in range(len(problem.capacity))
        )
        if valid(problem, matrix):
            feasible.append((times[-1] + 1, matrix))
    return min(feasible)


def costs(problem: Problem, words: torch.Tensor) -> torch.Tensor:
    result = torch.zeros(words.numel(), dtype=torch.float64)
    for support, coefficient in problem.polynomial:
        term = torch.full_like(result, coefficient)
        for bit in support:
            term *= ((words >> bit) & 1).to(torch.float64)
        result += term
    return result
