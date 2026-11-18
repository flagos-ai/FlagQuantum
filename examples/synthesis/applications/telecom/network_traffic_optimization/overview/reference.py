"""Independent local truth tables and public Circuit parity phases."""

import flagquantum as fq
from examples.synthesis.factor_reference import (
    CostFactor,
    factored_cost_circuit,
    squared_residual,
)

from . import problem


def factors() -> tuple[CostFactor, ...]:
    result: list[CostFactor] = []
    for demand, (source, sink) in enumerate(problem.DEMANDS):
        for edge, latency in enumerate(problem.LATENCY):
            result.append(((8 * demand + edge,), (0.0, latency / 15)))
        for node in problem.NODES:
            edges = tuple(
                i
                for i, (first, second) in enumerate(problem.EDGES)
                if first == node or second == node
            )
            weights = tuple(1.0 if problem.EDGES[i][1] == node else -1.0 for i in edges)
            result.append(
                squared_residual(
                    tuple(8 * demand + i for i in edges),
                    weights,
                    float(node == source) - float(node == sink),
                    0.2,
                )
            )
    for edge in range(8):
        result.append(
            (
                (edge, edge + 8, edge + 16),
                tuple(0.2 * int(word.bit_count() >= 2) for word in range(8)),
            )
        )
    return tuple(result)


def circuit(parameters: tuple[float, ...]) -> fq.Circuit:
    return factored_cost_circuit(24, factors(), parameters, mixer_scale=1.0)


def scalar_cost(word: int) -> float:
    return float(
        sum(
            costs[
                sum(
                    ((word >> bit) & 1) << position
                    for position, bit in enumerate(support)
                )
            ]
            for support, costs in factors()
        )
    )
