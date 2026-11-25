"""Independent bounded factor tables and complete public Circuit reference."""

import flagquantum as fq
from examples.synthesis.factor_reference import (
    CostFactor,
    factored_cost_circuit,
    squared_residual,
)

from . import problem


def factors() -> tuple[CostFactor, ...]:
    result: list[CostFactor] = []
    for demand, (source, target) in enumerate(problem.DEMANDS):
        for edge, latency in enumerate(problem.LATENCY):
            result.append(((12 * demand + edge,), (0.0, 0.025 * latency)))
        for index, node in enumerate(problem.NODES):
            incident = tuple(
                edge
                for edge, endpoints in enumerate(problem.EDGES)
                if node in endpoints
            )
            support = (12 * demand + 7 + index,) + tuple(
                12 * demand + edge for edge in incident
            )
            result.append(
                squared_residual(
                    support,
                    (-2.0,) + (1.0,) * len(incident),
                    float(node in (source, target)),
                    0.125,
                )
            )
        for node in (source, target):
            result.append(
                ((12 * demand + 7 + problem.NODES.index(node),), (0.125, 0.0))
            )
    for index, node in enumerate(problem.NODES):
        if node != "S2":
            result.append(((7 + index, 19 + index), (0.0, 0.0, 0.0, 0.25)))
    result.extend((((1, 14), (0.0, 0.0, 0.0, 1.0)), ((2, 13), (0.0, 0.0, 0.0, 1.0))))
    return tuple(result)


def circuit(parameters: tuple[float, ...]) -> fq.Circuit:
    return factored_cost_circuit(24, factors(), parameters)


def scalar_cost(word: int) -> float:
    return sum(
        table[sum(((word >> bit) & 1) << index for index, bit in enumerate(support))]
        for support, table in factors()
    )
