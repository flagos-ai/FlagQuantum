"""Build an independent graph-cost Circuit from at-most-four-bit factors."""

from itertools import combinations

import flagquantum as fq
from examples.synthesis.applications.optimization.max_induced_k_color_subgraph.main import (
    DEFAULT_PARAMS,
    build_program,
)
from examples.synthesis.compare_states import compare_states
from examples.synthesis.factor_reference import CostFactor, factored_cost_circuit

EDGES = (
    (0, 1),
    (0, 2),
    (0, 4),
    (1, 2),
    (1, 3),
    (1, 5),
    (2, 3),
    (2, 4),
    (2, 5),
    (3, 4),
    (4, 5),
)
BINDINGS = (
    DEFAULT_PARAMS,
    tuple(-0.002 * (i + 1) if i % 3 else 0.005 * (i + 1) for i in range(16)),
)


def graph_cost(bits: tuple[int, ...]) -> float:
    """Evaluate the source's directed-edge square and equal-color penalties."""
    if len(bits) != 12:
        raise ValueError("Two color rows require twelve logical bits")
    equal = sum(int(bits[j] == bits[6 + j]) for j in range(6))
    directed = 2 * sum(
        bits[offset + i] * bits[offset + j] for offset in (0, 6) for i, j in EDGES
    )
    return float(6 - equal - directed**2)


def cost_factors() -> list[CostFactor]:
    """Expand one graph square into small truth tables without compiler analysis."""
    factors: list[CostFactor] = [((), (6.0,))]
    for j in range(6):
        factors.append(((j, j + 6), (-1.0, 0.0, 0.0, -1.0)))
    edges = [(offset + i, offset + j) for offset in (0, 6) for i, j in EDGES]
    # Each undirected edge is counted in both source directions.
    for edge in edges:
        factors.append((edge, (0.0, 0.0, 0.0, -4.0)))
    for first, second in combinations(edges, 2):
        support = tuple(sorted(set(first + second)))
        factors.append(
            (
                support,
                tuple(
                    -8.0 if value == (1 << len(support)) - 1 else 0.0
                    for value in range(1 << len(support))
                ),
            )
        )
    return factors


def reference_circuit(params: tuple[float, ...] = DEFAULT_PARAMS) -> fq.Circuit:
    interleaved = tuple(value for i in range(8) for value in (params[i], params[8 + i]))
    return factored_cost_circuit(12, cost_factors(), interleaved)


def run_example() -> None:
    for params in BINDINGS:
        compare_states(build_program(params), reference_circuit(params), ("a", "b"))


if __name__ == "__main__":
    run_example()
