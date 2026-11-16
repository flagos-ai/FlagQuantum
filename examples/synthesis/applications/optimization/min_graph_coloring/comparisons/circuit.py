"""Independent graph constraints and bounded truth-factor Circuit reference."""

import flagquantum as fq
from examples.synthesis.applications.optimization.min_graph_coloring.main import (
    DEFAULT_PARAMS,
    build_program,
)
from examples.synthesis.compare_states import compare_states
from examples.synthesis.factor_reference import (
    CostFactor,
    factored_cost_circuit,
    squared_residual,
)
from examples.synthesis.quadratic_reference import order_register_factors

EDGES = ((0, 4), (0, 1), (0, 3), (1, 3), (2, 3))
BINDINGS = (DEFAULT_PARAMS, tuple((-1) ** i * (0.004 + 0.001 * i) for i in range(12)))
SOURCE_FIELDS = ("v.x[0]", "v.x[1]", "v.x[2]")


def cost(bits: tuple[int, ...]) -> int:
    empty = sum(not any(bits[5 * c : 5 * c + 5]) for c in range(3))
    missing = sum((sum(bits[5 * c + v] for c in range(3)) - 1) ** 2 for v in range(5))
    conflicts = sum(
        bits[5 * c + a] * bits[5 * c + b] for c in range(3) for a, b in EDGES
    )
    return empty - 10 * missing - 40 * conflicts**2 - 3


def cost_factors() -> list[CostFactor]:
    factors: list[CostFactor] = [((), (-3.0,))]
    for color in range(3):
        factors.append((tuple(5 * color + v for v in range(5)), (1.0,) + (0.0,) * 31))
    for vertex in range(5):
        factors.append(
            squared_residual(
                tuple(5 * c + vertex for c in range(3)), (1.0,) * 3, -1.0, -10.0
            )
        )
    edges = [tuple(5 * c + v for v in edge) for c in range(3) for edge in EDGES]
    # Square the edge sum independently using e**2=e and pair cross terms.
    for i, first in enumerate(edges):
        factors.append((first, (0.0, 0.0, 0.0, -40.0)))
        for second in edges[i + 1 :]:
            support = tuple(sorted(set(first + second)))
            factors.append((support, (0.0,) * ((1 << len(support)) - 1) + (-80.0,)))
    return factors


def reference_circuit(params: tuple[float, ...] = DEFAULT_PARAMS) -> fq.Circuit:
    alternating = tuple(value for i in range(6) for value in (params[i], params[6 + i]))
    return factored_cost_circuit(15, cost_factors(), alternating)


def source_reference_circuit(params: tuple[float, ...] = DEFAULT_PARAMS) -> fq.Circuit:
    alternating = tuple(value for i in range(6) for value in (params[i], params[6 + i]))
    return factored_cost_circuit(
        15, order_register_factors(cost_factors(), (5, 5, 5)), alternating
    )


def run_example() -> None:
    for params in BINDINGS:
        compare_states(build_program(params), reference_circuit(params), ("colors",))


if __name__ == "__main__":
    run_example()
