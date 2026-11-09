"""Independent classical assignment costs, truth factors and Circuit mixers."""

import flagquantum as fq
from examples.synthesis.applications.logistics.traveling_salesman_problem.main import (
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

DISTANCES = ((5, 1, 2, 5), (5, 5, 1, 2), (5, 1, 5, 1), (5, 5, 5, 5))


def cost(bits: tuple[int, ...]) -> float:
    distance = sum(
        DISTANCES[city][other] * bits[4 * city + slot] * bits[4 * other + slot + 1]
        for city in range(4)
        for other in range(4)
        for slot in range(3)
    )
    row_errors = sum(
        (sum(bits[4 * city + slot] for slot in range(4)) - 1) ** 2 for city in range(4)
    )
    column_errors = sum(
        (sum(bits[4 * city + slot] for city in range(4)) - 1) ** 2 for slot in range(4)
    )
    return float(-distance - row_errors - column_errors)


def cost_factors() -> list[CostFactor]:
    factors: list[CostFactor] = []
    for city in range(4):
        for other in range(4):
            for slot in range(3):
                factors.append(
                    (
                        (4 * city + slot, 4 * other + slot + 1),
                        (0.0, 0.0, 0.0, -float(DISTANCES[city][other])),
                    )
                )
        factors.append(
            squared_residual(
                tuple(4 * city + slot for slot in range(4)), (1.0,) * 4, -1.0, -1.0
            )
        )
    for slot in range(4):
        factors.append(
            squared_residual(
                tuple(4 * city + slot for city in range(4)), (1.0,) * 4, -1.0, -1.0
            )
        )
    return factors


BINDINGS = (DEFAULT_PARAMS, tuple((-1) ** i * (0.004 + 0.001 * i) for i in range(16)))


def reference_circuit(params: tuple[float, ...] = DEFAULT_PARAMS) -> fq.Circuit:
    interleaved = tuple(value for i in range(8) for value in (params[i], params[8 + i]))
    return factored_cost_circuit(16, cost_factors(), interleaved)


def run_example() -> None:
    for params in BINDINGS:
        compare_states(build_program(params), reference_circuit(params), ("visits",))


SOURCE_FIELDS = ("v.x[0]", "v.x[1]", "v.x[2]", "v.x[3]")


def source_reference_circuit(params: tuple[float, ...] = DEFAULT_PARAMS) -> fq.Circuit:
    interleaved = tuple(value for i in range(8) for value in (params[i], params[8 + i]))
    return factored_cost_circuit(
        16, order_register_factors(cost_factors(), (4, 4, 4, 4)), interleaved
    )


if __name__ == "__main__":
    run_example()
