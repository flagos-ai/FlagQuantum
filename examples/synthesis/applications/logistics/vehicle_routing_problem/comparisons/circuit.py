"""Independent routing costs as small classical factors and Circuit parities."""

import flagquantum as fq
from examples.synthesis.applications.logistics.vehicle_routing_problem.main import (
    ARRIVAL,
    DEPARTURE,
    DISTANCES,
    build_program,
)
from examples.synthesis.compare_states import compare_states
from examples.synthesis.factor_reference import (
    CostFactor,
    factored_cost_circuit,
    squared_residual,
)

BINDINGS: tuple[tuple[float, ...], ...] = (
    tuple(0.01 * (i + 1) for i in range(24)),
    tuple((-1) ** i * (0.03 + 0.011 * i) for i in range(24)),
)


def cost_factors() -> tuple[CostFactor, ...]:
    factors: list[CostFactor] = []
    for city in range(4):
        for slot in range(4):
            coefficient = DEPARTURE[city] if slot % 2 == 0 else ARRIVAL[city]
            factors.append(((4 * city + slot,), (0.0, 0.15 * coefficient)))
        for other in range(4):
            if city != other:
                for lane in range(2):
                    factors.append(
                        (
                            (4 * city + lane, 4 * other + 2 + lane),
                            (0.0, 0.0, 0.0, 0.15 * DISTANCES[4 * city + other]),
                        )
                    )
        factors.append(
            squared_residual(
                tuple(4 * city + i for i in range(4)), (1.0,) * 4, -1.0, 0.9
            )
        )
    for slot in range(4):
        factors.append(
            squared_residual(
                tuple(4 * city + slot for city in range(4)), (1.0,) * 4, -1.0, 0.9
            )
        )
    return tuple(factors)


def reference_circuit(params: tuple[float, ...] = BINDINGS[0]) -> fq.Circuit:
    return factored_cost_circuit(16, cost_factors(), params, mixer_scale=0.1)


def run_example() -> None:
    for params in BINDINGS:
        compare_states(build_program(params), reference_circuit(params), ("x",))


if __name__ == "__main__":
    run_example()
