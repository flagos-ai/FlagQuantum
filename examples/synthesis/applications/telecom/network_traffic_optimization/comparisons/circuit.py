"""Independent flow residuals and congestion threshold factors."""

import flagquantum as fq
from examples.synthesis.applications.telecom.network_traffic_optimization.main import (
    build_program,
)
from examples.synthesis.compare_states import compare_states
from examples.synthesis.factor_reference import (
    CostFactor,
    factored_cost_circuit,
    squared_residual,
)

BINDINGS: tuple[tuple[float, ...], ...] = (
    tuple(0.01 * (i + 1) for i in range(10)),
    tuple((-1) ** i * (0.03 + 0.011 * i) for i in range(10)),
)


def cost_factors() -> tuple[CostFactor, ...]:
    weights = (
        0.133333333333333,
        0.2,
        0.133333333333333,
        0.066666666666667,
        0.066666666666667,
        0.133333333333333,
        0.133333333333333,
        0.066666666666667,
    )
    factors: list[CostFactor] = []
    supplies = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
    sinks = (0, 1, 1)
    for group in range(3):
        start = 8 * group
        for edge, weight in enumerate(weights):
            factors.append(((start + edge,), (0.0, weight)))
        residuals = (
            ((0, 1), (-1.0, -1.0), float(supplies[group][0])),
            ((0, 2, 3), (1.0, -1.0, -1.0), float(supplies[group][1])),
            ((1, 4, 5), (1.0, -1.0, -1.0), float(supplies[group][2])),
            ((3, 5, 6), (1.0, 1.0, -1.0), 0.0),
            ((2, 4, 7), (1.0, 1.0, -1.0), float(sinks[group] - 1)),
            ((6, 7), (1.0, 1.0), float(-sinks[group])),
        )
        for bits, coefficients, offset in residuals:
            factors.append(
                squared_residual(
                    tuple(start + i for i in bits), coefficients, offset, 0.2
                )
            )
    # The original cubic polynomial equals a capacity-two threshold, including 111.
    congestion = tuple(0.2 * int(basis.bit_count() >= 2) for basis in range(8))
    for edge in range(8):
        factors.append(((edge, edge + 8, edge + 16), congestion))
    return tuple(factors)


def reference_circuit(params: tuple[float, ...] = BINDINGS[0]) -> fq.Circuit:
    return factored_cost_circuit(24, cost_factors(), params)


def run_example() -> None:
    for params in BINDINGS:
        compare_states(build_program(params), reference_circuit(params), ("z",))


if __name__ == "__main__":
    run_example()
