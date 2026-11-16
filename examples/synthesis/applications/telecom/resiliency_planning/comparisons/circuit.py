"""Independent network constraints, small truth tables and Circuit rotations."""

import flagquantum as fq
from examples.synthesis.applications.telecom.resiliency_planning.main import (
    build_program,
)
from examples.synthesis.compare_states import compare_states
from examples.synthesis.factor_reference import (
    CostFactor,
    factored_cost_circuit,
    squared_residual,
)

BINDINGS: tuple[tuple[float, ...], ...] = (
    tuple(0.01 * (i + 1) for i in range(40)),
    tuple((-1) ** i * (0.03 + 0.011 * i) for i in range(40)),
)


def cost_factors() -> tuple[CostFactor, ...]:
    factors: list[CostFactor] = []
    for group in (0, 12):
        for bit in range(6):
            factors.append(((group + bit,), (0.0, 0.025)))
        factors.append(((group + 6,), (0.0, 0.05)))
    for a, b in ((7, 19), (8, 20), (9, 21), (11, 23)):
        factors.append(((a, b), (0.0, 0.0, 0.0, 0.25)))
    for a, b in ((1, 14), (2, 13)):
        factors.append(((a, b), (0.0, 0.0, 0.0, 1.0)))
    residuals = (
        ((7, 0, 1), (-2.0, 1.0, 1.0), 1.0),
        ((19, 12, 13), (-2.0, 1.0, 1.0), 0.0),
        ((8, 3, 5), (-2.0, 1.0, 1.0), 0.0),
        ((20, 15, 17), (-2.0, 1.0, 1.0), 1.0),
        ((9, 0, 2, 3, 4), (-2.0, 1.0, 1.0, 1.0, 1.0), 0.0),
        ((21, 12, 14, 15, 16), (-2.0, 1.0, 1.0, 1.0, 1.0), 0.0),
        ((10, 1, 2, 6), (-2.0, 1.0, 1.0, 1.0), 1.0),
        ((22, 13, 14, 18), (-2.0, 1.0, 1.0, 1.0), 1.0),
        ((11, 4, 5, 6), (-2.0, 1.0, 1.0, 1.0), 0.0),
        ((23, 16, 17, 18), (-2.0, 1.0, 1.0, 1.0), 0.0),
    )
    for bits, weights, offset in residuals:
        factors.append(squared_residual(bits, weights, offset, 0.125))
    for bit in (10, 7, 22, 20):
        factors.append(((bit,), (0.125, 0.0)))
    return tuple(factors)


def reference_circuit(params: tuple[float, ...] = BINDINGS[0]) -> fq.Circuit:
    return factored_cost_circuit(24, cost_factors(), params)


def run_example() -> None:
    for params in BINDINGS:
        compare_states(build_program(params), reference_circuit(params), ("z",))


if __name__ == "__main__":
    run_example()
