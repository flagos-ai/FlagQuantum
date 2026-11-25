"""Independent supply, balance and slack factors on the 18 logical bits."""

import flagquantum as fq
from examples.synthesis.applications.optimization.electric_grid_optimization.main import (
    build_program,
)
from examples.synthesis.compare_states import compare_states
from examples.synthesis.factor_reference import (
    CostFactor,
    factored_cost_circuit,
    squared_residual,
)

BINDINGS: tuple[tuple[float, ...], ...] = (
    (0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08),
    (-0.17, 0.23, -0.31, 0.41, -0.53, 0.67, -0.79, 0.83),
)
ROWS = ((14, 15, 16, 17), (10, 11, 12, 13), (6, 7, 8, 9))
SLACK = ((4, 5), (2, 3), (0, 1))


def cost_factors() -> tuple[CostFactor, ...]:
    factors: list[CostFactor] = []
    costs = ((0.5, 1.0, 1.0, 2.1), (1.0, 0.6, 1.4, 1.0), (1.0, 1.4, 0.4, 2.3))
    for row, values in zip(ROWS, costs, strict=True):
        for bit, value in zip(row, values, strict=True):
            factors.append(((bit,), (0.0, -value)))
    for slot in range(4):
        factors.append(
            squared_residual(
                tuple(row[slot] for row in ROWS), (1.0, 1.0, 1.0), -1.0, -10.0
            )
        )
    for row, slack in zip(ROWS, SLACK, strict=True):
        factors.append(squared_residual((*slack, *row), (1.0,) * 6, -2.0, -10.0))
    return tuple(factors)


def reference_circuit(params: tuple[float, ...] = BINDINGS[0]) -> fq.Circuit:
    interleaved = tuple(
        value for layer in range(4) for value in (params[layer], params[4 + layer])
    )
    return factored_cost_circuit(18, cost_factors(), interleaved)


def run_example() -> None:
    for params in BINDINGS:
        compare_states(
            build_program(params),
            reference_circuit(params),
            ("a", "b", "c", "sa", "sb", "sc"),
        )


if __name__ == "__main__":
    run_example()
