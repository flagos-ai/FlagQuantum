"""Independent retained-program Circuit reference."""

import flagquantum as fq
from examples.synthesis.compare_states import compare_states
from examples.synthesis.cost_reference import cost_phase_circuit
from examples.synthesis.tutorials.basics.optimization.learning_optimization.main import (
    build_program,
)

BINDINGS: tuple[tuple[float, ...], ...] = ((0.01, 0.02), (-0.17, 0.23))


def cost(bits: tuple[int, ...]) -> float:
    x0 = bits[6] + 2 * bits[7]
    x1 = bits[4] + 2 * bits[5]
    slack = bits[0] + 2 * bits[1] + 4 * bits[2] + 3 * bits[3]
    residual = slack - 3 * x0 - x1 + 2
    return float(-(3 * x0 + 2 * x1 + 20 * residual**2))


def reference_circuit(params: tuple[float, ...] = BINDINGS[0]) -> fq.Circuit:
    return cost_phase_circuit(cost, 8, params)


def run_example() -> None:
    for params in BINDINGS:
        compare_states(
            build_program(params), reference_circuit(params), ("x0", "x1", "slack")
        )


if __name__ == "__main__":
    run_example()
