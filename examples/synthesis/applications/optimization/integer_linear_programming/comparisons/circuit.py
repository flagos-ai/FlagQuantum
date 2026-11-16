"""Independent integer cost table and explicit Circuit parity/RX layers."""

import flagquantum as fq
from examples.synthesis.applications.optimization.integer_linear_programming.main import (
    build_program,
)
from examples.synthesis.compare_states import compare_states
from examples.synthesis.cost_reference import cost_phase_circuit

BINDINGS: tuple[tuple[float, ...], ...] = (
    (0.01, 0.02, 0.03, 0.04, 0.05, 0.06),
    (-0.17, 0.23, -0.31, 0.41, -0.53, 0.67),
)


def cost(bits: tuple[int, ...]) -> float:
    a = bits[6] + 2 * bits[7]
    b = bits[4] + 2 * bits[5]
    c = bits[2] + 2 * bits[3]
    slack0, slack1 = bits[1], bits[0]
    reward = 10 * a * b + 10 * a * c - a + 10 * b * c - 2 * b - 3 * c
    residual0 = slack0 + a + b + c - 1
    residual1 = slack1 + a + b + c - 1
    return float(-(reward + 10 * residual0**2 + 10 * residual1**2))


def reference_circuit(params: tuple[float, ...] = BINDINGS[0]) -> fq.Circuit:
    return cost_phase_circuit(cost, 8, params)


def run_example() -> None:
    for params in BINDINGS:
        compare_states(
            build_program(params),
            reference_circuit(params),
            ("x0", "x1", "x2", "s0", "s1"),
        )


if __name__ == "__main__":
    run_example()
