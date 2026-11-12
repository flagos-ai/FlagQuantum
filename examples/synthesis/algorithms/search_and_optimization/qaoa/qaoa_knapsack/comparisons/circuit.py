"""Independent weight predicate and value table with explicit Circuit mixers."""

import flagquantum as fq
from examples.synthesis.algorithms.search_and_optimization.qaoa.qaoa_knapsack.main import (
    build_program,
)
from examples.synthesis.compare_states import compare_states
from examples.synthesis.cost_reference import cost_phase_circuit

BINDINGS: tuple[tuple[float, ...], ...] = (
    (0.01, 0.02, 0.03, 0.04, 0.05, 0.06),
    (-0.17, 0.23, -0.31, 0.41, -0.53, 0.67),
)


def cost(bits: tuple[int, ...]) -> float:
    # Reference register order is a (three bits) followed by b (two bits).
    a = bits[2] + 2 * bits[3] + 4 * bits[4]
    b = bits[0] + 2 * bits[1]
    return float(-3 * a - 5 * b) if 2 * a + 3 * b <= 12 else 0.0


def reference_circuit(params: tuple[float, ...] = BINDINGS[0]) -> fq.Circuit:
    return cost_phase_circuit(cost, 5, params)


def run_example() -> None:
    for params in BINDINGS:
        compare_states(build_program(params), reference_circuit(params), ("a", "b"))


if __name__ == "__main__":
    run_example()
