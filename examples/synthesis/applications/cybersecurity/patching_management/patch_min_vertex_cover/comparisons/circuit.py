"""Independent retained-program Circuit reference."""

import flagquantum as fq
from examples.synthesis.applications.cybersecurity.patching_management.patch_min_vertex_cover.main import (
    build_program,
)
from examples.synthesis.compare_states import compare_states
from examples.synthesis.cost_reference import cost_phase_circuit

BINDINGS: tuple[tuple[float, ...], ...] = (
    (0.01, 0.02, 0.03, 0.04, 0.05, 0.06),
    (-0.17, 0.23, -0.31, 0.41, -0.53, 0.67),
)


def cost(bits: tuple[int, ...]) -> float:
    edges = (
        (0, 1),
        (0, 2),
        (0, 3),
        (0, 5),
        (0, 6),
        (0, 7),
        (1, 2),
        (1, 3),
        (1, 6),
        (2, 7),
        (3, 5),
        (3, 6),
        (3, 7),
        (4, 5),
        (4, 7),
        (5, 6),
        (5, 7),
        (6, 7),
    )
    uncovered = sum(1 for a, b in edges if bits[a] == bits[b] == 0)
    return float(-sum(bits) - 10 * uncovered)


def reference_circuit(params: tuple[float, ...] = BINDINGS[0]) -> fq.Circuit:
    return cost_phase_circuit(cost, 8, params)


def run_example() -> None:
    for params in BINDINGS:
        compare_states(build_program(params), reference_circuit(params), ("v",))


if __name__ == "__main__":
    run_example()
