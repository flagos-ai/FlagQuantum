"""Independent edge-cut costs and parity-phase/RX Circuit."""

import flagquantum as fq
from examples.synthesis.algorithms.search_and_optimization.qaoa.qaoa_max_cut.main import (
    DEFAULT_PARAMS,
    build_program,
)
from examples.synthesis.compare_states import compare_states
from examples.synthesis.cost_reference import cost_phase_circuit

EDGES = ((0, 1), (0, 2), (1, 2), (1, 3), (2, 4), (3, 4))
BINDINGS = (DEFAULT_PARAMS, (-0.17, 0.23, -0.31, 0.41, -0.53, 0.67, -0.71, 0.83))


def cost(bits: tuple[int, ...]) -> float:
    return sum(bits[a] != bits[b] for a, b in EDGES) / 6


def reference_circuit(params: tuple[float, ...] = DEFAULT_PARAMS) -> fq.Circuit:
    return cost_phase_circuit(cost, 5, params)


def run_example() -> None:
    for params in BINDINGS:
        compare_states(build_program(params), reference_circuit(params), ("v",))


if __name__ == "__main__":
    run_example()
