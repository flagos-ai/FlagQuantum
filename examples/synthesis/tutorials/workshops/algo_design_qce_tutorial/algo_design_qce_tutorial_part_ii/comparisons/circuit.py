"""Independent Knapsack predicate and explicit RX Circuit for the tutorial."""

import flagquantum as fq
from examples.synthesis.algorithms.search_and_optimization.qaoa.comparisons.circuit import (
    reference_circuit as knapsack_circuit,
)
from examples.synthesis.compare_states import compare_states
from examples.synthesis.tutorials.workshops.algo_design_qce_tutorial.algo_design_qce_tutorial_part_ii.main import (
    DEFAULT_PARAMS,
    build_program,
)

BINDINGS = (DEFAULT_PARAMS, (-0.17, 0.23, -0.31, 0.41, -0.53, 0.67))
SOURCE_FIELDS = ("v.a", "v.b")


def reference_circuit(params: tuple[float, ...] = DEFAULT_PARAMS) -> fq.Circuit:
    return knapsack_circuit(params)


def run_example() -> None:
    for params in BINDINGS:
        compare_states(build_program(params), reference_circuit(params), ("a", "b"))


if __name__ == "__main__":
    run_example()
