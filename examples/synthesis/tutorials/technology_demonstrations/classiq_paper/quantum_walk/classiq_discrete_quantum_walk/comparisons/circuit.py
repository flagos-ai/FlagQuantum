"""Compare the original-width native walk with an independent permutation."""

import flagquantum as fq
from examples.synthesis.compare_states import compare_states
from examples.synthesis.tutorials.advanced_tutorials.discrete_quantum_walk.comparisons.circuit import (
    reference_circuit as circle_reference,
)
from examples.synthesis.tutorials.technology_demonstrations.classiq_paper.quantum_walk.classiq_discrete_quantum_walk.main import (
    build_program,
)


def reference_circuit() -> fq.Circuit:
    return circle_reference(width=6, steps=1, balanced=False)


def run_example() -> None:
    compare_states(build_program(), reference_circuit(), ("x", "coin"))


if __name__ == "__main__":
    run_example()
