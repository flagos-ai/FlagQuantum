"""Compare the newly factored GQSP model to an independent complete Circuit."""

from examples.synthesis.algorithms.quantum_primitives.gqsp.comparisons.circuit import (
    reference_circuit,
)
from examples.synthesis.algorithms.quantum_primitives.gqsp.overview.main import (
    run_workflow,
)
from examples.synthesis.compare_states import compare_states


def run_example() -> None:
    workflow = run_workflow()
    compare_states(
        workflow.program, reference_circuit(phases=workflow.fit.phases), ("x", "ind")
    )


if __name__ == "__main__":
    run_example()
