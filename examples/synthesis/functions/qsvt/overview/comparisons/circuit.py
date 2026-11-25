"""Compare newly fitted angles with the existing independent complete Circuit."""

from examples.synthesis.compare_states import compare_states
from examples.synthesis.functions.qsvt.comparisons.circuit import reference_circuit
from examples.synthesis.functions.qsvt.overview.main import run_workflow


def run_example() -> None:
    workflow = run_workflow()
    compare_states(
        workflow.program,
        reference_circuit(workflow.fit.phases),
        ("state", "ref", "ind", "qsvt_aux"),
    )


if __name__ == "__main__":
    run_example()
