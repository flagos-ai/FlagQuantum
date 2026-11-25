"""Run the literal Catalyst-reference QSVT compilation workflow with named output readout."""

import flagquantum as fq

from .workflow import run_workflow


def main() -> None:
    """Compile and verify. Example: ``python -m ...overview.main``."""
    workflow = run_workflow()
    print("width", workflow.resources.n_qubits)
    print("depth", workflow.resources.depth)
    print("CX", workflow.resources.operation_counts.get("cx", 0))
    print("U3", workflow.resources.operation_counts.get("u3", 0))
    print("synthesis_seconds", workflow.synthesis_seconds)
    print("conversion_seconds", workflow.conversion_seconds)
    print("local_workflow_seconds", workflow.elapsed_seconds)
    result = fq.run(workflow.program)
    print(dict(workflow.program.probabilities(result, registers=("qsvt_aux",))))


if __name__ == "__main__":
    main()
