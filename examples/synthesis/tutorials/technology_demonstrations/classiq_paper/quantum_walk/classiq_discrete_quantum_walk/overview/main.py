"""Compile the six-bit walk and report the measured CX/U3 candidate costs."""

import flagquantum as fq

from .workflow import run_workflow


def main() -> None:
    """Run the local bounded selection. Example: ``python -m ...overview.main``."""
    workflow = run_workflow()
    for candidate in workflow.candidates:
        cost = candidate.resources
        print(
            candidate.name,
            "qubits",
            cost.n_qubits,
            "depth",
            cost.depth,
            "CX",
            cost.operation_counts.get("cx", 0),
            "U3",
            cost.operation_counts.get("u3", 0),
            "synthesis_seconds",
            candidate.synthesis_seconds,
            "conversion_seconds",
            candidate.conversion_seconds,
        )
    chosen = workflow.selected
    result = fq.run(chosen.program)
    print("selected", chosen.name)
    print(dict(chosen.program.probabilities(result, registers=("x", "coin"))))
    print("local_workflow_seconds", workflow.elapsed_seconds)


if __name__ == "__main__":
    main()
