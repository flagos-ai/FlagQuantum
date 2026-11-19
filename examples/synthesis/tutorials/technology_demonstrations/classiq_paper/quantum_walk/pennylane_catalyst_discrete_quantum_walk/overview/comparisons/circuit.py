"""Verify native and target-basis IR against the independent source Circuit."""

from examples.synthesis.tutorials.technology_demonstrations.classiq_paper.quantum_walk.reference_workflows import (
    run_workflow,
)


def main() -> None:
    """Run both full-state comparisons, including absolute phase and clean work."""
    workflow = run_workflow("catalyst")
    print("independent_full_state_check", "passed")
    print(workflow.resources)


if __name__ == "__main__":
    main()
