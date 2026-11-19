"""Run the original eight-signal native BB84 teaching workflow."""

from examples.synthesis.community.paper_implementation_project.bb84.bb84_implementation.main import (
    ALICE_BASES,
    BOB_BASES,
    MESSAGE,
)

from .workflow import run_workflow


def main() -> None:
    """Report actual preparation inputs, observations and illustrative parity."""
    workflow = run_workflow()
    print("Actual message:", MESSAGE)
    print("Actual Alice bases:", ALICE_BASES)
    print("Actual Bob bases:", BOB_BASES)
    print("Observations:", sum(row.count for row in workflow.observations))
    print("Most observed packet:", workflow.selected)
    print("Teaching postprocessing:", workflow.report)
    print("Resources:", workflow.program.report["resources"])
    print("Temporary workspace:", workflow.program.report["workspace_qubits"])


if __name__ == "__main__":
    main()
