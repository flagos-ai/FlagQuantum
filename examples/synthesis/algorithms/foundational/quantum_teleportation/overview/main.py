"""Run native teleportation and inspect Bob's actual state and observations."""

from .workflow import run_workflow


def main() -> None:
    """Read the original plus-state teleportation using public native APIs."""
    workflow = run_workflow()
    print("Bob density matrix:", workflow.density)
    print("Bob observations:", dict(workflow.observations))
    print("Resources:", workflow.program.report["resources"])
    print("Clean workspace:", workflow.program.report["workspace_qubits"])


if __name__ == "__main__":
    main()
