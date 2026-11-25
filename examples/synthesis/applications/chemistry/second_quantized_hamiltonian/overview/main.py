"""Native four-mode variational energy minimization with explicit diagnostics."""

from .analysis import run_workflow


def run_example() -> None:
    """Report the full-space energy search and actual particle sectors."""
    result = run_workflow()
    print("Initial energy:", result.optimization.initial_energy)
    print("Best visited energy:", result.energy)
    print("Evaluations:", len(result.optimization.trace))
    print("Termination:", result.optimization.termination)
    print("Best parameters:", result.optimization.parameters)
    print("Particle-number populations:", dict(result.sectors))
    print("The X/RY/CX ansatz does not conserve the source particle metadata (1, 1).")
    print("Resources:", result.program.report["resources"])


if __name__ == "__main__":
    run_example()
