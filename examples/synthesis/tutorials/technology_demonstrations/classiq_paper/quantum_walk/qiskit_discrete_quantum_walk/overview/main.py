"""Run the native counterpart of the commented qiskit reference."""

from examples.synthesis.tutorials.technology_demonstrations.classiq_paper.quantum_walk.reference_workflows import (
    run_example,
)


def main() -> None:
    """Compile, verify and print the original six-bit source workflow."""
    run_example("qiskit")


if __name__ == "__main__":
    main()
