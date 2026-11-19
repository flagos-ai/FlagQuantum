"""Run the native linear Notebook computation without Jupyter."""

import flagquantum as fq
from examples.synthesis.functions.linear_pauli_rotations.main import build_program


def run_example() -> None:
    for program in (build_program(), build_program(coherent=True)):
        result = fq.run(program)
        print(
            "Probabilities:",
            dict(program.probabilities(result, registers=("x", "ind"))),
        )
        print("Resources:", program.report["resources"])


if __name__ == "__main__":
    run_example()
