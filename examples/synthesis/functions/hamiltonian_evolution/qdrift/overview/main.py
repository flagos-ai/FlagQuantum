"""Run the native qdrift Notebook computation without Jupyter."""

import flagquantum as fq
from examples.synthesis.functions.hamiltonian_evolution.qdrift.main import build_program


def run_example() -> None:
    for program in (build_program(),):
        result = fq.run(program)
        print("Probabilities:", dict(program.probabilities(result, registers=("qba",))))
        print("Resources:", program.report["resources"])


if __name__ == "__main__":
    run_example()
