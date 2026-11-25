"""Run the flexible-QPE source's concrete diagonal Hamiltonian instance."""

import flagquantum as fq
from examples.synthesis.functions.qpe.main import main
from flagquantum import synth


def build_program() -> synth.Program:
    """Both frozen QPE examples have exactly the same quantum program."""
    return synth.synthesize(main, max_workspace=0)


if __name__ == "__main__":
    program = build_program()
    result = fq.run(program)
    print(dict(program.numeric_probabilities(result, registers=("state", "phase"))))
    print(program.report["resources"])
