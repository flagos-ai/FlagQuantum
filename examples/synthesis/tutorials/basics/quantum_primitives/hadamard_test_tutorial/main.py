"""Share the native program for the byte-identical Hadamard-test tutorial."""

import flagquantum as fq
from examples.synthesis.algorithms.quantum_primitives.hadamard_test.main import (
    build_program,
    main,
)

__all__ = ("build_program", "main")


if __name__ == "__main__":
    program = build_program()
    result = fq.run(program)
    print(dict(program.probabilities(result, registers=("expectation_value",))))
    print(program.report["resources"])
