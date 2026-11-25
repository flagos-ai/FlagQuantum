"""The IQAE and counting originals share the same quantum program."""

import torch

import flagquantum as fq

from ...comparisons.circuit import preparation_matrix, reference_circuit

__all__ = ["preparation_matrix", "reference_circuit"]


def run_example() -> None:
    from ..main import build_program

    for k in (0, 1, 2):
        program = build_program(k)
        actual = fq.run(program).statevector().flatten()
        expected = fq.run(reference_circuit(k)).statevector().flatten()
        torch.testing.assert_close(actual, expected, atol=1e-11, rtol=0)
        print(k, program.report["resources"])
    print("Full states and absolute phases agree for zero, one and two iterations.")


if __name__ == "__main__":
    run_example()
