"""Independent six-bit gate construction for the numeric composition tutorial."""

import cmath

import torch

import flagquantum as fq
from examples.synthesis.compare_states import compare_states
from examples.synthesis.tutorials.composition.numeric_program.main import build_program


def reference_circuit() -> fq.Circuit:
    """Prepare the source and reversibly write both exact arithmetic results."""
    circuit = fq.Circuit(6, dtype=torch.complex128).h(0)
    circuit.any(
        0,
        matrix=torch.diag(
            torch.tensor(
                [cmath.exp(0.37j * 0.125), cmath.exp(0.37j * -0.375)],
                dtype=torch.complex128,
            )
        ),
    )
    # x raw 0/1 means 0/-1/2. doubled raw 00/10 means 0/-1;
    # result raw 001/111 means 1/2 or -1/2.
    return circuit.cx(0, 1).x(5).cx(0, 3).cx(0, 4)


if __name__ == "__main__":
    compare_states(build_program(), reference_circuit(), ("x", "doubled", "result"))
