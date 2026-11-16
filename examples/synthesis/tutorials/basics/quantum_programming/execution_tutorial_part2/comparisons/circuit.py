"""Check two nonzero bindings against independent native Circuit rotations."""

import torch

import flagquantum as fq
from examples.synthesis.compare_states import compare_states
from examples.synthesis.tutorials.basics.quantum_programming.execution_tutorial_part2.main import (
    main,
)
from flagquantum.synth import synthesize

BINDINGS = ({"angles": [0.37, -0.81]}, {"angles": [-1.12, 0.49]})
REGISTERS = ("x", "y")


def circuit_reference(angles: list[float]) -> fq.Circuit:
    return (
        fq.Circuit(2, dtype=torch.complex128)
        .rx(0, theta=angles[0])
        .ry(0, theta=angles[1])
        .cx(0, 1)
    )


def run_example() -> None:
    for bindings in BINDINGS:
        program = synthesize(main, classical_inputs=bindings)
        compare_states(program, circuit_reference(**bindings), REGISTERS)
        assert program.report["workspace_qubits"] == 0


if __name__ == "__main__":
    run_example()
