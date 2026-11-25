"""Compare native FlagQuantum and independent Circuit amplitudes."""

import math

import torch

import flagquantum as fq
from examples.synthesis.algorithms.machine_learning.quantum_autoencoder.main import (
    DATA,
    WEIGHTS,
    native_program,
)


def circuit(case: int) -> fq.Circuit:
    c = fq.Circuit(4, dtype=torch.complex128)
    for index, value in enumerate(DATA[case]):
        c.ry(3 - index, theta=math.pi * value)
    for layer in range(2):
        for index in range(4):
            c.ry(3 - index, theta=WEIGHTS[case][4 * layer + index])
        c.cx(3, 2).cx(2, 1).cx(1, 0)
    return c.ry(1, theta=WEIGHTS[case][8]).ry(0, theta=WEIGHTS[case][9])


def run_example() -> None:
    for case in (0, 1):
        expected = fq.run(circuit(case)).statevector().reshape(-1)
        programs = (native_program(case),)
        for index, program in enumerate(programs):
            result = fq.run(program)
            state = result.statevector().reshape(16, -1)
            torch.testing.assert_close(state[:, 0], expected, atol=1e-12, rtol=0)
            assert float(state[:, 1:].abs().square().sum()) < 1e-24
            registers = ("trash1", "trash0")
            print(case, dict(program.probabilities(result, registers=registers)))
            print(program.report["resources"])
    print(
        "Native qfunc and Circuit agree, including absolute phase and clean workspace."
    )


if __name__ == "__main__":
    run_example()
