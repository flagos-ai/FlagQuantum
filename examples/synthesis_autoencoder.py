"""Qmod lexical bindings, native scalar qfunc composition and independent Circuit.

The original uses array helpers and scoped bindings. Native qfunc comparison uses
existing scalar ports and explicit static R/CX calls, without claiming new native
array or loop support. Two nonzero parameter bindings are checked in full state.
"""

import math
from pathlib import Path

import torch

import flagquantum as fq
from flagquantum import synth
from flagquantum.services.synthesis import _synthesize
from flagquantum.synth import CX, Output, Program, QBit, R, allocate, qfunc

HALF_PI = math.pi / 2
WEIGHTS = (
    (0.11, -0.23, 0.37, -0.41, 0.53, -0.67, 0.71, -0.83, 0.97, -1.09),
    (-0.17, 0.29, -0.47, 0.61, -0.73, 0.89, -1.01, 1.13, -1.27, 1.39),
)
DATA = ((0.19, -0.31, 0.43, -0.59), (-0.21, 0.34, -0.55, 0.68))


def native_program(case: int) -> Program:
    a0, a1, a2, a3 = (math.pi * value for value in DATA[case])
    w0, w1, w2, w3, w4, w5, w6, w7, w8, w9 = WEIGHTS[case]

    @qfunc
    def main(
        trash1: Output[QBit],
        trash0: Output[QBit],
        coded1: Output[QBit],
        coded0: Output[QBit],
    ) -> None:
        allocate(trash1)
        allocate(trash0)
        allocate(coded1)
        allocate(coded0)
        R(a0, HALF_PI, coded0)
        R(a1, HALF_PI, coded1)
        R(a2, HALF_PI, trash0)
        R(a3, HALF_PI, trash1)
        R(w0, HALF_PI, coded0)
        R(w1, HALF_PI, coded1)
        R(w2, HALF_PI, trash0)
        R(w3, HALF_PI, trash1)
        CX(coded0, coded1)
        CX(coded1, trash0)
        CX(trash0, trash1)
        R(w4, HALF_PI, coded0)
        R(w5, HALF_PI, coded1)
        R(w6, HALF_PI, trash0)
        R(w7, HALF_PI, trash1)
        CX(coded0, coded1)
        CX(coded1, trash0)
        CX(trash0, trash1)
        R(w8, HALF_PI, trash0)
        R(w9, HALF_PI, trash1)

    return synth.synthesize(main, memory_limit_bytes=64 * 1024**2)


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
    source = (
        Path(__file__).resolve().parents[1]
        / "tests/fixtures/classiq_qmod/quantum_autoencoder.qmod"
    )
    for case in (0, 1):
        expected = fq.run(circuit(case)).statevector().reshape(-1)
        programs = (
            _synthesize(
                source, classical_inputs={"w": WEIGHTS[case], "input_data": DATA[case]}
            ),
            native_program(case),
        )
        for index, program in enumerate(programs):
            result = fq.run(program)
            state = result.statevector().reshape(16, -1)
            torch.testing.assert_close(state[:, 0], expected, atol=1e-12, rtol=0)
            assert float(state[:, 1:].abs().square().sum()) < 1e-24
            registers = ("trash",) if index == 0 else ("trash1", "trash0")
            print(case, dict(program.probabilities(result, registers=registers)))
            print(program.report["resources"])
    print("Original Qmod, native scalar qfunc and independent Circuit agree.")


if __name__ == "__main__":
    run_example()
