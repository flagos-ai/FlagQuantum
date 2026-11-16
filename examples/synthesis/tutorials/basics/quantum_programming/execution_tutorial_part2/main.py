"""Prepare a parameterized entangled pair using named quantum outputs."""

import math

import flagquantum as fq
from flagquantum.synth import CX, RX, Output, QBit, R, allocate, qfunc, synthesize


@qfunc
def main(angles: list[float], x: Output[QBit], y: Output[QBit]) -> None:
    allocate(x)
    allocate(y)
    RX(angles[0], x)
    # R about azimuth pi/2 is a Y rotation, including its absolute phase.
    R(angles[1], math.pi / 2, x)
    CX(x, y)


def run_example() -> None:
    program = synthesize(main, classical_inputs={"angles": [0.37, -0.81]})
    result = fq.run(program)
    print(dict(program.probabilities(result, registers=("x", "y"))))


if __name__ == "__main__":
    run_example()
