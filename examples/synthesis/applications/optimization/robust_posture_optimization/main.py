"""Compose parameterized posture rotations without assigning physical wires.

This is the quantum ansatz. Robot geometry, measurement and optimization remain
separate workflow requirements in the migration inventory.
"""

import math

import flagquantum as fq
from flagquantum.synth import (
    RX,
    RZ,
    Output,
    QArray,
    QBit,
    R,
    allocate,
    qfunc,
    synthesize,
)


@qfunc
def first_rotation(theta: list[float], target: QBit) -> None:
    RZ(theta[2], target)
    R(theta[1], math.pi / 2, target)
    RX(theta[0], target)


@qfunc
def second_rotation(theta: list[float], target: QBit) -> None:
    RZ(theta[5], target)
    R(theta[4], math.pi / 2, target)
    RX(theta[3], target)


@qfunc
def main(theta: list[float], q: Output[QArray[QBit]]) -> None:
    allocate(2, q)
    first_rotation(theta, q[0])
    second_rotation(theta, q[1])
    first_rotation(theta, q[1])


def run_example() -> None:
    program = synthesize(
        main,
        classical_inputs={"theta": [0.17, -0.23, 0.41, -0.59, 0.67, -0.83]},
    )
    result = fq.run(program)
    print(dict(program.probabilities(result, registers=("q",))))


if __name__ == "__main__":
    run_example()
