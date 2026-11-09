"""Compose the retained oracle and phase-exact uniform reflection."""

from math import pi

import flagquantum as fq
from flagquantum import synth
from flagquantum.synth import (
    Output,
    QArray,
    QBit,
    allocate,
    hadamard_transform,
    phase,
    qfunc,
)


@qfunc
def mark(q: QArray[QBit]) -> None:
    phase(
        (1 - (1 - q[1]) * (1 - q[2]) * (1 - q[3]))
        * (1 - q[0] * (1 - q[1]) * (1 - q[2]))
        * (1 - q[0] * (1 - q[1]) * q[2])
        * (1 - q[0] * q[1] * (1 - q[2]))
        * (1 - (1 - q[0]) * q[1] * q[2])
        * (1 - (1 - q[0]) * q[1] * (1 - q[2]))
        * (1 - q[0] * q[1] * q[3])
        * (1 - q[0] * q[1] * (1 - q[3]))
        * (1 - q[1] * q[2] * q[3])
        * (1 - (1 - q[1]) * q[2] * (1 - q[3]))
        * (1 - (1 - q[0]) * q[2] * (1 - q[3]))
        * (1 - (1 - q[0]) * q[1] * q[3])
        * (1 - q[0] * q[1] * q[2]),
        pi,
    )


@qfunc
def diffuse(q: QArray[QBit]) -> None:
    hadamard_transform(q)
    phase((1 - q[0]) * (1 - q[1]) * (1 - q[2]) * (1 - q[3]), pi)
    phase(1, pi)
    hadamard_transform(q)


@qfunc
def main(r: int, q: Output[QArray[QBit]]) -> None:
    allocate(4, q)
    hadamard_transform(q)
    for i in range(r):
        mark(q)
        diffuse(q)


def build_program(r: int = 2) -> synth.Program:
    if type(r) is not int or not 0 <= r <= 32:
        raise ValueError("Use an integer Grover iteration count in 0..32")
    return synth.synthesize(main, classical_inputs={"r": r})


def run_example() -> None:
    program = build_program()
    print(dict(program.probabilities(fq.run(program), registers=("q",))))
    print(program.report["resources"])


if __name__ == "__main__":
    run_example()
