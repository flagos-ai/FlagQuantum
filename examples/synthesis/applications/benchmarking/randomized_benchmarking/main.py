"""Retained randomized sequence followed by its checked inverse."""

import math

import flagquantum as fq
from flagquantum import synth
from flagquantum.synth import (
    RX,
    RZ,
    Output,
    QArray,
    QBit,
    X,
    adjoint,
    allocate,
    phase,
    qfunc,
    unitary,
)

IDENTITY = ((1, 0), (0, 1))


@qfunc
def sx(q: QBit) -> None:
    # RX alone differs from SX by a physically relevant global phase.
    phase(1, math.pi / 4)
    RX(math.pi / 2, q)


@qfunc
def sequence(target: QArray[QBit]) -> None:
    RZ(math.pi, target[0])
    X(target[0])
    sx(target[0])
    unitary(IDENTITY, target[0])
    unitary(IDENTITY, target[0])
    sx(target[0])
    unitary(IDENTITY, target[0])
    RZ(math.pi, target[0])
    sx(target[0])
    RZ(math.pi, target[0])
    X(target[0])
    X(target[0])
    unitary(IDENTITY, target[0])
    X(target[0])
    unitary(IDENTITY, target[0])
    sx(target[0])
    RZ(math.pi, target[0])
    X(target[0])
    RZ(math.pi, target[0])
    X(target[0])
    X(target[0])
    X(target[0])
    RZ(math.pi, target[0])
    X(target[0])
    unitary(IDENTITY, target[0])


@qfunc
def main(target: Output[QArray[QBit]]) -> None:
    allocate(1, target)
    sequence(target)
    with adjoint():
        sequence(target)


def build_program() -> synth.Program:
    return synth.synthesize(main, max_workspace=0)


if __name__ == "__main__":
    program = build_program()
    print(dict(program.probabilities(fq.run(program), registers=("target",))))
