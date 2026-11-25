"""Checkerboard block encoding with explicit packed-coordinate fields."""

import math

import flagquantum as fq
from flagquantum import synth
from flagquantum.synth import (
    CX,
    SWAP,
    Output,
    QArray,
    QBit,
    R,
    allocate,
    apply,
    control,
    hadamard_transform,
    qfunc,
    within,
)

ANGLES = (2 * math.acos(-0.69), 2 * math.acos(-0.34))


@qfunc
def block(
    angles: list[float],
    data: QBit,
    s1: QArray[QBit],
    s0: QBit,
    j1: QBit,
    j0: QArray[QBit],
) -> None:
    SWAP(s1[0], j0[0])
    SWAP(s1[1], j0[1])
    CX(s0, j1)
    R(angles[0], 0, data)
    control(s0, lambda: R(angles[1] - angles[0], 0, data))


@qfunc
def main(
    angles: list[float],
    data: Output[QBit],
    s1: Output[QArray[QBit]],
    s0: Output[QBit],
    j1: Output[QBit],
    j0: Output[QArray[QBit]],
) -> None:
    allocate(data)
    allocate(2, s1)
    allocate(s0)
    allocate(j1)
    allocate(2, j0)
    hadamard_transform(j1)
    hadamard_transform(j0)
    with within():
        hadamard_transform(s1)
        hadamard_transform(s0)
    with apply():
        block(angles, data, s1, s0, j1, j0)


def build_program(angles: tuple[float, float] = ANGLES) -> synth.Program:
    return synth.synthesize(
        main, classical_inputs={"angles": list(angles)}, max_workspace=0
    )


if __name__ == "__main__":
    program = build_program()
    result = fq.run(program)
    print(
        dict(program.probabilities(result, registers=("data", "s1", "s0", "j1", "j0")))
    )
    print(program.report["resources"])
