"""Compose affine Y rotations controlled by an unsigned quantum register."""

import math
from collections.abc import Sequence

import flagquantum as fq
from flagquantum import synth
from flagquantum.synth import (
    PHASE,
    Output,
    QArray,
    QBit,
    R,
    X,
    allocate,
    control,
    hadamard_transform,
    qfunc,
)

SLOPES = (2.1, 1.0, 7.0)
OFFSETS = (0.1, 0.3, 0.33)


@qfunc
def rotate_if_one(angle: float, selector: QBit, target: QBit) -> None:
    control(selector, lambda: R(angle, math.pi / 2, target))


@qfunc
def main(
    angles: list[float],
    offsets: list[float],
    input_bits: list[float],
    coherent: int,
    x: Output[QArray[QBit]],
    ind: Output[QArray[QBit]],
) -> None:
    allocate(6, x)
    allocate(3, ind)
    for bit in range(6):
        if input_bits[bit] != 0:
            X(x[bit])
    if coherent == 1:
        hadamard_transform(x)
        PHASE(math.pi, x[0])
    for target in range(3):
        R(offsets[target], math.pi / 2, ind[target])
        for bit in range(6):
            rotate_if_one(angles[6 * target + bit], x[bit], ind[target])


def build_program(
    slopes: Sequence[float] = SLOPES,
    offsets: Sequence[float] = OFFSETS,
    *,
    input_value: int = 0,
    coherent: bool = False,
) -> synth.Program:
    """Rotate target j by slopes[j] * x + offsets[j], without work qubits."""
    if len(slopes) != 3 or len(offsets) != 3:
        raise ValueError("three target slopes and offsets are required")
    if type(input_value) is not int or not 0 <= input_value < 64:
        raise ValueError("input_value must be an unsigned six-bit integer")
    return synth.synthesize(
        main,
        classical_inputs={
            "angles": [slope * (1 << bit) for slope in slopes for bit in range(6)],
            "offsets": list(offsets),
            "input_bits": [float((input_value >> bit) & 1) for bit in range(6)],
            "coherent": int(coherent),
        },
        max_workspace=0,
    )


if __name__ == "__main__":
    program = build_program()
    result = fq.run(program)
    print(dict(program.probabilities(result, registers=("x", "ind"))))
    print(program.report["resources"])
