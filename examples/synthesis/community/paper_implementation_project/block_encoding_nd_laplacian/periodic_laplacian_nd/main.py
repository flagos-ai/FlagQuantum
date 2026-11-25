"""Split-coordinate periodic Laplacian using complete-register Fourier shifts."""

import math

from flagquantum import synth
from flagquantum.synth import (
    Output,
    QArray,
    QBit,
    QNum,
    adjoint,
    allocate,
    hadamard_transform,
    phase,
    qft,
    qfunc,
)


@qfunc
def shift_coordinate(
    position: QNum,
    selector: QArray[QBit],
    direction: QBit,
    dimension: int,
    period: float,
) -> None:
    qft(position)
    if dimension == 0:
        phase(
            (1 - direction) * (selector[0] + selector[1] - 1) * position,
            math.tau / period,
        )
    else:
        phase(direction * (selector[0] + selector[1] - 1) * position, math.tau / period)
    with adjoint():
        qft(position)


@qfunc
def main(
    j1: Output[QNum],
    j0: Output[QBit],
    selector: Output[QArray[QBit]],
    dimension: Output[QBit],
) -> None:
    allocate(2, j1)
    allocate(j0)
    allocate(2, selector)
    allocate(dimension)
    hadamard_transform(selector)
    phase(selector[0] + selector[1], math.pi)
    hadamard_transform(dimension)
    shift_coordinate(j0, selector, dimension, 0, 2.0)
    shift_coordinate(j1, selector, dimension, 1, 4.0)
    hadamard_transform(selector)
    hadamard_transform(dimension)


def build_program() -> synth.Program:
    return synth.synthesize(main, max_workspace=0)
