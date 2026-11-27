"""Tuned product preparation and truncated Fourier transform in logical ports."""

import math

from flagquantum.synth import (
    SWAP,
    H,
    Output,
    QArray,
    QBit,
    R,
    X,
    allocate,
    control,
    phase,
    qfunc,
)


@qfunc
def rotate_y(angle: float, q: QBit) -> None:
    R(angle, math.pi / 2, q)


@qfunc
def gaussian_tree(
    width: int,
    rotations: list[float],
    phases: list[float],
    tweak: float,
    q: Output[QArray[QBit]],
) -> None:
    allocate(width, q)
    for bit in range(width):
        if rotations[bit] > 0.000001:
            R(rotations[bit], math.pi / 2, q[bit])
    control(q[0], lambda: rotate_y(tweak, q[1:2]))
    for bit in range(width):
        if 2 * bit < width - 1:
            SWAP(q[bit], q[width - 1 - bit])
    for bit in range(width):
        H(q[bit])
        for distance in range(1, 9):
            if bit + distance < width:
                if phases[distance - 1] > 0.01:
                    phase(q[bit + distance] * q[bit], phases[distance - 1])
    X(q[0])
