"""Retained commuting Hamiltonian and approximate reciprocal rotation network."""

import math

import flagquantum as fq

HAMILTONIAN = (
    0.03 * (fq.X(0) @ fq.X(1))
    + 0.05 * (fq.Z(1) @ fq.Z(0))
    + 0.02 * (fq.Y(0) @ fq.Y(1))
    + 0.08 * fq.I()
)
RHS = (math.sqrt(1 / 6), math.sqrt(1 / 6), math.sqrt(1 / 6), math.sqrt(1 / 2))
ROTATIONS = (
    -0.0157,
    -0.0157,
    -0.3379,
    0.3066,
    -0.1047,
    -0.1047,
    -0.3181,
    0.4649,
    -0.0475,
    -0.0475,
    -0.3407,
    0.2457,
    -0.0939,
    -0.0939,
    -0.3122,
    0.8154,
)
CONTROL_BITS = (0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 3)


def inversion_angle(word: int) -> float:
    """Move the retained even-parity X network through Y rotations exactly."""
    parity, total = 0, 0.0
    for theta, bit in zip(ROTATIONS, CONTROL_BITS, strict=True):
        total += theta * (-1 if parity else 1)
        parity ^= (word >> bit) & 1
    if parity:
        raise ValueError("retained inversion network must restore the indicator parity")
    return total


INVERSION_ANGLES = tuple(inversion_angle(word) for word in range(16))
