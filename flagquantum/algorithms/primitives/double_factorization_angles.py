"""Quantize real orbital vectors for the adjacent Givens chain."""

from collections.abc import Sequence
from math import atan2, hypot, isclose, isfinite, tau
from numbers import Real


def orbital_rotation_words(vector: Sequence[float], bits: int) -> tuple[int, ...]:
    """Solve von Burg et al. Eq. 57 in increasing orbital-pair order.

    Each word stores theta in fractions of a turn. The single-particle
    Givens angle is twice theta. The signed last component fixes the last
    angle's quadrant. Halfway words round up and wrap modulo one turn.
    """
    if type(bits) is not int or bits < 1:
        raise ValueError("DF rotation precision must be positive")
    if not len(vector) or any(
        not isinstance(value, Real) or not isfinite(value) for value in vector
    ):
        raise ValueError("DF orbital vectors must be nonempty, finite and real")
    if not isclose(hypot(*vector), 1.0, rel_tol=1e-7):
        raise ValueError("DF orbital vectors must have unit norm")

    scale = 1 << bits
    tail = vector[-1]
    words = []
    for value in reversed(vector[:-1]):
        turns = atan2(tail, value) / (2 * tau)
        numerator, denominator = turns.as_integer_ratio()
        word = ((numerator << (bits + 1)) + denominator) // (2 * denominator)
        words.append(word % scale)
        tail = hypot(value, tail)
    return tuple(reversed(words))
