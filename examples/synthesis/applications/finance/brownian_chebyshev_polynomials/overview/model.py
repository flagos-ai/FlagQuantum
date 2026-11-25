"""Generate the Gaussian bins and the exact integer reference vectors."""

import math
from dataclasses import dataclass
from fractions import Fraction


@dataclass(frozen=True)
class Gaussian:
    points: tuple[float, ...]
    probabilities: tuple[float, ...]


def gaussian(bits: int) -> Gaussian:
    """Integrate unit Gaussian mass in the original left-endpoint bins."""
    if bits not in (1, 4):
        raise ValueError("The Notebook uses the one- and four-bit source grids")
    count = 1 << bits
    edges = tuple(-3 + 6 * index / count for index in range(count + 1))
    cdf = tuple(math.erf(value / math.sqrt(2)) / 2 for value in edges)
    masses = tuple(
        upper - lower for lower, upper in zip(cdf[:-1], cdf[1:], strict=True)
    )
    total = math.fsum(masses)
    return Gaussian(edges[:-1], tuple(mass / total for mass in masses))


def sine_values() -> tuple[Fraction, ...]:
    """The source fits sine with a quadratic around one-half."""
    return tuple(1 - 5 * (Fraction(word, 8) - Fraction(1, 2)) ** 2 for word in range(8))


def chebyshev_values() -> tuple[tuple[Fraction, ...], ...]:
    """Preserve the source standalone demo's x factor in the two later terms."""
    values = []
    for word in range(4):
        x = Fraction(word, 4)
        u0, u1 = Fraction(1), 2 * x
        u2 = x * u1 - u0
        u3 = x * u2 - u1
        values.append((x, u0, u1, u2, u3))
    return tuple(values)
