"""Two original asset distributions and an exactly scaled finite payoff model."""

from fractions import Fraction

PROBABILITIES = (
    0.065634503010069,
    0.434365496989931,
    0.434365496989931,
    0.065634503010069,
)
ANGLE_TABLE = (
    1.46997779512378,
    1.46997779512378,
    1.46997779512378,
    1.46997779512378,
    1.613710603790383,
    1.870987524791402,
    2.226584824607679,
    3.141592653589793,
)


def effective_angles() -> list[float]:
    """Compose the original scaled maximum with its raw-index rotation table."""
    angles = []
    for word in range(16):
        x1, x2 = word % 4, word // 4
        payoff = max(Fraction(x1), Fraction(x1 + x2, 2) - Fraction(3, 2))
        raw = 2 * payoff
        if raw.denominator != 1 or not 0 <= raw < len(ANGLE_TABLE):
            raise ValueError(
                "Payoff is not exactly representable in the original selector"
            )
        angles.append(ANGLE_TABLE[int(raw)])
    return angles
