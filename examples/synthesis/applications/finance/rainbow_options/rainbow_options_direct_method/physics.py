"""Classical model constants and angles for the pinned direct rainbow payoff."""

import math

PROBABILITIES = (
    0.065634503010069,
    0.434365496989931,
    0.434365496989931,
    0.065634503010069,
)
EXPONENT_RATE = 0.434093883854633
PRICE_SCALE = 193.97
LOG_SHIFT = -0.740780267709267
STRIKE = 190.0
LOW_SCALE = 13.784048752090222
FRACTION_BITS = 2
PAYOFF_WIDTH = 5


def loading_angles(rate: float = EXPONENT_RATE) -> list[float]:
    """Compute ordinary classical angles before capturing the quantum program."""
    if type(rate) not in (int, float) or not math.isfinite(rate) or rate <= 0:
        raise ValueError("Use a positive finite exponent rate")
    step = rate / (1 << FRACTION_BITS)
    return [
        2 * math.acos(math.exp(-step * (1 << bit) / 2)) for bit in range(PAYOFF_WIDTH)
    ]


def low_angle(rate: float = EXPONENT_RATE) -> float:
    """Retain the source's lower-branch payoff formula and five-bit range."""
    maximum = ((1 << PAYOFF_WIDTH) - 1) / (1 << FRACTION_BITS)
    return 2 * math.asin(
        LOW_SCALE
        / math.sqrt(max(PRICE_SCALE * math.exp(rate * maximum + LOG_SHIFT), STRIKE))
    )
