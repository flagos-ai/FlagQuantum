"""Finite Rainbow model and checked classical exponential-loading parameters."""

import math

EXPONENT_RATE = 0.434093883854633
PROBABILITIES = (
    0.065634503010069,
    0.434365496989931,
    0.434365496989931,
    0.065634503010069,
)
FRACTION_BITS = 2
PAYOFF_WIDTH = 5
PRICE_SCALE = 193.97
LOG_SHIFT = -0.740780267709267
STRIKE = 190.0


def classical_parameters(rate: float = EXPONENT_RATE) -> tuple[list[float], float]:
    """Bind rotations for the original normalized exponential integrator."""
    if type(rate) not in (float, int) or not math.isfinite(rate) or not 0 < rate <= 4:
        raise ValueError("Use a finite positive exponent rate at most 4")
    step = rate / (1 << FRACTION_BITS)
    scale = PRICE_SCALE * math.exp(LOG_SHIFT)
    correction = math.exp(-step)
    probability = (STRIKE - scale * correction) / (
        scale * math.expm1((1 << PAYOFF_WIDTH) * step) * correction
    )
    if not 0 <= probability <= 1:
        raise ValueError("Exponent rate gives a low-branch probability outside [0, 1]")
    angles = [
        2 * math.atan(math.exp(step * (1 << bit) / 2)) for bit in range(PAYOFF_WIDTH)
    ]
    return angles, 2 * math.asin(math.sqrt(probability))
