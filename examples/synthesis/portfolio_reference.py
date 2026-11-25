"""Independent three-asset covariance and budget factors for portfolio examples."""

from examples.synthesis.factor_reference import CostFactor
from examples.synthesis.quadratic_reference import affine_square_factors

COVARIANCE = ((0.9, 0.5, -0.7), (0.5, 0.9, -0.2), (-0.7, -0.2, 0.9))
RETURNS = (3.0, 4.0, -1.0)


def portfolio_cost(
    values: tuple[int, int, int],
    slack: float,
    penalty: float,
    return_scale: float,
    sign: float,
) -> float:
    covariance = sum(
        COVARIANCE[i][j] * values[i] * values[j] for i in range(3) for j in range(3)
    )
    returns = return_scale * sum(
        rate * value for rate, value in zip(RETURNS, values, strict=True)
    )
    return sign * (covariance - returns + penalty * (sum(values) + slack - 6) ** 2)


def portfolio_factors(
    width: int,
    slack_weights: tuple[float, ...],
    penalty: float,
    return_scale: float,
    sign: float,
) -> list[CostFactor]:
    """Tabulate each at-most-six-bit covariance term independently of Compiler."""
    if width not in (2, 3):
        raise ValueError("Retained portfolio words have width two or three")
    factors: list[CostFactor] = []
    mask = (1 << width) - 1
    for first in range(3):
        for second in range(first, 3):
            support = tuple(range(first * width, (first + 1) * width))
            if second != first:
                support += tuple(range(second * width, (second + 1) * width))
            coefficient = (
                COVARIANCE[first][second] * (1 if first == second else 2) * sign
            )
            values = tuple(
                coefficient
                * (basis & mask)
                * ((basis & mask) if first == second else (basis >> width) & mask)
                for basis in range(1 << len(support))
            )
            factors.append((support, values))
    for asset, rate in enumerate(RETURNS):
        for bit in range(width):
            factors.append(
                (
                    (asset * width + bit,),
                    (0.0, -sign * return_scale * rate * (1 << bit)),
                )
            )
    count = 3 * width + len(slack_weights)
    coefficients = (
        tuple(float(1 << bit) for _ in range(3) for bit in range(width)) + slack_weights
    )
    factors.extend(
        affine_square_factors(tuple(range(count)), coefficients, -6.0, sign * penalty)
    )
    return factors
