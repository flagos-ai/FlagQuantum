"""Rebuild the original direct exponential payoff and its exact finite reference."""

import math
from dataclasses import dataclass
from fractions import Fraction

STRIKE = 190.0
SOURCE_PRICE = 23.0238
EPSILON = 0.05
ALPHA = 0.1
ASSERTION_ALPHA = 1e-5
FRACTION_BITS = 2
SELECTOR_BITS = 5


@dataclass(frozen=True)
class Model:
    grid: tuple[float, ...]
    probabilities: tuple[float, ...]
    coefficients: tuple[tuple[Fraction, ...], ...]
    offset: Fraction
    threshold: Fraction
    selectors: tuple[int, ...]
    rate: float
    angles: tuple[float, ...]
    lower_angle: float
    scale: float
    payoff: tuple[float, ...]
    exact_price: float


def fresh_model() -> Model:
    days = 250
    spot = (193.97, 189.12)
    means = (0.00050963 * days, 0.00062552 * days)
    variance1, covariance, variance2 = 0.000335, 0.000257, 0.000418
    first = math.sqrt(variance1)
    lower = covariance / first
    chol = (
        (first * math.sqrt(days), 0.0),
        (lower * math.sqrt(days), math.sqrt(variance2 - lower**2) * math.sqrt(days)),
    )
    edges = (-3.0, -1.5, 0.0, 1.5, 3.0)
    cdf = tuple((1 + math.erf(edge / math.sqrt(2))) / 2 for edge in edges)
    probabilities = tuple(
        (right - left) / (cdf[-1] - cdf[0])
        for left, right in zip(cdf[:-1], cdf[1:], strict=True)
    )
    fraction_scale = 1 << FRACTION_BITS
    scaling = 1 / chol[0][0]
    rate = (edges[1] - edges[0]) / scaling
    coefficients = tuple(
        tuple(
            Fraction(round(fraction_scale * scaling * value), fraction_scale)
            for value in row
        )
        for row in chol
    )
    log_shift = means[0] + edges[0] * sum(chol[0])
    relative_log = math.log(spot[1] / spot[0]) + means[1] - means[0]
    relative_log += edges[0] * (sum(chol[1]) - sum(chol[0]))
    offset = Fraction(round(fraction_scale * relative_log / rate), fraction_scale)
    threshold = Fraction(
        math.floor(fraction_scale * (math.log(STRIKE / spot[0]) - log_shift) / rate),
        fraction_scale,
    )
    selectors = []
    for word in range(16):
        assets = (word % 4, word // 4)
        left, right = (
            sum(c * value for c, value in zip(row, assets, strict=True))
            for row in coefficients
        )
        index = fraction_scale * max(left, right + offset)
        if index.denominator != 1 or not 0 <= index < 1 << SELECTOR_BITS:
            raise ValueError(
                "The exact affine maximum must fit the unsigned source selector"
            )
        selectors.append(int(index))

    payoff = tuple(
        max(spot[0] * math.exp(log_shift + rate * index / fraction_scale), STRIKE)
        for index in range(1 << SELECTOR_BITS)
    )
    scale = payoff[-1]
    angles = tuple(
        2 * math.acos(math.exp(-rate * (1 << bit) / (2 * fraction_scale)))
        for bit in range(SELECTOR_BITS)
    )
    lower_angle = 2 * math.asin(math.sqrt(STRIKE / scale))
    exact_price = sum(
        probabilities[word % 4] * probabilities[word // 4] * (payoff[index] - STRIKE)
        for word, index in enumerate(selectors)
    )
    return Model(
        edges[:-1],
        probabilities,
        coefficients,
        offset,
        threshold,
        tuple(selectors),
        rate,
        angles,
        lower_angle,
        scale,
        payoff,
        exact_price,
    )
