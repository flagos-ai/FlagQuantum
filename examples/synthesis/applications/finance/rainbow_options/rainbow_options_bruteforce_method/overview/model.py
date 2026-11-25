"""Regenerate the two-asset source grid, selector, and scaled payoff."""

import math
from dataclasses import dataclass
from fractions import Fraction

STRIKE = 190.0
SOURCE_PRICE = 23.0238
EPSILON = 0.05
ALPHA = 0.1
ASSERTION_ALPHA = 1e-5


@dataclass(frozen=True)
class Model:
    grid: tuple[float, ...]
    probabilities: tuple[float, ...]
    coefficients: tuple[tuple[Fraction, ...], ...]
    offset: Fraction
    selectors: tuple[int, ...]
    payoff_grid: tuple[float, ...]
    angles: tuple[float, ...]
    scale: float
    exact_price: float


def fresh_model() -> Model:
    days = 250
    spot = (193.97, 189.12)
    means = (0.00050963 * days, 0.00062552 * days)
    variance1, covariance, variance2 = 0.000335, 0.000257, 0.000418
    first = math.sqrt(variance1)
    below = covariance / first
    chol = (
        (first * math.sqrt(days), 0.0),
        (below * math.sqrt(days), math.sqrt(variance2 - below**2) * math.sqrt(days)),
    )
    edges = (-3.0, -1.5, 0.0, 1.5, 3.0)
    cdf = tuple((1 + math.erf(edge / math.sqrt(2))) / 2 for edge in edges)
    probabilities = tuple(
        (right - left) / (cdf[-1] - cdf[0])
        for left, right in zip(cdf[:-1], cdf[1:], strict=True)
    )
    scaling = 1 / chol[0][0]
    spacing = edges[1] - edges[0]
    coefficients = tuple(
        tuple(Fraction(math.floor(2 * scaling * value), 2) for value in row)
        for row in chol
    )
    log_base = means[0] + edges[0] * sum(chol[0])
    relative_log = math.log(spot[1] / spot[0]) + means[1] - means[0]
    relative_log += edges[0] * (sum(chol[1]) - sum(chol[0]))
    offset = Fraction(math.floor(2 * scaling * relative_log / spacing), 2)
    selectors = []
    for word in range(16):
        assets = (word % 4, word // 4)
        first_value, second_value = (
            sum(c * value for c, value in zip(row, assets, strict=True))
            for row in coefficients
        )
        selector = 2 * max(first_value, second_value + offset)
        if selector.denominator != 1 or not 0 <= selector < 8:
            raise ValueError(
                "The payoff selector must fit the source three-bit register"
            )
        selectors.append(int(selector))

    payoff_grid = tuple(
        max(spot[0] * math.exp(log_base + spacing * index / (2 * scaling)), STRIKE)
        for index in range(8)
    )
    scale = payoff_grid[-1]
    angles = tuple(
        2 * math.asin(math.sqrt(payoff_grid[index] / scale)) for index in selectors
    )
    exact_price = sum(
        probabilities[word % 4]
        * probabilities[word // 4]
        * (payoff_grid[index] - STRIKE)
        for word, index in enumerate(selectors)
    )
    return Model(
        edges[:-1],
        probabilities,
        coefficients,
        offset,
        tuple(selectors),
        payoff_grid,
        angles,
        scale,
        exact_price,
    )
