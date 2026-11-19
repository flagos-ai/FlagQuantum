"""Financial assumptions, fixed-point grid and discounted payment rules."""

import math
from dataclasses import dataclass
from enum import Enum
from itertools import accumulate, product

NOTIONAL = 18
YEARS = 3
DAYS_PER_YEAR = 250
RATE = 0.04
ANNUAL_VOLATILITY = 0.0150694 * math.sqrt(DAYS_PER_YEAR)
ANNUAL_DRIFT = 0.00050963 * DAYS_PER_YEAR
GRID_BITS = 2
FRACTION_BITS = 2
UNIT = 1 << FRACTION_BITS
STRIKE = NOTIONAL
BARRIER = 0.7
CALL_LEVELS = (1.1, 1.1)
CALL_PAYMENTS = (2 * math.exp(-RATE), 3 * math.exp(-2 * RATE))
DISCOUNT = math.exp(-RATE * YEARS)


class Payment(Enum):
    """Mutually exclusive outcomes in priority order."""

    FIRST_CALL = "first call"
    SECOND_CALL = "second call"
    PUT = "down-and-in put"
    ZERO = "zero"


@dataclass(frozen=True)
class Grid:
    points: tuple[float, ...]
    probabilities: tuple[float, ...]
    raw_offset: int
    raw_slope: int


@dataclass(frozen=True)
class Encoding:
    return_bound: float
    integer_bits: int
    denominator: float
    numerator: float

    @property
    def width(self) -> int:
        return self.integer_bits + FRACTION_BITS

    @property
    def normalization(self) -> float:
        return NOTIONAL * self.numerator / self.denominator

    def decode(self, probability: float) -> float:
        return DISCOUNT * (
            probability * self.normalization + NOTIONAL / self.denominator - STRIKE
        )

    def amplitude_probability(self, discounted_payment: float) -> float:
        return (
            (discounted_payment / DISCOUNT + STRIKE) * self.denominator / NOTIONAL - 1
        ) / self.numerator

    def rotation(self, discounted_payment: float) -> float:
        return 2 * math.asin(math.sqrt(self.amplitude_probability(discounted_payment)))

    def reference_probabilities(self) -> tuple[float, ...]:
        weights = tuple(math.exp(word / UNIT) for word in range(1 << self.width))
        total = math.fsum(weights)
        return tuple(weight / total for weight in weights)


@dataclass(frozen=True)
class Path:
    words: tuple[int, ...]
    log_returns: tuple[int, ...]
    probability: float
    payment: Payment
    discounted_payoff: float


def gaussian_grid() -> Grid:
    """Integrate the four truncated normal bin masses and retain left endpoints."""
    size = 1 << GRID_BITS
    edges = tuple(-3 + 6 * edge / size for edge in range(size + 1))
    cdf = tuple((1 + math.erf(edge / math.sqrt(2))) / 2 for edge in edges)
    total = cdf[-1] - cdf[0]
    return Grid(
        edges[:-1],
        tuple(
            (right - left) / total
            for left, right in zip(cdf[:-1], cdf[1:], strict=True)
        ),
        math.trunc(UNIT * (ANNUAL_DRIFT + ANNUAL_VOLATILITY * edges[0])),
        math.trunc(UNIT * ANNUAL_VOLATILITY * (edges[1] - edges[0])),
    )


def encoding(grid: Grid) -> Encoding:
    """Derive the signed width and exponential normalization from source bounds."""
    extremal_return = max(
        abs(YEARS * (ANNUAL_DRIFT + ANNUAL_VOLATILITY * point))
        for point in (grid.points[0], grid.points[-1])
    )
    call_bound = math.log(
        max(STRIKE, *(STRIKE + price / DISCOUNT for price in CALL_PAYMENTS)) / NOTIONAL
    )
    bound = max(extremal_return, call_bound)
    integer_bits = 1 if bound < 1 else math.ceil(math.log2(math.ceil(bound))) + 1
    width = integer_bits + FRACTION_BITS
    return Encoding(
        bound,
        integer_bits,
        math.exp(((1 << (width - 1)) + 1) / UNIT),
        math.expm1((1 << width) / UNIT),
    )


def comparisons(grid: Grid, transform: Encoding) -> tuple[int, ...]:
    """Translate signed fixed-point comparisons into the reused unsigned kernels."""
    if grid.raw_slope != 1:
        raise ValueError("The finite preparation requires a unit raw grid slope")
    offsets = tuple(period * grid.raw_offset for period in (1, 2, 3))
    calls = tuple(math.trunc(UNIT * math.log(level)) for level in CALL_LEVELS)
    barrier = math.trunc(UNIT * math.log(BARRIER))
    strike = math.trunc(UNIT * math.log(STRIKE / NOTIONAL))
    return (
        calls[0] - offsets[0],
        calls[1] - offsets[1],
        strike - offsets[2],
        *(barrier - offset for offset in offsets),
        (1 << (transform.width - 1)) + offsets[2],
    )


def paths(grid: Grid) -> tuple[Path, ...]:
    """Enumerate the independent classical payoff table for all 64 paths."""
    barrier = math.trunc(UNIT * math.log(BARRIER))
    calls = tuple(math.trunc(UNIT * math.log(level)) for level in CALL_LEVELS)
    strike = math.trunc(UNIT * math.log(STRIKE / NOTIONAL))
    rows = []
    for words in product(range(1 << GRID_BITS), repeat=YEARS):
        returns = tuple(
            accumulate(grid.raw_offset + grid.raw_slope * word for word in words)
        )
        if returns[0] > calls[0]:
            outcome, payoff = Payment.FIRST_CALL, CALL_PAYMENTS[0]
        elif returns[1] > calls[1]:
            outcome, payoff = Payment.SECOND_CALL, CALL_PAYMENTS[1]
        elif min(returns) < barrier and returns[-1] < strike:
            outcome = Payment.PUT
            payoff = (
                NOTIONAL * (math.exp(returns[-1] / UNIT) - STRIKE / NOTIONAL) * DISCOUNT
            )
        else:
            outcome, payoff = Payment.ZERO, 0.0
        probability = math.prod(grid.probabilities[word] for word in words)
        rows.append(Path(words, returns, probability, outcome, payoff))
    return tuple(rows)
