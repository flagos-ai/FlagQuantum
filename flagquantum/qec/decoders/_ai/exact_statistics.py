"""Conservatively bracketed exact-binomial intervals for fixed QEC studies.

Decimal directed rounding encloses binomial tails; inversion returns outward
brackets, never an asymptotic interval. No optional numerical package is needed.
The certified scope is at most 2**22 shots and marginal alpha >= 1e-12.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import MAX_EMAX, MIN_EMIN, ROUND_CEILING, ROUND_FLOOR, Context, Decimal
from typing import Literal

from flagquantum.qec.decoders._ai.assessment import PairedCounts, paired_p_value

_MAX_SHOTS = 2**22
_WIDTH = 2.0**-44
_ONE = Decimal(1)


def _context(rounding: str) -> Context:
    return Context(prec=128, rounding=rounding, Emin=MIN_EMIN, Emax=MAX_EMAX)


def _power(value: Decimal, exponent: int, context: Context) -> Decimal:
    result = _ONE
    while exponent:
        if exponent & 1:
            result = context.multiply(result, value)
        exponent >>= 1
        if exponent:
            value = context.multiply(value, value)
    return result


def _cdf_bounds(k: int, n: int, p: float) -> tuple[Decimal, Decimal]:
    """Enclose the exact binomial CDF using positive-term recurrences."""
    if p == 0:
        return _ONE, _ONE
    if p == 1:
        return Decimal(0), Decimal(0)
    low, high = _context(ROUND_FLOOR), _context(ROUND_CEILING)
    probability = Decimal.from_float(p)
    complement = low.subtract(_ONE, probability)
    # All queried floats have <=97 fractional binary digits, so this subtraction
    # is exact at precision128. Refuse rather than assume that for other inputs.
    if complement != high.subtract(_ONE, probability):
        raise ArithmeticError("probability complement cannot be represented exactly")
    reflected = k > n - k - 1
    if reflected:
        k, probability, complement = n - k - 1, complement, probability
    ratio_low = low.divide(probability, complement)
    ratio_high = high.divide(probability, complement)
    term_low = _power(complement, n, low)
    term_high = _power(complement, n, high)
    sum_low, sum_high = term_low, term_high
    for j in range(k):
        numerator, denominator = Decimal(n - j), Decimal(j + 1)
        term_low = low.multiply(
            low.divide(low.multiply(term_low, numerator), denominator), ratio_low
        )
        term_high = high.multiply(
            high.divide(high.multiply(term_high, numerator), denominator), ratio_high
        )
        sum_low = low.add(sum_low, term_low)
        sum_high = high.add(sum_high, term_high)
    if reflected:
        sum_low, sum_high = low.subtract(_ONE, sum_high), high.subtract(_ONE, sum_low)
    if not (sum_low.is_finite() and sum_high.is_finite() and sum_low <= sum_high):
        raise ArithmeticError("binomial tail enclosure is nonfinite or unordered")
    return max(Decimal(0), sum_low), min(_ONE, sum_high)


def _upper_limit(k: int, n: int, tail: Decimal) -> float:
    left, right = 0.0, 1.0
    for _ in range(64):
        if right - left <= _WIDTH:
            return math.nextafter(right, 1.0)
        midpoint = (left + right) / 2
        lower, upper = _cdf_bounds(k, n, midpoint)
        if lower > tail:
            left = midpoint
        elif upper < tail:
            right = midpoint
        else:
            # Exact dyadic roots may straddle the target even at high precision.
            # Certify a small enclosing neighborhood rather than guess its side.
            a = max(left, math.nextafter(midpoint - _WIDTH / 4, 0.0))
            b = min(right, math.nextafter(midpoint + _WIDTH / 4, 1.0))
            if _cdf_bounds(k, n, a)[0] >= tail and _cdf_bounds(k, n, b)[1] <= tail:
                return math.nextafter(b, 1.0)
            raise ArithmeticError("binomial root side could not be certified")
    raise ArithmeticError("binomial inversion exceeded its iteration budget")


def exact_rate_interval(failures: int, shots: int, alpha: float) -> tuple[float, float]:
    """Equal-tailed Clopper-Pearson interval with outward floating-point endpoints.

    Root brackets have width <=2**-44 before outward float rounding.
    Supports integer shots in [1, 2**22] and finite marginal alpha in [1e-12, 1).
    Out-of-scope inputs or uncertified numerical enclosures fail closed. Tail
    summation costs O(min(failures, shots-failures)+1) per inversion iteration.
    """
    if (
        isinstance(shots, bool)
        or not isinstance(shots, int)
        or not 1 <= shots <= _MAX_SHOTS
    ):
        raise ValueError("exact interval requires integer shots in [1, 2**22]")
    if (
        isinstance(failures, bool)
        or not isinstance(failures, int)
        or not 0 <= failures <= shots
    ):
        raise ValueError("failures must be an integer between zero and shots")
    if isinstance(alpha, bool) or not math.isfinite(alpha) or not 1e-12 <= alpha < 1:
        raise ValueError("exact interval requires finite alpha in [1e-12, 1)")
    tail = _context(ROUND_FLOOR).divide(Decimal.from_float(alpha), Decimal(2))
    upper = 1.0 if failures == shots else _upper_limit(failures, shots, tail)
    lower = (
        0.0
        if failures == 0
        else max(
            0.0, math.nextafter(1.0 - _upper_limit(shots - failures, shots, tail), 0.0)
        )
    )
    return lower, upper


@dataclass(frozen=True)
class ExactSummary:
    interval_method: str
    comparison_alpha: float
    baseline_ler: float
    hybrid_ler: float
    baseline_interval: tuple[float, float]
    hybrid_interval: tuple[float, float]
    ratio_upper: float | None
    paired_p_value: float
    quality_gate: Literal["supported_10_percent_reduction", "inconclusive"]


def summarize_exact(counts: PairedCounts, *, alpha: float) -> ExactSummary:
    """Use comparison alpha; a protocol owns any family-wise allocation."""
    if isinstance(alpha, bool) or not math.isfinite(alpha) or not 2e-12 <= alpha < 1:
        raise ValueError("comparison alpha must be finite and in [2e-12, 1)")
    values = (
        counts.shots,
        counts.baseline_failures,
        counts.hybrid_failures,
        counts.wins,
        counts.losses,
    )
    if any(isinstance(v, bool) or not isinstance(v, int) for v in values):
        raise ValueError("paired counts must be integers")
    n, first, second, wins, losses = values
    if not (
        1 <= n <= _MAX_SHOTS
        and 0 <= first <= n
        and 0 <= second <= n
        and 0 <= wins <= first
        and 0 <= losses <= second
        and first - wins == second - losses
        and first + losses <= n
    ):
        raise ValueError(
            "paired counts are inconsistent or outside the certified shot budget"
        )
    baseline = exact_rate_interval(first, n, alpha / 2)
    hybrid = exact_rate_interval(second, n, alpha / 2)
    upper = (
        math.nextafter(hybrid[1] / baseline[0], math.inf) if baseline[0] > 0 else None
    )
    return ExactSummary(
        "clopper_pearson_outward_binomial_tail",
        alpha,
        first / n,
        second / n,
        baseline,
        hybrid,
        upper,
        paired_p_value(wins, losses),
        (
            "supported_10_percent_reduction"
            if upper is not None and upper <= 0.9
            else "inconclusive"
        ),
    )
