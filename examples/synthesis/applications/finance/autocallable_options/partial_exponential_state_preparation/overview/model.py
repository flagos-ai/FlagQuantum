"""Fresh source interval hints and discrete exponential overlap diagnostics."""

import math
from dataclasses import dataclass

from examples.synthesis.applications.finance.autocallable_options.partial_exponential_state_preparation.main import (
    exponential_amplitudes,
)


@dataclass(frozen=True)
class IntervalCase:
    rate: float
    lower: int
    upper: int
    shift: int
    amplitudes: tuple[float, ...]
    declared_overlap: float
    observed_overlap: float
    iterations: int
    beta: float


def make_case(
    rate: float,
    lower: int,
    upper: int,
    shift: int = 0,
    *,
    supplied_overlap: float | None = None,
) -> IntervalCase:
    if (
        type(rate) not in (int, float)
        or not math.isfinite(rate)
        or not 0.01 <= abs(rate) <= 1
        or any(type(x) is not int for x in (lower, upper, shift))
        or not 0 <= lower < upper <= 31
        or not -31 <= shift <= 31
    ):
        raise ValueError("use a finite nonzero bounded rate and a five-bit interval")
    amplitudes = tuple(exponential_amplitudes(-rate, 5))
    declared = math.sqrt(
        (math.exp(rate * upper) - math.exp(rate * lower)) / (math.exp(rate * 31) - 1)
    )
    observed = math.sqrt(
        math.fsum(x * x for x in amplitudes[lower : upper + 1])
        / math.fsum(x * x for x in amplitudes)
    )
    if supplied_overlap is not None and (
        type(supplied_overlap) not in (int, float)
        or not math.isfinite(supplied_overlap)
        or abs(supplied_overlap - observed) > 1e-12
    ):
        raise ValueError("supplied overlap disagrees with the proven discrete overlap")
    iterations = max(0, math.ceil(math.pi / (4 * math.asin(observed)) - 0.5))
    if iterations > 128:
        raise ValueError("source hint exceeds the bounded128iteration profile")
    target = math.sin(math.pi / (4 * iterations + 2))
    beta = 2 * math.acos(min(1.0, target / observed))
    return IntervalCase(
        float(rate),
        lower,
        upper,
        shift,
        amplitudes,
        declared,
        observed,
        iterations,
        beta,
    )


def source_cases(rate: float = 0.5) -> tuple[IntervalCase, IntervalCase]:
    initial = make_case(rate, 15, 29)
    lower, upper, shift = (21, 31, -18) if rate > 0 else (0, 10, 3)
    return initial, make_case(rate, lower, upper, shift)
