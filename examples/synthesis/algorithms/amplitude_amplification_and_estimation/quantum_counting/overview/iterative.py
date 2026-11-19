"""Staged bounded iterative amplitude estimate from sampled Grover indicators."""

import math
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class Iteration:
    k: int
    shots: int
    successes: int
    theta_interval: tuple[float, float]


@dataclass(frozen=True)
class Estimate:
    estimate: float
    confidence_interval: tuple[float, float]
    iterations: tuple[Iteration, ...]


def estimate(
    sample: Callable[[int, int], int],
    epsilon: float = 1 / 32,
    alpha: float = 0.01,
    *,
    max_steps: int = 64,
    max_k: int = 16,
    batch: int = 256,
) -> Estimate:
    """Invert sampled sin²((2k+1)theta) on a checked monotone quadrant.

    Union-bound Hoeffding intervals cover all bounded factors and sample sizes.
    The sampler returns counts only, never an exact amplitude to the estimator.
    This is a finite example-local profile, not the Classiq IQAE implementation.
    """
    if (
        type(epsilon) not in (int, float)
        or not math.isfinite(epsilon)
        or not 0 < epsilon < 0.5
        or type(alpha) not in (int, float)
        or not math.isfinite(alpha)
        or not 0 < alpha < 1
        or type(max_steps) is not int
        or not 1 <= max_steps <= 64
        or type(max_k) is not int
        or not 0 <= max_k <= 16
        or type(batch) is not int
        or not 1 <= batch <= 2048
    ):
        raise ValueError(
            "Use finite precision/confidence values and bounded sample budgets."
        )
    lower, upper = 0.0, math.pi / 2
    totals: dict[int, tuple[int, int]] = {}
    history: list[Iteration] = []
    for _ in range(max_steps):
        multiplier, quadrant = 1, 0
        for candidate in range(2 * max_k + 1, 0, -2):
            segment = math.floor(candidate * lower / (math.pi / 2))
            if candidate * upper <= (segment + 1) * math.pi / 2:
                multiplier, quadrant = candidate, segment
                break
        k = (multiplier - 1) // 2
        successes = sample(k, batch)
        if type(successes) is not int or not 0 <= successes <= batch:
            raise ValueError(
                "Sampler must return an integer success count within the batch."
            )
        old_successes, old_shots = totals.get(k, (0, 0))
        count, shots = old_successes + successes, old_shots + batch
        totals[k] = count, shots
        delta = math.sqrt(math.log(2 * (max_k + 1) * max_steps / alpha) / (2 * shots))
        lo = math.asin(math.sqrt(max(0.0, count / shots - delta)))
        hi = math.asin(math.sqrt(min(1.0, count / shots + delta)))
        if quadrant % 2:
            offset = ((quadrant + 1) // 2) * math.pi
            lo, hi = offset - hi, offset - lo
        else:
            offset = (quadrant // 2) * math.pi
            lo, hi = lo + offset, hi + offset
        lower, upper = max(lower, lo / multiplier), min(upper, hi / multiplier)
        if lower > upper:
            raise ValueError(
                "Finite confidence intervals conflict; no repaired answer."
            )
        history.append(Iteration(k, batch, successes, (lower, upper)))
        interval = math.sin(lower) ** 2, math.sin(upper) ** 2
        if interval[1] - interval[0] <= 2 * epsilon:
            return Estimate(sum(interval) / 2, interval, tuple(history))
    raise ValueError(
        "Bounded iterative estimate exhausted its sample/iteration budget."
    )
