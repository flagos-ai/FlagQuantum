"""The one speedup confidence interval the release producers publish.

Every capability that publishes a matched-speed ratio divides a single-device
leg's latency by a sharded leg's latency and has to say how sure it is of the
result. The MPS, statevector and tensor-network producers all need that
computation, and three copies of an interval that a release gate reads as a
significance bound would be three chances for the bounds to drift apart, so the
implementation lives here and each producer imports it.

The interval is a percentile bootstrap over the ratio of the two medians. The
two legs are independent samples of the same frozen unit, so each resample draws
with replacement from each leg separately; the ratio of the resampled medians is
one plausible value of the published ratio, and the interval is the 2.5th and
97.5th percentiles of that distribution. The generator is seeded from a fixed
constant so the interval a payload reports is reproducible from the samples it
publishes, which is what makes the sealed number checkable rather than merely
stated.
"""

from __future__ import annotations

import random
from collections.abc import Sequence

BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 440044
CONFIDENCE_LEVEL = 0.95


def median(values: Sequence[float]) -> float:
    """Return the median of a sample, without depending on the sample's order."""

    ordered = sorted(float(item) for item in values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return 0.5 * (ordered[middle - 1] + ordered[middle])


def bootstrap_ratio_interval(
    baseline: Sequence[float],
    sharded: Sequence[float],
    *,
    confidence_level: float = CONFIDENCE_LEVEL,
) -> tuple[float, float]:
    """Percentile interval for the ratio of two independent latency samples.

    The interval is reported as ``(lower, upper)`` on the ratio
    ``median(baseline) / median(sharded)``, which is the speedup a payload
    publishes. An empty or entirely zero sharded sample yields
    ``(-inf, -inf)`` rather than a fabricated bound: a ratio with no denominator
    has no interval, and a caller that treats the return value as a lower bound
    must see it fail the comparison rather than pass it.
    """

    if not 0.0 < confidence_level < 1.0:
        raise ValueError(
            "a confidence interval needs a level strictly inside the unit "
            f"interval, not {confidence_level}"
        )
    if not baseline or not sharded:
        return (float("-inf"), float("-inf"))
    generator = random.Random(BOOTSTRAP_SEED)
    ratios: list[float] = []
    for _ in range(BOOTSTRAP_RESAMPLES):
        left = median([generator.choice(baseline) for _ in baseline])
        right = median([generator.choice(sharded) for _ in sharded])
        if right > 0:
            ratios.append(left / right)
    if not ratios:
        return (float("-inf"), float("-inf"))
    ratios.sort()
    tail = 0.5 * (1.0 - confidence_level)
    last = len(ratios) - 1
    return (ratios[int(tail * last)], ratios[int((1.0 - tail) * last)])
