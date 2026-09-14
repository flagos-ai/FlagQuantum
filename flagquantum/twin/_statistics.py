"""Statistical helpers shared by QPU Twin evidence objects."""

from __future__ import annotations

import math
from collections.abc import Sequence


def finite_shot_tv_radius(
    *, outcome_count: int, shots: int, confidence_level: float
) -> float:
    """Return a distribution-free multinomial total-variation radius."""

    confidence = float(confidence_level)
    if not math.isfinite(confidence) or not 0.0 < confidence < 1.0:
        raise ValueError("confidence_level must be finite and in (0, 1)")
    if outcome_count < 2 or shots <= 0:
        raise ValueError("finite-shot bounds require outcomes and positive shots")
    log_two = math.log(2.0)
    log_prefactor = outcome_count * log_two + math.log1p(
        -2.0 * math.exp(-outcome_count * log_two)
    )
    return min(
        1.0,
        math.sqrt((log_prefactor - math.log1p(-confidence)) / (2.0 * float(shots))),
    )


def total_variation(left: Sequence[float], right: Sequence[float]) -> float:
    """Return total-variation distance between equal-length vectors."""

    if len(left) != len(right):
        raise ValueError("probability vectors must have equal length")
    return 0.5 * sum(abs(float(a) - float(b)) for a, b in zip(left, right, strict=True))


__all__ = ("finite_shot_tv_radius", "total_variation")
