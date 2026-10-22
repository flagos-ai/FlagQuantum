"""Finite Cartesian input semantics shared by physical program consumers."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Literal


def finite_real(value: object, field: str) -> float:
    """Accept ordinary Python reals without detaching tensors or coercing IDs."""
    if type(value) not in (int, float):
        raise TypeError(f"{field} must be an ordinary real number, excluding bool")
    assert isinstance(value, (int, float))
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError(f"{field} must be finite") from exc
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def cartesian_pair(value: object, field: str) -> tuple[float, float]:
    """Validate one finite two-dimensional Cartesian point."""
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{field} must be a Cartesian pair")
    if len(value) != 2:
        raise ValueError(f"{field} must have exactly two coordinates")
    return finite_real(value[0], field), finite_real(value[1], field)


def same_time(
    left: float, right: float, *, precision: Literal["float32", "float64"] = "float64"
) -> bool:
    """Compare already-normalized boundaries at eight ULPs of their precision."""
    if not math.isfinite(left) or not math.isfinite(right):
        return False
    if precision == "float64":
        left_ulp, right_ulp = math.ulp(left), math.ulp(right)
    elif precision == "float32":

        def ulp32(value: float) -> float:
            return (
                max(2.0**-149, math.ldexp(1.0, math.frexp(abs(value))[1] - 24))
                if value
                else 2.0**-149
            )

        left_ulp, right_ulp = ulp32(left), ulp32(right)
    else:
        raise ValueError("time precision must be float32 or float64")
    return abs(left - right) <= 8 * max(left_ulp, right_ulp)
