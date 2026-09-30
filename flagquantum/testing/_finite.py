"""Finiteness gate shared by the certification artifact validators."""

from __future__ import annotations

import math
from typing import Any


def require_finite(value: Any, *, error_type: type[ValueError], label: str) -> float:
    """Return a serialized artifact number, rejecting anything not finite.

    Certification artifacts travel as JSON, which carries ``NaN`` and
    ``Infinity``. Every comparison against ``NaN`` is false, so a gate written
    as ``value > limit`` treats a ``NaN`` measurement as a comfortable margin,
    and a ``NaN`` limit as a limit that nothing can exceed. Both sides of every
    numeric gate therefore pass through here first.

    Args:
        value: the serialized measurement or limit.
        error_type: the certification error the calling module raises, so each
            module keeps reporting its own contract violation.
        label: field name used to identify the rejected value.

    Raises:
        error_type: if the value is absent, not numeric, or not finite.
    """
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise error_type(f"{label} is missing or not a number: {value!r}") from exc
    if not math.isfinite(number):
        raise error_type(f"{label} is not finite: {number}")
    return number
