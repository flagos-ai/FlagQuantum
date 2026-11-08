# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 _BUInt, BQUInt and BCUInt value methods.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/_infra/data_types.py

"""Bounded unsigned value domains and validation without register descriptors."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from ..core.value_parameters import is_symbolic_value

if TYPE_CHECKING:
    from sympy import Expr


def bounded_integer_default_bound(width: int | Expr) -> int | Expr:
    """Return the source default exponentiation without normalizing width."""
    return 2**width


def validate_bounded_integer_shape(width: int | Expr, bound: int | Expr) -> None:
    """Retain symbolic short circuits and only the source upper-capacity check."""
    if is_symbolic_value(width) or is_symbolic_value(bound):
        return
    if bound > 2**width:
        raise ValueError(
            "BUInt value bound is too large for given bitsize. "
            f"{bound} vs {2**width}"
        )


def bounded_integer_domain(width: int | Expr, bound: int | Expr) -> range:
    """Return the original domain only for Python-int bounds, including bool."""
    validate_bounded_integer_shape(width, bound)
    if isinstance(bound, int):
        return range(bound)
    raise ValueError("Classical domain not defined for a non-Python-integer bound")


def validate_bounded_integer_value(value: object, bound: int | Expr) -> None:
    """Check integer type, then negativity, then the exclusive bound in source order."""
    if not isinstance(value, (int, np.integer)):
        raise ValueError(f"val should be an integer, not {value!r}")
    if value < 0:
        raise ValueError("Negative classical value encountered in val")
    if value >= bound:
        raise ValueError("Too-large classical value encountered in val")


def validate_bounded_integer_array(values: NDArray[Any], bound: int | Expr) -> None:
    """Retain range-only array comparisons, with no scalar integer-type guard."""
    if np.any(values < 0):
        raise ValueError("Negative classical values encountered in val")
    if np.any(values >= bound):
        raise ValueError("Too-large classical values encountered in val")
