# Copyright 2025 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 rotation_synthesis/_math_config.py.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/rotation_synthesis/_math_config.py

"""Numerical dispatch for rotation synthesis with the locked precision rules."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import numpy as np

from ._rotation_types import Real

_SQRT2 = math.sqrt(2)
_NUMPY_ISCLOSE = cast("Callable[[Real, Real], bool]", np.isclose)


@dataclass(frozen=True, init=False)
class MathConfig:
    """Immutable numerical operations and constants for one arithmetic backend."""

    name: str
    zero: Real
    one: Real
    pi: Real
    sqrt2: Real
    _log: Callable[[Real], Real]
    _sqrt: Callable[[Real], Real]
    _isclose: Callable[[Real, Real], bool]
    _cos: Callable[[Real], Real]
    _sin: Callable[[Real], Real]
    _floor: Callable[[Real], int]
    _ceil: Callable[[Real], int]
    _arctan2: Callable[[Real, Real], Real]
    _arcsin: Callable[[Real], Real]
    _arccos: Callable[[Real], Real]
    _number: Callable[[Real | str], Real]

    def __init__(
        self,
        name: str = "vanilla",
        zero: Real = 0.0,
        one: Real = 1.0,
        pi: Real = math.pi,
        sqrt2: Real = _SQRT2,
        log: Callable[[Real], Real] = math.log,
        sqrt: Callable[[Real], Real] = math.sqrt,
        isclose: Callable[[Real, Real], bool] = _NUMPY_ISCLOSE,
        cos: Callable[[Real], Real] = math.cos,
        sin: Callable[[Real], Real] = math.sin,
        floor: Callable[[Real], int] = math.floor,
        ceil: Callable[[Real], int] = math.ceil,
        arctan2: Callable[[Real, Real], Real] = math.atan2,
        arcsin: Callable[[Real], Real] = math.asin,
        arccos: Callable[[Real], Real] = math.acos,
        number: Callable[[Real | str], Real] = float,
    ) -> None:
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "zero", zero)
        object.__setattr__(self, "one", one)
        object.__setattr__(self, "pi", pi)
        object.__setattr__(self, "sqrt2", sqrt2)
        object.__setattr__(self, "_log", log)
        object.__setattr__(self, "_sqrt", sqrt)
        object.__setattr__(self, "_isclose", isclose)
        object.__setattr__(self, "_cos", cos)
        object.__setattr__(self, "_sin", sin)
        object.__setattr__(self, "_floor", floor)
        object.__setattr__(self, "_ceil", ceil)
        object.__setattr__(self, "_arctan2", arctan2)
        object.__setattr__(self, "_arcsin", arcsin)
        object.__setattr__(self, "_arccos", arccos)
        object.__setattr__(self, "_number", number)

    def number(self, x: Real | str) -> Real:
        return self._number(x)

    def log(self, x: Real) -> Real:
        return self._log(x)

    def sqrt(self, x: Real) -> Real:
        return self._sqrt(x)

    def isclose(self, x: Real, y: Real) -> bool:
        return self._isclose(x, y)

    def cos(self, x: Real) -> Real:
        return self._cos(x)

    def sin(self, x: Real) -> Real:
        return self._sin(x)

    def floor(self, x: Real) -> int:
        return self._floor(x)

    def ceil(self, x: Real) -> int:
        return self._ceil(x)

    def arctan2(self, y: Real, x: Real) -> Real:
        return self._arctan2(y, x)

    def arcsin(self, x: Real) -> Real:
        return self._arcsin(x)

    def arccos(self, x: Real) -> Real:
        return self._arccos(x)

    def __hash__(self) -> int:
        return hash(self.name)


NumpyConfig = MathConfig(
    "numpy",
    np.float64(0),
    np.float64(1),
    np.pi,
    np.sqrt(2, dtype=np.float64),
    np.log,
    np.sqrt,
    _NUMPY_ISCLOSE,
    np.cos,
    np.sin,
    lambda x: int(np.floor(x)),
    lambda x: int(np.ceil(x)),
    np.arctan2,
    np.arcsin,
    np.arccos,
    np.longdouble,
)


def isclose(x: Real, y: Real, eps: Real = 1e-6) -> bool:
    """Check the source's strict absolute or first-operand relative tolerance."""
    return cast(bool, abs(x - y) < eps or abs(x - y) < eps * abs(x))


def with_dps(dps: int) -> MathConfig:
    """Use the upstream process-global mpmath precision and tolerance."""
    import mpmath

    mpmath.mp.dps = dps

    def close(x: Real, y: Real) -> bool:
        return cast(bool, abs(x - y) <= mpmath.power(10, -mpmath.mp.dps // 2 + 2))

    return MathConfig(
        f"mpmath_{dps}",
        mpmath.mpf(0),
        mpmath.mpf(1),
        mpmath.pi,
        mpmath.sqrt(2),
        mpmath.log,
        mpmath.sqrt,
        close,
        mpmath.cos,
        mpmath.sin,
        lambda x: int(mpmath.floor(x)),
        lambda x: int(mpmath.ceil(x)),
        mpmath.atan2,
        mpmath.asin,
        mpmath.acos,
        mpmath.mpf,
    )
