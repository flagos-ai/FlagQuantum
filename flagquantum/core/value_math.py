# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 symbolics.math_funcs.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/symbolics/math_funcs.py

"""Ordered neutral width arithmetic shared by metadata and resource models."""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import TYPE_CHECKING, TypeAlias, cast

from .value_parameters import is_symbolic_value

if TYPE_CHECKING:
    from sympy import Expr

SymbolicInt: TypeAlias = "int | Expr"
SymbolicReal: TypeAlias = "int | float | Expr"
SymbolicComplex: TypeAlias = "int | float | complex | Expr"


def prod(values: Iterable[SymbolicInt]) -> SymbolicInt:
    """Multiply in iteration order, starting with the original integer identity."""
    result: SymbolicInt = 1
    for value in values:
        result = result * value
    return result


def ssum(values: Iterable[SymbolicInt]) -> SymbolicInt:
    """Add in iteration order, without coercion or a numerical reduction."""
    result: SymbolicInt = 0
    for value in values:
        result = result + value
    return result


def _extreme(
    values: tuple[SymbolicReal | Iterable[SymbolicReal], ...], *, maximum: bool
) -> SymbolicReal:
    name = "smax" if maximum else "smin"
    if not values:
        raise ValueError(f"{name} expected at least 1 argument, got 0")
    items = (
        tuple(values[0])
        if len(values) == 1 and isinstance(values[0], Iterable)
        else values
    )
    if not items:
        raise ValueError(f"{name}() arg is an empty sequence")
    scalars = cast(tuple[SymbolicReal, ...], items)
    if len(scalars) == 1:
        return scalars[0]
    # Exact built-ins have no object protocol. Subclasses must still be checked.
    concrete = all(type(value) in (int, float, bool) for value in scalars)
    if not concrete and is_symbolic_value(*scalars):
        from sympy import Max, Min

        return cast("Expr", (Max if maximum else Min)(*scalars))
    return max(scalars) if maximum else min(scalars)


def smax(*values: SymbolicReal | Iterable[SymbolicReal]) -> SymbolicReal:
    """Return the source maximum, preserving singleton identity and errors."""
    return _extreme(values, maximum=True)


def smin(*values: SymbolicReal | Iterable[SymbolicReal]) -> SymbolicReal:
    """Return the source minimum, preserving singleton identity and errors."""
    return _extreme(values, maximum=False)


def ceil(value: SymbolicReal) -> SymbolicInt:
    """Round with the source scalar conversion and symbolic classification."""
    if type(value) in (int, float, bool) or not is_symbolic_value(value):
        import numpy as np

        return int(np.ceil(value))
    from sympy import ceiling

    return cast("Expr", ceiling(value))


def floor(value: SymbolicReal) -> SymbolicInt:
    """Retain the source numeric floor conversion and symbolic expression type."""
    if type(value) in (int, float, bool) or not is_symbolic_value(value):
        import numpy as np

        return int(np.floor(value))
    from sympy import floor as symbolic_floor

    return cast("Expr", symbolic_floor(value))


def log2(value: SymbolicReal) -> SymbolicReal:
    """Evaluate concrete base-two logarithms or retain the source symbolic form."""
    if type(value) in (int, float, bool) or not is_symbolic_value(value):
        return math.log2(value)
    from sympy.codegen.cfunctions import log2 as symbolic_log2

    return cast("Expr", symbolic_log2(value))


def bit_length(value: SymbolicReal) -> SymbolicInt:
    """Keep the source zero interval and ordered floor/log2/ceil calculation."""
    concrete = type(value) in (int, float, bool) or not is_symbolic_value(value)
    if concrete and 0 <= value < 1:
        return 0
    return ceil(log2(floor(value) + 1))


def pi(*values: object) -> SymbolicReal:
    """Select symbolic pi when any declared parameter is symbolic."""
    if is_symbolic_value(*values):
        from sympy import pi as symbolic_pi

        return cast("Expr", symbolic_pi)
    import numpy as np

    return np.pi


def ln(value: SymbolicReal) -> SymbolicReal:
    """Evaluate the natural logarithm with the source scalar dispatch."""
    if not is_symbolic_value(value):
        return math.log(value)
    from sympy.codegen.cfunctions import log

    return cast("Expr", log(value))


def sexp(value: SymbolicComplex) -> SymbolicComplex:
    """Preserve NumPy exponential dispatch and symbolic exact expressions."""
    if is_symbolic_value(value):
        from sympy import exp

        return cast("Expr", exp(value))
    import numpy as np

    return cast(SymbolicComplex, np.exp(value))


def sarg(value: SymbolicComplex) -> SymbolicReal:
    """Return the principal argument with the source concrete float conversion."""
    if is_symbolic_value(value):
        from sympy import arg

        return cast("Expr", arg(value))
    import numpy as np

    return float(np.angle(value))


def sabs(value: SymbolicReal) -> SymbolicReal:
    """Apply the value's absolute-value protocol without conversion."""
    return cast(SymbolicReal, abs(value))


def ssqrt(value: SymbolicReal) -> SymbolicReal:
    """Preserve exact symbolic square roots and concrete math domain errors."""
    if is_symbolic_value(value):
        from sympy import sqrt

        return cast("Expr", sqrt(value))
    return math.sqrt(value)


def acos(value: SymbolicReal) -> SymbolicReal:
    """Preserve NumPy arccos dtype, warning and symbolic dispatch."""
    if not is_symbolic_value(value):
        import numpy as np

        return cast(SymbolicReal, np.arccos(value))
    from sympy import acos as symbolic_acos

    return cast("Expr", symbolic_acos(value))


def sconj(value: SymbolicComplex) -> SymbolicComplex:
    """Keep the source symbolic or NumPy conjugation protocol."""
    if is_symbolic_value(value):
        from sympy import conjugate

        return cast("Expr", conjugate(value))
    import numpy as np

    return cast(SymbolicComplex, np.conjugate(value))


def is_zero(value: SymbolicInt) -> bool | None:
    """Return the source zero verdict, retaining an undecidable None result."""
    if is_symbolic_value(value):
        try:
            return cast("bool | None", cast("Expr", value).equals(0))
        except ValueError:
            return False
    return value == 0
