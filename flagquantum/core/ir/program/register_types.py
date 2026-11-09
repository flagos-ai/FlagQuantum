# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 QBit, QAny, QUInt and BQUInt constructors.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/_infra/data_types.py

"""Construct native register metadata using the pinned type-specific rules.

Proposal 103 prerequisite 0.3 separates constructor validation from numerical
codecs, physical wire binding and executable component graph admission.
"""

from __future__ import annotations

from enum import Enum, auto

from ...value_math import SymbolicInt as RegisterWidth
from ...value_parameters import is_symbolic_value
from .registers import RegisterType


class _DefaultBound(Enum):
    OMITTED = auto()


def quantum_bit_type() -> RegisterType:
    """Return the single-qubit encoding."""
    return RegisterType("bit", 1)


def opaque_register_type(width: RegisterWidth) -> RegisterType:
    """Preserve symbolic-first validation of an opaque quantum word."""
    if not is_symbolic_value(width) and not isinstance(width, int):
        raise ValueError(f"Bad bitsize for opaque register: {width}")
    return RegisterType("opaque", width)


def unsigned_register_type(width: RegisterWidth) -> RegisterType:
    """Retain the declared unsigned width without invoking its numerical codec."""
    return RegisterType("uint", width)


def bounded_unsigned_register_type(
    width: RegisterWidth,
    bound: RegisterWidth | _DefaultBound = _DefaultBound.OMITTED,
) -> RegisterType:
    """Construct an unsigned word with an exclusive iteration bound.

    An omitted bound is computed before symbolic classification. Numerical value
    validation and physical binding apply at their existing owning boundaries.
    """
    if bound is _DefaultBound.OMITTED:
        bound = 2**width
    if not is_symbolic_value(width, bound):
        if bound > 2**width:
            raise ValueError(
                f"Bounded unsigned register({width}, {bound}) iteration length "
                f"is too large for given bitsize. {bound} vs {2**width}"
            )
    return RegisterType("uint", width, parameters=(("bound", bound),))
