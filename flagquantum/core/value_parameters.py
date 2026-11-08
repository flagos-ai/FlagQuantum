# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 is_symbolic.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/symbolics/types.py

"""Classify neutral metadata and numerical parameters through one object protocol."""

from collections.abc import Callable
from typing import cast


def is_symbolic_value(*values: object) -> bool:
    """Keep original variadic short circuits and single-checker result propagation."""
    if len(values) != 1:
        return any(is_symbolic_value(value) for value in values)
    (value,) = values
    try:
        from sympy import Basic
    except ModuleNotFoundError as error:
        if error.name != "sympy":
            raise
    else:
        if isinstance(value, Basic):
            return True
    checker = getattr(value, "is_symbolic", None)
    if checker is not None:
        # A typing-only protocol boundary: do not cast the returned runtime value.
        return cast(Callable[[], bool], checker)()
    return False
