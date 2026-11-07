# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 ClassicalSimState._update_assign_from_vals.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/simulation/classical_sim.py

"""Bind integer-register values without graph objects or compiler dependencies."""

from __future__ import annotations

from collections.abc import Mapping
from itertools import product
from typing import TYPE_CHECKING

import numpy as np

from .integer_values import validate_integer_word_array, validate_integer_word_value

if TYPE_CHECKING:
    from sympy import Expr


def bind_integer_register_values(
    name: str,
    values: Mapping[str, object],
    width: int | Expr,
    *,
    shape: tuple[int, ...] = (),
    signed: bool = False,
) -> dict[tuple[int, ...], object]:
    """Bind one named register's row-major indices to original classical values.

    Shapes must be concrete. Scalar symbolic expressions bypass domain checks,
    as in the source. Arrays use asanyarray and source range-only validation,
    preserving subclasses and element types. Extra named values are ignored.
    No graph traversal, stochastic-value sampling or quantum allocation occurs.
    """
    try:
        value = values[name]
    except KeyError as exc:
        raise ValueError(f"Classical input requires register {name!r}") from exc
    if not isinstance(shape, tuple) or any(
        not isinstance(dimension, (int, np.integer)) or dimension < 0
        for dimension in shape
    ):
        raise ValueError("Classical register shapes must be concrete dimension tuples")
    if shape:
        array = np.asanyarray(value)
        if array.shape != shape:
            raise ValueError(f"Classical register {name!r} shape differs from {shape}")
        validate_integer_word_array(array, width, signed=signed)
        return {idx: array[idx] for idx in product(*(range(n) for n in shape))}
    if not isinstance(value, (int, np.integer)):
        try:
            from sympy import Expr
        except ImportError:
            # A missing optional symbolic package cannot make a noninteger
            # scalar valid. The scalar validator below still fails closed.
            pass
        else:
            if isinstance(value, Expr):
                return {(): value}
    validate_integer_word_value(value, width, signed=signed)
    return {(): value}
