# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 ClassicalSimState._update_assign_from_vals and _get_in_vals.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/simulation/classical_sim.py

"""Bind integer-register values without graph objects or compiler dependencies."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from itertools import product
from typing import TYPE_CHECKING

import numpy as np

from .classical_distributions import ClassicalDistribution
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
    resolve_distribution: Callable[[ClassicalDistribution], object] | None = None,
) -> dict[tuple[int, ...], object]:
    """Bind one named register's row-major indices to original classical values.

    Shapes must be concrete. Scalar symbolic expressions bypass domain checks,
    as in the source. Arrays use asanyarray and source range-only validation,
    preserving subclasses and element types. Extra named values are ignored.
    Scalar distributions are resolved exactly once before integer validation.
    Their resolver owns the RNG or fixed instance identity. Without a resolver,
    stochastic values are rejected. Shaped values never invoke the resolver.
    No graph traversal or quantum allocation occurs.
    """
    try:
        value = values[name]
    except KeyError as exc:
        raise ValueError(f"Classical input requires register {name!r}") from exc
    _validate_shape(shape)
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
    if isinstance(value, ClassicalDistribution):
        if resolve_distribution is None:
            raise ValueError(
                f"Classical register {name!r} has non-deterministic action."
                "Cannot simulate with classical values."
            )
        value = resolve_distribution(value)
    validate_integer_word_value(value, width, signed=signed)
    return {(): value}


def _validate_shape(shape: tuple[int, ...]) -> None:
    if not isinstance(shape, tuple) or any(
        not isinstance(dimension, (int, np.integer)) or dimension < 0
        for dimension in shape
    ):
        raise ValueError("Classical register shapes must be concrete dimension tuples")


def collect_integer_register_value(
    assignments: Mapping[tuple[int, ...], object],
    width: int | Expr,
    *,
    shape: tuple[int, ...] = (),
    signed: bool = False,
) -> object:
    """Collect indexed values using the source integer-array storage policy.

    Scalar values are returned unchanged. Arrays use the smallest 8/16/32/64-bit
    signed or unsigned NumPy storage containing the declared word width, with
    object storage above 64 bits. Values are assigned in row-major index order;
    missing entries and NumPy conversion errors propagate. No range validation
    is added at collection: source validation belongs to assignment time.
    This is value assembly, not graph traversal or quantum execution.
    """
    _validate_shape(shape)
    if not shape:
        return assignments[()]
    storage: np.dtype[np.generic] = np.dtype(object)
    for limit in (8, 16, 32, 64):
        if width <= limit:
            storage = np.dtype(f"{'int' if signed else 'uint'}{limit}")
            break
    array = np.empty(shape, dtype=storage)
    for idx in product(*(range(n) for n in shape)):
        array[idx] = assignments[idx]
    return array
