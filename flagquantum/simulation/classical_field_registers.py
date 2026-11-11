# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 ClassicalSimState assignment and _empty_ndarray_from_reg.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/simulation/classical_sim.py

"""Bind finite-field values and collect field arrays through shared traversal."""

from collections.abc import Callable, Mapping

from .classical_distributions import ClassicalDistribution
from .classical_registers import (
    bind_classical_register_values,
    collect_classical_register_value,
)
from .finite_field_encoding import FiniteFieldEncoding


def bind_field_register_values(
    name: str,
    values: Mapping[str, object],
    encoding: FiniteFieldEncoding,
    *,
    shape: tuple[int, ...] = (),
    resolve_distribution: Callable[[ClassicalDistribution], object] | None = None,
) -> dict[tuple[int, ...], object]:
    """Use nominal scalar and range-only array validation without integer casts.

    The shared binder owns shape, Expr bypass and stochastic resolution order.
    The numerical field codec owns dtype validation; integers are not implicitly
    converted into field scalars, while ordinary arrays retain source behavior.
    """
    return bind_classical_register_values(
        name,
        values,
        lambda value: encoding.assert_valid_val(value, name),
        lambda array: encoding.assert_valid_val_array(array, name),
        shape=shape,
        resolve_distribution=resolve_distribution,
    )


def collect_field_register_value(
    assignments: Mapping[tuple[int, ...], object],
    encoding: FiniteFieldEncoding,
    *,
    shape: tuple[int, ...] = (),
) -> object:
    """Return scalars unchanged or allocate source field Zeros before writes.

    Field class resolution is lazy for scalars, including symbolic expressions.
    Arrays resolve their field class even for an empty shape. Conversion errors
    and missing indexed assignments propagate without integer coercion.
    """
    return collect_classical_register_value(
        assignments, shape, lambda dimensions: encoding.gf_type.Zeros(dimensions)
    )
