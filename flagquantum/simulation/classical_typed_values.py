# Copyright 2023 Google LLC
# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Composed from Qualtran v0.7.0 ClassicalSimState and dtype validation methods.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/simulation/classical_sim.py

"""Bind typed numerical declarations through the existing value owners."""

from collections.abc import Callable, Mapping
from functools import partial
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..core.ir.program.registers import Register
from .bit_encoding import validate_bit_array, validate_bit_value
from .bounded_integer_values import (
    validate_bounded_integer_array,
    validate_bounded_integer_value,
)
from .classical_distributions import ClassicalDistribution
from .classical_field_registers import (
    bind_field_register_values,
    collect_field_register_value,
)
from .classical_register_metadata import concrete_register_metadata
from .classical_registers import (
    bind_classical_register_values,
    bind_integer_register_values,
    bind_opaque_register_values,
    collect_classical_register_value,
    collect_integer_register_value,
)
from .finite_field_encoding import FiniteFieldEncoding
from .finite_field_metadata import resolve_finite_field_encoding
from .fixed_point_encoding import validate_fixed_point_array, validate_fixed_point_value
from .ones_complement_encoding import (
    validate_ones_complement_array,
    validate_ones_complement_value,
)


def bind_typed_register_values(
    register: Register,
    values: Mapping[str, object],
    *,
    field_encoding: FiniteFieldEncoding | None = None,
    resolve_distribution: Callable[[ClassicalDistribution], object] | None = None,
) -> dict[tuple[int, ...], object]:
    """Dispatch concrete dtype validation; shared binding owns shape and sampling.

    This is numerical metadata composition, not executable IR admission. A field
    override is already resolved by the caller, which owns metadata matching.
    Scalar Expr bypass and post-distribution validation stay in the shared owner.
    """
    kind = register.dtype.encoding
    if kind == "finite_field":
        encoding = field_encoding or resolve_finite_field_encoding(register.dtype)
        return bind_field_register_values(
            register.name,
            values,
            encoding,
            shape=register.shape,
            resolve_distribution=resolve_distribution,
        )
    width, parameters = concrete_register_metadata(register)
    if kind == "opaque":
        return bind_opaque_register_values(
            register.name,
            values,
            shape=register.shape,
            resolve_distribution=resolve_distribution,
        )
    if kind in {"signed", "montgomery"} or (
        kind == "uint" and "bound" not in parameters
    ):
        return bind_integer_register_values(
            register.name,
            values,
            width,
            shape=register.shape,
            signed=kind == "signed",
            resolve_distribution=resolve_distribution,
        )
    scalar: Callable[[object], None]
    array: Callable[[NDArray[Any]], None]
    if kind == "bit":
        scalar, array = validate_bit_value, validate_bit_array
    elif kind == "uint" and "bound" in parameters:
        scalar = partial(validate_bounded_integer_value, bound=parameters["bound"])
        array = partial(validate_bounded_integer_array, bound=parameters["bound"])
    elif kind == "ones_complement":
        scalar = partial(validate_ones_complement_value, width=width)
        array = partial(validate_ones_complement_array, width=width)
    elif kind == "fixed_point":
        scalar = partial(
            validate_fixed_point_value,
            width=width,
            fractional_width=parameters["fractional_width"],
            signed=bool(parameters["signed"]),
        )
        array = partial(
            validate_fixed_point_array,
            width=width,
            fractional_width=parameters["fractional_width"],
            signed=bool(parameters["signed"]),
        )
    else:
        raise ValueError(f"no native classical value binding for {kind!r}")
    return bind_classical_register_values(
        register.name,
        values,
        scalar,
        array,
        shape=register.shape,
        resolve_distribution=resolve_distribution,
    )


def collect_typed_register_value(
    register: Register,
    assignments: Mapping[tuple[int, ...], object],
    *,
    field_encoding: FiniteFieldEncoding | None = None,
) -> object:
    """Collect with source quantum-only integer/field storage special cases.

    Classical integer/field, bounded, fixed-point, Montgomery and ones-complement
    arrays use object storage. Scalars remain unchanged and are not revalidated.
    """
    kind = register.dtype.encoding
    classical = ("storage", "classical") in register.dtype.parameters
    if kind == "finite_field":
        if classical:
            return collect_classical_register_value(
                assignments, register.shape, lambda shape: np.empty(shape, dtype=object)
            )
        encoding = field_encoding or resolve_finite_field_encoding(register.dtype)
        return collect_field_register_value(assignments, encoding, shape=register.shape)
    width, parameters = concrete_register_metadata(register)
    if kind not in {
        "opaque",
        "bit",
        "uint",
        "signed",
        "montgomery",
        "ones_complement",
        "fixed_point",
    }:
        raise ValueError(f"no native classical value collection for {kind!r}")
    if kind == "bit" or (
        not classical
        and (
            kind in {"opaque", "signed"}
            or (kind == "uint" and "bound" not in parameters)
        )
    ):
        return collect_integer_register_value(
            assignments, width, shape=register.shape, signed=kind == "signed"
        )
    return collect_classical_register_value(
        assignments, register.shape, lambda shape: np.empty(shape, dtype=object)
    )
