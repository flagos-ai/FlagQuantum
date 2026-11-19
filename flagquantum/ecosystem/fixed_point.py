# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 _Fxp real conversion methods.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/_infra/data_types.py

"""Optional real fixed-point objects at the external conversion boundary."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from importlib import import_module
from typing import TYPE_CHECKING, Any

from ..core.value_parameters import is_symbolic_value
from ..simulation.fixed_point_encoding import (
    fixed_point_domain,
    validate_fixed_point_shape,
)
from ..simulation.integer_encoding import integer_from_bits

if TYPE_CHECKING:
    from sympy import Expr


def fixed_point_template(
    width: int | Expr, fractional_width: int | Expr, *, signed: bool = False
) -> Any:
    """Construct the source external template with same-width wrapping options."""
    validate_fixed_point_shape(width, fractional_width, signed=signed)
    fxpmath = import_module("fxpmath")

    if is_symbolic_value(width) or is_symbolic_value(fractional_width):
        raise ValueError(
            "Cannot construct fixed-point template for symbolic widths "
            f"{width=}, {fractional_width=}"
        )
    return fxpmath.Fxp(
        None,
        n_word=width,
        n_frac=fractional_width,
        signed=signed,
        op_sizing="same",
        const_op_sizing="same",
        shifting="trunc",
        overflow="wrap",
    )


def fixed_point_real_domain(
    width: int | Expr, fractional_width: int | Expr, *, signed: bool = False
) -> Iterator[Any]:
    """Yield external real values in the existing raw integer domain order."""
    domain = fixed_point_domain(width, fractional_width, signed=signed)

    def values() -> Iterator[Any]:
        fxpmath = import_module("fxpmath")

        for word in domain:
            yield fxpmath.Fxp(
                word / 2**fractional_width,
                like=fixed_point_template(width, fractional_width, signed=signed),
            )

    return values()


def validate_fixed_point_real(
    value: Any,
    width: int | Expr,
    fractional_width: int | Expr,
    *,
    signed: bool = False,
    debug_str: str = "val",
) -> None:
    """Apply the source exactness comparison, including external input mutation."""
    validate_fixed_point_shape(width, fractional_width, signed=signed)
    fxpmath = import_module("fxpmath")

    fixed_value = value if isinstance(value, fxpmath.Fxp) else fxpmath.Fxp(value)
    if (
        fixed_value.get_val()
        != fixed_value.like(
            fixed_point_template(width, fractional_width, signed=signed)
        ).get_val()
    ):
        raise ValueError(
            f"{debug_str}={value} cannot be accurately represented using Fxp {fixed_value}"
        )


def fixed_point_real_to_bits(
    value: Any,
    width: int | Expr,
    fractional_width: int | Expr,
    *,
    signed: bool = False,
    require_exact: bool = True,
    complement: bool = True,
) -> list[int]:
    """Convert a real value through the frozen external quantization call chain."""
    validate_fixed_point_shape(width, fractional_width, signed=signed)
    fxpmath = import_module("fxpmath")

    if require_exact:
        validate_fixed_point_real(value, width, fractional_width, signed=signed)
    if value < 0 and not signed:
        raise ValueError(f"unsigned fixed-point encoding cannot represent {value}.")
    if signed and not complement:
        sign = int(value < 0)
        value = abs(value)
    fixed_value = value if isinstance(value, fxpmath.Fxp) else fxpmath.Fxp(value)
    bits = [
        int(bit)
        for bit in fixed_value.like(
            fixed_point_template(width, fractional_width, signed=signed)
        ).bin()
    ]
    if signed and not complement:
        bits[0] = sign
    return bits


def fixed_point_real_to_integer(
    value: Any,
    width: int | Expr,
    fractional_width: int | Expr,
    *,
    signed: bool = False,
    require_exact: bool = False,
    complement: bool = True,
) -> int:
    """Quantize then decode with the existing same-width integer codec."""
    bits = fixed_point_real_to_bits(
        value,
        width,
        fractional_width,
        signed=signed,
        require_exact=require_exact,
        complement=complement,
    )
    return integer_from_bits(bits, width, signed=signed)


def fixed_point_bits_to_real(
    bits: Sequence[int],
    width: int | Expr,
    fractional_width: int | Expr,
    *,
    signed: bool = False,
) -> Any:
    """Reconstruct the external object with the source's exact string slicing."""
    validate_fixed_point_shape(width, fractional_width, signed=signed)
    fxpmath = import_module("fxpmath")

    if is_symbolic_value(fractional_width):
        raise ValueError(f"Symbolic {fractional_width} cannot be represented using Fxp")
    binary = "".join(str(bit) for bit in bits[:])
    fixed_binary = (
        "0b" + binary[: -int(fractional_width)] + "." + binary[-int(fractional_width) :]
    )
    return fxpmath.Fxp(
        fixed_binary,
        like=fixed_point_template(width, fractional_width, signed=signed),
    )
