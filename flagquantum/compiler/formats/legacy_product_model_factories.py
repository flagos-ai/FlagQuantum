# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Constructor adaptation from Qualtran v0.7.0 arithmetic/multiplication.py.

"""Retain integer and fixed-point product declarations without inventing bodies."""

from collections.abc import Mapping
from functools import partial

from ...core.ir.program import FunctionRecord
from ...core.ir.program.register_types import (
    fixed_point_register_type,
    unsigned_register_type,
)
from ...core.ir.program.registers import Register, RegisterSide
from ...errors import SerializationError
from .legacy_component_resolution import ComponentFactory, ResolvedArgument
from .legacy_modular_addition_factories import _integer

_FIELDS = {
    "PlusEqualProduct": ("a_bitsize", "b_bitsize", "result_bitsize"),
    "Square": ("bitsize",),
    "SumOfSquares": ("bitsize", "k"),
    "Product": ("a_bitsize", "b_bitsize"),
    "ScaleIntByReal": ("r_bitsize", "i_bitsize"),
    "MultiplyTwoReals": ("bitsize",),
    "SquareRealNumber": ("bitsize",),
    "InvertRealNumber": ("bitsize", "num_frac"),
}


def _product_factory(
    kind: str,
    name: str,
    registers: tuple[Register, ...] | None,
    arguments: Mapping[str, ResolvedArgument],
) -> FunctionRecord:
    required = set(_FIELDS[kind])
    optional = (
        {"is_adjoint"}
        if kind == "PlusEqualProduct"
        else {"uncompute"} if kind == "Square" else set()
    )
    if not required <= arguments.keys() or arguments.keys() - required - optional:
        raise SerializationError(f"Invalid constructor arguments for {kind}")
    values = {key: _integer(arguments, key) for key in required}
    direction = arguments.get(next(iter(optional)), False) if optional else False
    if type(direction) not in (bool, int) or direction not in (0, 1):
        raise SerializationError("Product direction requires a boolean")
    side = RegisterSide.LEFT if direction else RegisterSide.RIGHT
    ports: tuple[Register, ...]
    if kind in {"PlusEqualProduct", "Product"}:
        a, b = values["a_bitsize"], values["b_bitsize"]
        if kind == "PlusEqualProduct":
            result = values["result_bitsize"]
            if (
                all(isinstance(value, int) for value in (a, b, result))
                and result < a + b
            ):
                raise SerializationError("Product result width must cover both inputs")
            side = RegisterSide.THRU
        else:
            # The original uses Python max, including its symbolic refusal.
            result = 2 * max(a, b)
        ports = (
            Register("a", unsigned_register_type(a)),
            Register("b", unsigned_register_type(b)),
            Register("result", unsigned_register_type(result), side=side),
        )
    elif kind == "ScaleIntByReal":
        real, integer = values["r_bitsize"], values["i_bitsize"]
        ports = (
            Register("real_in", fixed_point_register_type(real, real)),
            Register("int_in", unsigned_register_type(integer)),
            Register(
                "result",
                fixed_point_register_type(real, real - integer),
                side=RegisterSide.RIGHT,
            ),
        )
    else:
        width = values["bitsize"]
        if kind == "Square":
            ports = (
                Register("a", unsigned_register_type(width)),
                Register("result", unsigned_register_type(2 * width), side=side),
            )
        elif kind == "SumOfSquares":
            count = values["k"]
            if not isinstance(count, int):
                raise SerializationError(
                    "Original sum-of-squares shape requires concrete k"
                )
            ports = (
                Register("input", unsigned_register_type(width), (count,)),
                Register(
                    "result",
                    unsigned_register_type(2 * width + (count - 1).bit_length()),
                    side=RegisterSide.RIGHT,
                ),
            )
        elif kind == "InvertRealNumber":
            fractional = values["num_frac"]
            if fractional == width:
                raise SerializationError(
                    "Inverse fractional width must be below total width"
                )
            dtype = fixed_point_register_type(width, fractional)
            ports = (Register("a", dtype), Register("result", dtype))
        else:
            if kind == "SquareRealNumber" and isinstance(width, int) and width < 3:
                raise SerializationError(
                    "SquareRealNumber requires at least three bits"
                )
            dtype = fixed_point_register_type(width, width)
            # SquareRealNumber deliberately retains the source's extra b port.
            ports = (
                Register("a", dtype),
                Register("b", dtype),
                Register("result", dtype, side=RegisterSide.RIGHT),
            )
    if registers is not None and registers != ports:
        raise SerializationError(f"Invalid register signature for {kind}")
    return FunctionRecord(name, (), registers=ports)


def product_model_factories() -> dict[str, ComponentFactory]:
    """Supply all eight source constructors at their original no-body boundary."""
    return {
        f"qualtran.bloqs.arithmetic.multiplication.{kind}": partial(
            _product_factory, kind
        )
        for kind in _FIELDS
    }
