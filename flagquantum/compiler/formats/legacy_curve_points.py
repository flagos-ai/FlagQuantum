# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Wire layout adapted from Qualtran v0.7.0 protos/ec_point.proto.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/protos/ec_point.proto

"""Standalone legacy wire encoding for affine elliptic-curve point values."""

from collections.abc import Callable
from typing import Any, Protocol, TypeVar, cast

from ...core.value_math import SymbolicInt
from ...errors import SerializationError
from ._legacy_protobuf import message_factory, parse_message
from .legacy_arguments import (
    _integer_message,
    _read_integer_argument,
    _write_integer_argument,
)

_FIELDS = ("x", "y", "mod", "curve_a")
_Point = TypeVar("_Point")


class _AffinePoint(Protocol):
    @property
    def x(self) -> SymbolicInt: ...

    @property
    def y(self) -> SymbolicInt: ...

    @property
    def mod(self) -> SymbolicInt: ...

    @property
    def curve_a(self) -> SymbolicInt: ...


def _curve_message() -> Any:
    return message_factory(
        "AffineCurvePoint",
        tuple((name, _integer_message()) for name in _FIELDS),
        optional=("curve_a",),
    )


def serialize_curve_point(point: _AffinePoint) -> bytes:
    """Serialize coordinates and modulus, including an explicit coefficient."""
    message = _curve_message()()
    for name, value in zip(
        _FIELDS, (point.x, point.y, point.mod, point.curve_a), strict=True
    ):
        _write_integer_argument(getattr(message, name), value)
    return cast(bytes, message.SerializeToString())


def deserialize_curve_point(
    data: bytes,
    point_factory: Callable[
        [SymbolicInt, SymbolicInt, SymbolicInt, SymbolicInt], _Point
    ],
) -> _Point:
    """Decode into the caller's point type, defaulting an absent coefficient to zero."""
    message = parse_message(_curve_message(), data, "Curve point")
    values: list[SymbolicInt] = []
    for name in _FIELDS:
        if name == "curve_a" and not message.HasField(name):
            values.append(0)
            continue
        try:
            values.append(_read_integer_argument(getattr(message, name)))
        except SerializationError as error:
            raise SerializationError(f"Invalid curve point field {name}") from error
    return point_factory(values[0], values[1], values[2], values[3])
