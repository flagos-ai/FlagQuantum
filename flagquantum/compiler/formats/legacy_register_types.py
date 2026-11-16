# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Wire layout adapted from Qualtran v0.7.0 protos/data_types.proto.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/protos/data_types.proto

"""Standalone wire interchange for the eight legacy quantum register types."""

from collections.abc import Callable
from functools import cache
from typing import Any, cast

from ...core.ir.program import register_types as types
from ...core.ir.program.registers import RegisterType
from ...core.value_math import SymbolicInt
from ...errors import SerializationError
from ._legacy_protobuf import message_factory, parse_message
from .legacy_arguments import (
    _integer_message,
    _read_integer_argument,
    _write_integer_argument,
)

_SIMPLE_FACTORIES: dict[str, Callable[[SymbolicInt], RegisterType]] = {
    "qany": types.opaque_register_type,
    "qint": types.signed_register_type,
    "qint_ones_comp": types.ones_complement_register_type,
    "quint": types.unsigned_register_type,
    "qmontgomery_uint": types.montgomery_register_type,
}
_ENCODING_ARMS = {
    "bit": "qbit",
    "opaque": "qany",
    "signed": "qint",
    "ones_complement": "qint_ones_comp",
    "uint": "quint",
    "fixed_point": "qfxp",
    "montgomery": "qmontgomery_uint",
}


@cache
def _register_type_message() -> Any:
    integer = _integer_message()
    width = (("bitsize", integer),)
    message_names = (
        ("qbit", "BitType", ()),
        ("qany", "OpaqueType", width),
        ("qint", "SignedType", width),
        ("qint_ones_comp", "OnesComplementType", width),
        ("quint", "UnsignedType", width),
        ("bquint", "BoundedUnsignedType", (*width, ("iteration_length", integer))),
        ("qfxp", "FixedPointType", (*width, ("num_frac", integer), ("signed", "BOOL"))),
        ("qmontgomery_uint", "MontgomeryType", width),
    )
    return message_factory(
        "RegisterEncoding",
        tuple(
            (
                arm,
                message_factory(name, fields, optional=tuple(key for key, _ in fields)),
            )
            for arm, name, fields in message_names
        ),
        oneof="val",
    )


def serialize_register_type(dtype: RegisterType) -> bytes:
    """Encode a representable quantum register declaration."""
    message = _register_type_message()()
    _write_register_type(message, dtype)
    return cast(bytes, message.SerializeToString())


def _write_register_type(message: Any, dtype: RegisterType) -> None:
    params = dict(dtype.parameters)
    arm = _ENCODING_ARMS.get(dtype.encoding)
    if dtype.classical_bits != 0 or arm is None or "storage" in params:
        raise SerializationError("Register type has no legacy quantum representation")
    if arm == "qmontgomery_uint" and "modulus" in params:
        raise SerializationError(
            "Legacy register type wire cannot retain a Montgomery modulus"
        )
    if arm == "quint" and "bound" in params:
        arm = "bquint"
    allowed = {
        "bquint": {"bound"},
        "qfxp": {"fractional_width", "signed"},
    }.get(arm, set())
    if params.keys() - allowed:
        raise SerializationError("Register type has unrepresentable parameters")

    child = getattr(message, arm)
    if arm == "qbit":
        if dtype.quantum_bits != 1:
            raise SerializationError("Legacy quantum bit requires width one")
        child.SetInParent()
    else:
        _write_integer_argument(child.bitsize, dtype.quantum_bits)
    if arm == "bquint":
        _write_integer_argument(
            child.iteration_length, cast(SymbolicInt, params["bound"])
        )
    if arm == "qfxp":
        if "fractional_width" not in params:
            raise SerializationError(
                "Legacy fixed-point type requires a fractional width"
            )
        _write_integer_argument(
            child.num_frac, cast(SymbolicInt, params["fractional_width"])
        )
        signed = params.get("signed", False)
        if type(signed) is not bool:
            raise SerializationError("Legacy fixed-point signed flag must be boolean")
        child.signed = signed


def deserialize_register_type(data: bytes) -> RegisterType:
    """Decode the selected alternative through the owned native factories."""
    return _read_register_type(
        parse_message(_register_type_message(), data, "Register type")
    )


def _read_register_type(message: Any) -> RegisterType:
    arm = message.WhichOneof("val")
    if arm is None:
        raise SerializationError("Register type has no selected value")
    if arm == "qbit":
        return types.quantum_bit_type()
    child = getattr(message, arm)
    try:
        width = _read_integer_argument(child.bitsize)
        if arm == "bquint":
            return types.bounded_unsigned_register_type(
                width, _read_integer_argument(child.iteration_length)
            )
        if arm == "qfxp":
            return types.fixed_point_register_type(
                width, _read_integer_argument(child.num_frac), signed=bool(child.signed)
            )
        return _SIMPLE_FACTORIES[arm](width)
    except (ValueError, TypeError) as error:
        raise SerializationError(f"Invalid legacy register type field {arm}") from error
