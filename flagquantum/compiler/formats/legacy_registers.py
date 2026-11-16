# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Wire layout adapted from Qualtran v0.7.0 protos/registers.proto.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/protos/registers.proto

"""Legacy interchange for ordered, shaped register declarations."""

from collections.abc import Iterable
from typing import Any, cast

from ...core.ir.program.registers import Register, RegisterSide
from ...errors import SerializationError
from ._legacy_protobuf import message_factory, parse_message
from .legacy_arguments import (
    _integer_message,
    _read_integer_argument,
    _write_integer_argument,
)
from .legacy_register_types import (
    _read_register_type,
    _register_type_message,
    _write_register_type,
)

_SIDES = {
    1: RegisterSide.LEFT,
    2: RegisterSide.RIGHT,
    3: RegisterSide.THRU,
}


def _register_message() -> Any:
    return message_factory(
        "RegisterDeclaration",
        (
            ("name", "STRING"),
            ("dtype", _register_type_message()),
            ("shape", _integer_message()),
            ("side", "Side"),
        ),
        repeated=("shape",),
        enums=(("Side", ("UNKNOWN", "LEFT", "RIGHT", "THRU")),),
    )


def _registers_message() -> Any:
    return message_factory(
        "RegisterDeclarations",
        (("registers", _register_message()),),
        repeated=("registers",),
    )


def _write_register(message: Any, register: Register) -> None:
    side = register.side
    if side not in _SIDES.values():
        raise SerializationError(f"Unknown register side {side}")
    message.name = register.name
    _write_register_type(message.dtype, register.dtype)
    for dimension in register.shape_symbolic:
        _write_integer_argument(message.shape.add(), dimension)
    message.side = side.value


def _read_register(message: Any) -> Register:
    dtype = _read_register_type(message.dtype)
    shape = tuple(_read_integer_argument(dimension) for dimension in message.shape)
    side = _SIDES.get(message.side)
    if side is None:
        raise SerializationError(f"Unknown register side {message.side}")
    return Register(message.name, dtype, shape, side)


def serialize_register(register: Register) -> bytes:
    """Write a shaped declaration with its explicit direction."""
    message = _register_message()()
    _write_register(message, register)
    return cast(bytes, message.SerializeToString())


def deserialize_register(data: bytes) -> Register:
    """Read one declaration through the existing register owner."""
    return _read_register(parse_message(_register_message(), data, "Register"))


def serialize_registers(registers: Iterable[Register]) -> bytes:
    """Write an ordered sequence, including an empty collection."""
    message = _registers_message()()
    for register in registers:
        _write_register(message.registers.add(), register)
    return cast(bytes, message.SerializeToString())


def deserialize_registers(data: bytes) -> tuple[Register, ...]:
    """Read an ordered collection for subsequent native signature validation."""
    message = parse_message(_registers_message(), data, "Registers")
    return tuple(_read_register(register) for register in message.registers)
