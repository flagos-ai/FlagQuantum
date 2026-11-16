# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Wire fields adapted from Qualtran v0.7.0 protos/args.proto and serialization/args.py.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/protos/args.proto

"""Standalone legacy wire helpers for symbolic integers, arrays, and complex values."""

from io import BytesIO
from operator import index
from sys import modules
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from ...core.value_math import SymbolicInt
from ...errors import SerializationError
from ._legacy_expression import parse_legacy_expression
from ._legacy_protobuf import message_factory, parse_message


def _integer_message() -> Any:
    return message_factory(
        "IntegerOrExpression",
        (("int_val", "INT64"), ("sympy_expr", "STRING")),
        oneof="val",
    )


def _array_message() -> Any:
    return message_factory("ArrayArgument", (("ndarray", "BYTES"),))


def _complex_message() -> Any:
    return message_factory("ComplexArgument", (("real", "FLOAT"), ("imag", "FLOAT")))


def serialize_integer_argument(value: SymbolicInt) -> bytes:
    """Write the integer oneof arm or the admitted SymPy expression text."""
    factory = _integer_message()
    sympy = modules.get("sympy")
    if sympy is not None and isinstance(value, sympy.Expr):
        text = str(value)
        parse_legacy_expression(text)
        return cast(bytes, factory(sympy_expr=text).SerializeToString())
    try:
        return cast(bytes, factory(int_val=index(value)).SerializeToString())
    except (TypeError, ValueError, OverflowError) as error:
        raise SerializationError(
            "Integer argument requires a signed int64 or SymPy expression"
        ) from error


def deserialize_integer_argument(data: bytes) -> SymbolicInt:
    """Read the selected arm, preserving explicitly encoded integer zero."""
    message = parse_message(_integer_message(), data, "Integer argument")
    choice = message.WhichOneof("val")
    if choice == "int_val":
        return int(message.int_val)
    if choice == "sympy_expr":
        return parse_legacy_expression(message.sympy_expr)
    raise SerializationError("Integer argument has no selected value")


def serialize_array_argument(value: NDArray[Any]) -> bytes:
    """Write a NumPy .npy payload in the standalone bytes field."""
    with BytesIO() as stream:
        try:
            np.save(stream, value, allow_pickle=False)
        except (TypeError, ValueError, OverflowError) as error:
            raise SerializationError("Invalid array argument") from error
        message = _array_message()(ndarray=stream.getvalue())
    return cast(bytes, message.SerializeToString())


def deserialize_array_argument(data: bytes) -> NDArray[Any]:
    """Read a NumPy array, rejecting pickle and non-array containers."""
    message = parse_message(_array_message(), data, "Array argument")
    with BytesIO(message.ndarray) as stream:
        try:
            result = np.load(stream, allow_pickle=False)
        except (ValueError, EOFError, OSError, OverflowError) as error:
            raise SerializationError("Invalid NumPy array argument payload") from error
        if isinstance(result, np.ndarray):
            return result
        result.close()
    raise SerializationError("Array argument requires a single NumPy array")


def serialize_complex_argument(value: complex) -> bytes:
    """Encode complex components with the source float32 field precision."""
    try:
        message = _complex_message()(real=np.real(value), imag=np.imag(value))
        return cast(bytes, message.SerializeToString())
    except (TypeError, ValueError, OverflowError) as error:
        raise SerializationError("Invalid complex argument") from error


def deserialize_complex_argument(data: bytes) -> complex:
    """Read the dedicated two-field complex message."""
    message = parse_message(_complex_message(), data, "Complex argument")
    return complex(message.real, message.imag)
