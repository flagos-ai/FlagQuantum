# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Wire layout adapted from Qualtran v0.7.0 protos/ctrl_spec.proto.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/protos/ctrl_spec.proto

"""Legacy control specification interchange with owned type and array codecs."""

from operator import index
from typing import Any, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

from ...core.ir.program.registers import RegisterType
from ...errors import SerializationError
from ..registers.control.control_predicate import ControlRegister, IntegerEncoding
from ._legacy_protobuf import message_factory, parse_message
from .legacy_arguments import (
    _array_message,
    _read_array_argument,
    _write_array_argument,
)
from .legacy_register_types import (
    _read_register_type,
    _register_type_message,
    _write_register_type,
)

ControlClauses: TypeAlias = tuple[tuple[RegisterType, NDArray[Any]], ...]


def _control_spec_message() -> Any:
    return message_factory(
        "ControlSpecification",
        (("qdtypes", _register_type_message()), ("cvs", _array_message())),
        repeated=("qdtypes", "cvs"),
    )


def serialize_control_spec(clauses: ControlClauses) -> bytes:
    """Encode the ordered register declarations and concrete control shapes."""
    message = _control_spec_message()()
    for dtype, values in clauses:
        if not isinstance(values, np.ndarray):
            raise SerializationError(
                "Control specification requires concrete NumPy arrays"
            )
        _write_register_type(message.qdtypes.add(), dtype)
        _write_array_argument(message.cvs.add(), values)
    return cast(bytes, message.SerializeToString())


def deserialize_control_spec(data: bytes) -> ControlClauses:
    """Read paired clauses without discarding dtype, shape, or array order."""
    message = parse_message(_control_spec_message(), data, "Control specification")
    if len(message.qdtypes) != len(message.cvs):
        raise SerializationError("Control specification type and value counts differ")
    return tuple(
        (_read_register_type(dtype), _read_array_argument(values))
        for dtype, values in zip(message.qdtypes, message.cvs, strict=True)
    )


def control_registers_from_spec(clauses: ControlClauses) -> tuple[ControlRegister, ...]:
    """Bind concrete wire clauses to C-order native integer control groups."""
    registers = []
    for dtype, values in clauses:
        width = dtype.quantum_bits
        params = dict(dtype.parameters)
        if dtype.classical_bits or type(width) is not int or dtype.is_symbolic():
            raise ValueError("Control specification requires a concrete quantum word")
        encoding: IntegerEncoding
        if dtype.encoding == "ones_complement":
            encoding = "ones_complement"
        elif dtype.encoding == "signed" or (
            dtype.encoding == "fixed_point" and params.get("signed", False)
        ):
            encoding = "twos_complement"
        elif dtype.encoding in {"bit", "opaque", "uint", "montgomery", "fixed_point"}:
            encoding = "unsigned"
        else:
            raise ValueError("Control specification has no integer encoding")
        words = tuple(index(value) for value in values.flat)
        bound = params.get("bound")
        if isinstance(bound, int) and any(value >= bound for value in words):
            raise ValueError("Control value is outside the bounded integer domain")
        registers.append(ControlRegister(width, words, encoding))
    return tuple(registers)
