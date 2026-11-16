# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Argument dispatch adapted from Qualtran v0.7.0 serialization/bloq.py.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/serialization/bloq.py

"""Typed argument values for legacy component libraries."""

from dataclasses import dataclass
from sys import modules
from typing import Any, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

from ...core.ir.program.registers import Register, RegisterType
from ...core.value_math import SymbolicInt, SymbolicReal
from ...errors import SerializationError
from ._legacy_component_schema import _component_argument_message
from ._legacy_protobuf import parse_message
from .legacy_arguments import (
    _read_array_argument,
    _write_array_argument,
    deserialize_complex_argument,
    serialize_complex_argument,
)
from .legacy_control_specs import (
    ControlClauses,
    deserialize_control_spec,
    serialize_control_spec,
)
from .legacy_curve_points import deserialize_curve_point, serialize_curve_point
from .legacy_register_types import _read_register_type, _write_register_type
from .legacy_registers import (
    _read_register,
    _write_register,
    deserialize_registers,
    serialize_registers,
)
from .legacy_symbolic_terms import _read_term, _write_term


@dataclass(frozen=True)
class ComponentReference:
    """Reference another definition by its ID within this library."""

    identifier: int


@dataclass(frozen=True)
class ExternalGatePayload:
    """Keep compressed legacy gate data for an explicit resolver to interpret."""

    gzip_json: bytes


@dataclass(frozen=True)
class ControlSpecification:
    """Distinguish ordered control clauses from a plain array argument."""

    clauses: ControlClauses


@dataclass(frozen=True)
class AffinePointArgument:
    """Coordinates and field parameters at the Compiler interchange boundary."""

    x: SymbolicInt
    y: SymbolicInt
    mod: SymbolicInt
    curve_a: SymbolicInt


ComponentArgumentValue: TypeAlias = (
    "SymbolicReal | str | complex | NDArray[Any] | Register | tuple[object, ...] | list[Any]"
    " | RegisterType | ComponentReference | ExternalGatePayload"
    " | ControlSpecification | AffinePointArgument"
)


@dataclass(frozen=True)
class ComponentArgument:
    name: str
    value: ComponentArgumentValue


def _write_component_argument(message: Any, arg: ComponentArgument) -> None:
    message.name = arg.name
    value = arg.value
    sympy = modules.get("sympy")
    try:
        if isinstance(value, int):
            message.int_val = value
        elif isinstance(value, float):
            message.float_val = value
        elif isinstance(value, str):
            message.string_val = value
        elif sympy is not None and isinstance(value, sympy.Expr):
            _write_term(message.sympy_expr, value)
        elif isinstance(value, ComponentReference):
            message.subbloq = value.identifier
        elif isinstance(value, ExternalGatePayload):
            message.cirq_json_gzip = value.gzip_json
        elif isinstance(value, RegisterType):
            _write_register_type(message.qdata_type, value)
        elif isinstance(value, Register):
            _write_register(message.register, value)
        elif isinstance(value, ControlSpecification):
            message.ctrl_spec.ParseFromString(serialize_control_spec(value.clauses))
        elif isinstance(value, AffinePointArgument):
            message.ec_point.ParseFromString(serialize_curve_point(value))
        elif isinstance(value, tuple) and all(isinstance(x, Register) for x in value):
            registers = cast(tuple[Register, ...], value)
            message.registers.ParseFromString(serialize_registers(registers))
        elif isinstance(value, (np.ndarray, tuple, list)):
            _write_array_argument(message.ndarray, np.asarray(value))
        elif isinstance(value, (complex, np.complexfloating)):
            message.complex_val.ParseFromString(
                serialize_complex_argument(complex(value))
            )
        else:
            raise SerializationError(f"Unsupported component argument {arg.name!r}")
    except SerializationError:
        raise
    except (TypeError, ValueError, OverflowError) as error:
        raise SerializationError(f"Invalid component argument {arg.name!r}") from error


def _read_argument_value(message: Any) -> ComponentArgumentValue:
    match message.WhichOneof("val"):
        case "int_val":
            return int(message.int_val)
        case "float_val":
            return float(message.float_val)
        case "string_val":
            return str(message.string_val)
        case "sympy_expr":
            return _read_term(message.sympy_expr)
        case "ndarray":
            return _read_array_argument(message.ndarray)
        case "subbloq":
            return ComponentReference(message.subbloq)
        case "cirq_json_gzip":
            return ExternalGatePayload(message.cirq_json_gzip)
        case "qdata_type":
            return _read_register_type(message.qdata_type)
        case "register":
            return _read_register(message.register)
        case "registers":
            return deserialize_registers(message.registers.SerializeToString())
        case "ctrl_spec":
            return ControlSpecification(
                deserialize_control_spec(message.ctrl_spec.SerializeToString())
            )
        case "complex_val":
            return deserialize_complex_argument(message.complex_val.SerializeToString())
        case "ec_point":
            return deserialize_curve_point(
                message.ec_point.SerializeToString(), AffinePointArgument
            )
    raise SerializationError(
        f"Component argument {message.name!r} has no selected value"
    )


def _read_component_argument(message: Any) -> ComponentArgument:
    return ComponentArgument(message.name, _read_argument_value(message))


def serialize_component_argument(arg: ComponentArgument) -> bytes:
    """Encode a single named argument with its selected wire arm."""
    message = _component_argument_message()()
    _write_component_argument(message, arg)
    return cast(bytes, message.SerializeToString())


def deserialize_component_argument(data: bytes) -> ComponentArgument:
    """Decode a named argument without resolving external constructors."""
    message = parse_message(_component_argument_message(), data, "Component argument")
    return _read_component_argument(message)
