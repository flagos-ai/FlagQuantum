# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Wire layout adapted from Qualtran v0.7.0 protos/sympy.proto.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/protos/sympy.proto

"""Standalone interchange for the recursive legacy symbolic wire format."""

from __future__ import annotations

from functools import cache
from typing import TYPE_CHECKING, Any, cast

from ...core.value_math import SymbolicReal
from ...errors import SerializationError
from ._legacy_expression import _check_integer_size, _check_power_budget
from ._legacy_protobuf import _protobuf_runtime, parse_message

if TYPE_CHECKING:
    from sympy import Expr

_FUNCTION_NAMES = (
    "NONE",
    "Add",
    "Mul",
    "Pow",
    "Mod",
    "Log",
    "Floor",
    "Ceiling",
    "Max",
    "Min",
    "Sin",
    "Cos",
    "Tan",
    "Log2",
)
_CONSTANT_NAMES = ("Pi", "E", "EulerGamma", "Infinity", "ImaginaryUnit")
_ARITIES = {
    "Pow": (2,),
    "Mod": (2,),
    "Log": (1, 2),
    "Floor": (1,),
    "Ceiling": (1,),
    "Sin": (1,),
    "Cos": (1,),
    "Tan": (1,),
    "Log2": (1,),
}


@cache
def _term_message() -> Any:
    """Create mutually recursive descriptors together in a private pool."""
    descriptors, pools, messages = _protobuf_runtime()
    schema = descriptors.FileDescriptorProto(
        name="flagquantum/LegacySymbolicTerm.proto",
        package="flagquantum",
        syntax="proto3",
    )
    for name, values in (
        ("Function", _FUNCTION_NAMES),
        ("ConstSymbol", _CONSTANT_NAMES),
    ):
        enum = schema.enum_type.add(name=name)
        for number, value in enumerate(values):
            enum.value.add(name=value, number=number)

    proto = descriptors.FieldDescriptorProto
    for name, oneof, fields in (
        (
            "Rational",
            None,
            (
                ("numerator", "MESSAGE", "Parameter"),
                ("denominator", "MESSAGE", "Parameter"),
            ),
        ),
        (
            "Parameter",
            "parameter",
            (
                ("const_int", "INT32", None),
                ("symbol", "STRING", None),
                ("const_rat", "MESSAGE", "Rational"),
                ("const_float", "FLOAT", None),
                ("const_symbol", "ENUM", "ConstSymbol"),
            ),
        ),
        (
            "Operand",
            "operand",
            (("term", "MESSAGE", "Term"), ("parameter", "MESSAGE", "Parameter")),
        ),
        (
            "Term",
            None,
            (("function", "ENUM", "Function"), ("operands", "MESSAGE", "Operand")),
        ),
    ):
        message = schema.message_type.add(name=name)
        if oneof is not None:
            message.oneof_decl.add(name=oneof)
        for number, (key, kind, reference) in enumerate(fields, start=1):
            field = message.field.add(
                name=key,
                number=number,
                type=getattr(proto, f"TYPE_{kind}"),
                label=(
                    proto.LABEL_REPEATED if key == "operands" else proto.LABEL_OPTIONAL
                ),
            )
            if reference is not None:
                field.type_name = f".flagquantum.{reference}"
            if oneof is not None:
                field.oneof_index = 0
    descriptor = pools.DescriptorPool().Add(schema)
    return messages.GetMessageClass(descriptor.message_types_by_name["Term"])


@cache
def _symbolic_values() -> tuple[tuple[Any, ...], tuple[Expr, ...]]:
    import sympy as sp
    from sympy.codegen.cfunctions import log2

    functions = (
        None,
        sp.Add,
        sp.Mul,
        sp.Pow,
        sp.Mod,
        sp.log,
        sp.floor,
        sp.ceiling,
        sp.Max,
        sp.Min,
        sp.sin,
        sp.cos,
        sp.tan,
        log2,
    )
    constants = (sp.pi, sp.E, sp.EulerGamma, sp.oo, sp.I)
    return functions, constants


def _write_term(message: Any, value: SymbolicReal) -> None:
    from sympy import Expr

    functions, _ = _symbolic_values()
    if isinstance(value, Expr) and value.func in functions[1:]:
        message.function = functions.index(value.func)
        for arg in value.args:
            _write_term(message.operands.add().term, arg)
    else:
        _write_atom(message.operands.add().parameter, value)


def _write_atom(message: Any, value: SymbolicReal) -> None:
    from sympy import Float, Integer, Rational, Symbol

    _, constants = _symbolic_values()
    if isinstance(value, Symbol):
        message.symbol = str(value)
    elif type(value) is int or isinstance(value, Integer):
        message.const_int = int(value)
    elif type(value) is float or isinstance(value, Float):
        message.const_float = float(value)
    elif isinstance(value, Rational):
        message.const_rat.numerator.const_int = int(value.p)
        message.const_rat.denominator.const_int = int(value.q)
    elif value in constants:
        message.const_symbol = constants.index(value)
    else:
        raise SerializationError(
            f"Unsupported legacy symbolic value: {type(value).__name__}"
        )


def serialize_symbolic_term(value: SymbolicReal) -> bytes:
    """Write a supported scalar or expression tree to the legacy Term format."""
    message = _term_message()()
    try:
        _write_term(message, value)
        return cast(bytes, message.SerializeToString())
    except SerializationError:
        raise
    except (ValueError, TypeError, OverflowError, RecursionError) as error:
        raise SerializationError("Invalid legacy symbolic value") from error


def _read_atom(message: Any) -> SymbolicReal:
    from sympy import Rational, Symbol

    choice = message.WhichOneof("parameter")
    if choice == "const_int":
        return int(message.const_int)
    if choice == "const_float":
        return float(message.const_float)
    if choice == "symbol":
        return Symbol(message.symbol)
    if choice == "const_symbol":
        _, constants = _symbolic_values()
        if 0 <= message.const_symbol < len(constants):
            return constants[message.const_symbol]
    if choice == "const_rat":
        fraction = message.const_rat
        if (
            fraction.numerator.WhichOneof("parameter") == "const_int"
            and fraction.denominator.WhichOneof("parameter") == "const_int"
            and fraction.denominator.const_int != 0
        ):
            return Rational(
                fraction.numerator.const_int, fraction.denominator.const_int
            )
    raise SerializationError("Invalid legacy symbolic parameter")


def _read_term(message: Any) -> SymbolicReal:
    from sympy import Expr

    functions, _ = _symbolic_values()
    if not 0 <= message.function < len(functions):
        raise SerializationError("Unknown legacy symbolic function tag")
    args: list[SymbolicReal] = []
    for operand in message.operands:
        if operand.HasField("term"):
            args.append(_read_term(operand.term))
        else:
            args.append(_read_atom(operand.parameter))
    function = functions[message.function]
    if function is None:
        if len(args) != 1:
            raise SerializationError("Legacy symbolic atom requires one operand")
        return args[0]
    name = _FUNCTION_NAMES[message.function]
    arities = _ARITIES.get(name)
    if arities is not None and len(args) not in arities:
        raise SerializationError(f"Invalid arity for legacy symbolic function {name}")
    if name == "Pow":
        _check_power_budget(args[0], args[1])
    result = function(*args)
    if not isinstance(result, Expr):
        raise SerializationError("Legacy symbolic function did not produce a scalar")
    _check_integer_size(result)
    return result


def deserialize_symbolic_term(data: bytes) -> SymbolicReal:
    """Read a legacy Term through a fixed SymPy constructor vocabulary."""
    message = parse_message(_term_message(), data, "Symbolic term")
    try:
        return _read_term(message)
    except SerializationError:
        raise
    except (TypeError, ValueError, OverflowError, RecursionError) as error:
        raise SerializationError("Invalid legacy symbolic term") from error
