# Copyright 2023 Google LLC
# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Constructors adapted from locked Qualtran v0.7.0 addition.py and negate.py.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/arithmetic/addition.py

"""Resolve integer arithmetic libraries through existing typed component owners."""

from collections.abc import Mapping
from functools import partial
from typing import Literal

import sympy

from ...core.ir.program import FunctionRecord, ProgramRecord
from ...core.ir.program.register_types import unsigned_register_type
from ...core.ir.program.registers import Register, RegisterSide, RegisterType
from ...core.ir.program.signatures import Signature
from ...core.value_parameters import is_symbolic_value
from ...errors import SerializationError
from ..registers.arithmetic.addition_component import addition_component_program
from ..registers.arithmetic.constant_addition_component import constant_addition_program
from ..registers.arithmetic.negation_component import negation_program
from ..registers.arithmetic.out_of_place_addition_component import (
    out_of_place_addition_program,
)
from .legacy_component_resolution import ComponentFactory, ResolvedArgument
from .legacy_word_factories import ConstantEncoder, _constant_pattern, _rename_program

_Kind = Literal["Add", "AddK", "Negate"]


def _integer_dtype(arguments: Mapping[str, ResolvedArgument], key: str) -> RegisterType:
    dtype = arguments.get(key)
    if not isinstance(dtype, RegisterType) or dtype.encoding not in {
        "uint",
        "signed",
        "montgomery",
    }:
        raise SerializationError("Arithmetic requires an integer register dtype")
    return dtype


def _arithmetic_factory(
    encode_constant: ConstantEncoder,
    kind: _Kind,
    name: str,
    registers: tuple[Register, ...] | None,
    arguments: Mapping[str, ResolvedArgument],
) -> FunctionRecord | ProgramRecord:
    allowed = (
        {"a_dtype", "b_dtype"}
        if kind == "Add"
        else {"dtype", "k"} if kind == "AddK" else {"dtype"}
    )
    required = {"a_dtype"} if kind == "Add" else allowed
    if arguments.keys() - allowed or not required <= arguments.keys():
        raise SerializationError(f"Invalid constructor arguments for {kind}")
    dtype = _integer_dtype(arguments, "a_dtype" if kind == "Add" else "dtype")
    other = _integer_dtype(arguments, "b_dtype") if "b_dtype" in arguments else dtype
    signature = tuple(
        Signature.build_from_dtypes(a=dtype, b=other)
        if kind == "Add"
        else Signature.build_from_dtypes(x=dtype)
    )
    if registers is not None and signature != registers:
        raise SerializationError(f"Invalid register signature for {kind}")
    constant = arguments.get("k", 1)
    if (
        dtype.is_symbolic()
        or other.is_symbolic()
        or kind == "AddK"
        and is_symbolic_value(constant)
    ):
        return FunctionRecord(name, (), registers=signature)
    if kind == "Add":
        program = addition_component_program(
            dtype.num_qubits, other.num_qubits, left_dtype=dtype, right_dtype=other
        )
    else:
        if (
            kind == "AddK"
            and isinstance(constant, int)
            and constant < 0
            and dtype.encoding != "signed"
        ):
            constant %= 1 << dtype.num_qubits
        pattern = _constant_pattern(encode_constant, dtype, constant)
        program = (
            constant_addition_program(dtype, pattern)
            if kind == "AddK"
            else negation_program(dtype)
        )
    return _rename_program(program, name)


def _out_of_place_factory(
    name: str,
    registers: tuple[Register, ...] | None,
    arguments: Mapping[str, ResolvedArgument],
) -> FunctionRecord | ProgramRecord:
    if (
        arguments.keys() - {"bitsize", "is_adjoint", "include_most_significant_bit"}
        or "bitsize" not in arguments
    ):
        raise SerializationError(
            "OutOfPlaceAdder requires bitsize and direction/carry flags"
        )
    width = arguments["bitsize"]
    if type(width) is not int and not isinstance(width, sympy.Expr):
        raise SerializationError(
            "OutOfPlaceAdder requires an integer or symbolic width"
        )
    adjoint = arguments.get("is_adjoint", False)
    include_msb = arguments.get("include_most_significant_bit", True)
    if any(
        type(flag) not in (bool, int) or flag not in (0, 1)
        for flag in (adjoint, include_msb)
    ):
        raise SerializationError(
            "OutOfPlaceAdder requires binary direction/carry flags"
        )
    signature = (
        Register("a", unsigned_register_type(width)),
        Register("b", unsigned_register_type(width)),
        Register(
            "c",
            unsigned_register_type(width + int(bool(include_msb))),
            side=RegisterSide.LEFT if adjoint else RegisterSide.RIGHT,
        ),
    )
    if registers is not None and signature != registers:
        raise SerializationError("Invalid register signature for OutOfPlaceAdder")
    if isinstance(width, sympy.Expr):
        return FunctionRecord(name, (), registers=signature)
    return _rename_program(
        out_of_place_addition_program(
            width, include_msb=bool(include_msb), adjoint=bool(adjoint)
        ),
        name,
    )


def arithmetic_factories(
    *, encode_constant: ConstantEncoder
) -> dict[str, ComponentFactory]:
    """Bind complete addition and negation constructor groups to owned programs.

    The caller supplies the existing dtype-aware stored-integer encoder. Generic
    control and adjoint wrappers retain their independent constructor boundary.
    """
    kinds: tuple[_Kind, ...] = ("Add", "AddK", "Negate")
    return {
        **{
            f"qualtran.bloqs.arithmetic.{'negate' if kind == 'Negate' else 'addition'}.{kind}": partial(
                _arithmetic_factory, encode_constant, kind
            )
            for kind in kinds
        },
        "qualtran.bloqs.arithmetic.addition.OutOfPlaceAdder": _out_of_place_factory,
    }
