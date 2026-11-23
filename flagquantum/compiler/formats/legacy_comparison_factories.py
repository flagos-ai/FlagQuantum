# Copyright 2023 Google LLC
# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Constructors adapted from locked Qualtran v0.7.0 arithmetic/comparison.py.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/arithmetic/comparison.py

"""Resolve the complete comparison constructor family through owned components."""

from collections.abc import Mapping
from functools import partial
from typing import Literal

import sympy

from ...core.ir.program import FunctionRecord, ProgramRecord
from ...core.ir.program.register_types import quantum_bit_type, unsigned_register_type
from ...core.ir.program.registers import Register, RegisterSide, RegisterType
from ...core.ir.program.signatures import Signature
from ...core.value_math import SymbolicInt
from ...errors import SerializationError
from ..registers.comparison.constant_component import less_than_constant_program
from ..registers.comparison.controlled_component import controlled_greater_program
from ..registers.comparison.equality_component import (
    equality_constant_program,
    equality_program,
)
from ..registers.comparison.half_component import (
    half_comparison_program,
    half_greater_program,
)
from ..registers.comparison.linear_component import linear_greater_program
from ..registers.comparison.ordering_component import (
    greater_than_program,
    less_equal_program,
)
from ..registers.comparison.tree_component import (
    mixer_program,
    single_qubit_comparison_program,
)
from .legacy_component_resolution import ComponentFactory, ResolvedArgument
from .legacy_word_factories import _rename_program

_Relation = Literal["gt", "ge", "lt", "le"]


def _integer(arguments: Mapping[str, ResolvedArgument], key: str) -> SymbolicInt:
    value = arguments.get(key)
    if type(value) is not int and not isinstance(value, sympy.Expr):
        raise SerializationError(
            f"Comparison {key} requires integer or symbolic metadata"
        )
    assert isinstance(value, (int, sympy.Expr))
    return value


def _finish(
    name: str,
    registers: tuple[Register, ...] | None,
    signature: tuple[Register, ...],
    program: ProgramRecord | None,
) -> FunctionRecord | ProgramRecord:
    if registers is not None and registers != signature:
        raise SerializationError("Invalid register signature for comparison")
    return (
        FunctionRecord(name, (), registers=signature)
        if program is None
        else _rename_program(program, name)
    )


def _tree_factory(
    mixer: bool,
    name: str,
    registers: tuple[Register, ...] | None,
    arguments: Mapping[str, ResolvedArgument],
) -> ProgramRecord:
    if arguments.keys() - {"is_adjoint"}:
        raise SerializationError(
            "Unexpected constructor arguments for comparison flags"
        )
    program = (mixer_program if mixer else single_qubit_comparison_program)(
        adjoint=bool(arguments.get("is_adjoint", False))
    )
    result = _finish(name, registers, program.functions[0].registers or (), program)
    assert isinstance(result, ProgramRecord)
    return result


def _constant_factory(
    kind: Literal["LessThanConstant", "GreaterThanConstant", "EqualsAConstant"],
    name: str,
    registers: tuple[Register, ...] | None,
    arguments: Mapping[str, ResolvedArgument],
) -> FunctionRecord | ProgramRecord:
    key = "less_than_val" if kind == "LessThanConstant" else "val"
    if set(arguments) != {"bitsize", key}:
        raise SerializationError("Invalid constant comparison constructor arguments")
    width, value = _integer(arguments, "bitsize"), _integer(arguments, key)
    signature = tuple(
        Signature.build_from_dtypes(
            x=unsigned_register_type(width), target=quantum_bit_type()
        )
    )
    program = None
    # GreaterThanConstant declares only a model in the unchanged source.
    if (
        kind != "GreaterThanConstant"
        and isinstance(width, int)
        and isinstance(value, int)
    ):
        program = (
            less_than_constant_program(width, value)
            if kind == "LessThanConstant"
            else equality_constant_program(width, value)
        )
    return _finish(name, registers, signature, program)


def _ordering_factory(
    greater: bool,
    name: str,
    registers: tuple[Register, ...] | None,
    arguments: Mapping[str, ResolvedArgument],
) -> FunctionRecord | ProgramRecord:
    left_key, right_key = (
        ("a_bitsize", "b_bitsize") if greater else ("x_bitsize", "y_bitsize")
    )
    if set(arguments) != {left_key, right_key}:
        raise SerializationError("Invalid ordering constructor arguments")
    left, right = _integer(arguments, left_key), _integer(arguments, right_key)
    word, other, bit = (
        unsigned_register_type(left),
        unsigned_register_type(right),
        quantum_bit_type(),
    )
    signature = tuple(
        Signature.build_from_dtypes(a=word, b=other, target=bit)
        if greater
        else Signature.build_from_dtypes(x=word, y=other, target=bit)
    )
    program = None
    if isinstance(left, int) and isinstance(right, int):
        program = (
            greater_than_program(left, right)
            if greater
            else less_equal_program(left, right)
        )
    return _finish(name, registers, signature, program)


def _linear_factory(
    name: str,
    registers: tuple[Register, ...] | None,
    arguments: Mapping[str, ResolvedArgument],
) -> FunctionRecord | ProgramRecord:
    if arguments.keys() - {"bitsize", "signed"} or "bitsize" not in arguments:
        raise SerializationError("Invalid linear comparison constructor arguments")
    width = _integer(arguments, "bitsize")
    word = unsigned_register_type(width)
    signature = tuple(
        Signature.build_from_dtypes(a=word, b=word, target=quantum_bit_type())
    )
    program = (
        linear_greater_program(width, signed=bool(arguments.get("signed", False)))
        if isinstance(width, int)
        else None
    )
    return _finish(name, registers, signature, program)


def _typed_factory(
    kind: Literal["Equals", "controlled", "half"],
    relation: _Relation | None,
    name: str,
    registers: tuple[Register, ...] | None,
    arguments: Mapping[str, ResolvedArgument],
) -> FunctionRecord | ProgramRecord:
    extra = (
        {"cv"} if kind == "controlled" else {"uncompute"} if kind == "half" else set()
    )
    if arguments.keys() - {"dtype", *extra} or "dtype" not in arguments:
        raise SerializationError("Invalid typed comparison constructor arguments")
    dtype = arguments["dtype"]
    if (
        not isinstance(dtype, RegisterType)
        or kind != "Equals"
        and dtype.encoding not in {"uint", "signed", "montgomery"}
    ):
        raise SerializationError("Comparison requires an admitted register dtype")
    bit = quantum_bit_type()
    program = None
    if kind == "Equals":
        signature = tuple(Signature.build_from_dtypes(x=dtype, y=dtype, target=bit))
        if not dtype.is_symbolic():
            program = equality_program(dtype)
    elif kind == "controlled":
        cv = arguments.get("cv", 1)
        if not isinstance(cv, (int, float)) or cv not in (0, 1):
            raise SerializationError(
                "Controlled comparison requires a binary numeric activation"
            )
        signature = tuple(
            Signature.build_from_dtypes(ctrl=bit, a=dtype, b=dtype, target=bit)
        )
        if not dtype.is_symbolic():
            program = controlled_greater_program(dtype, control_value=int(cv))
    else:
        adjoint = bool(arguments.get("uncompute", False))
        side = RegisterSide.LEFT if adjoint else RegisterSide.RIGHT
        signature = (
            Register("a", dtype),
            Register("b", dtype),
            Register("c", unsigned_register_type(dtype.num_qubits + 1), side=side),
            Register("target", bit, side=side),
        )
        if not dtype.is_symbolic():
            program = (
                half_greater_program(dtype, adjoint=adjoint)
                if relation is None
                else half_comparison_program(dtype, relation, adjoint=adjoint)
            )
    return _finish(name, registers, signature, program)


def comparison_factories() -> dict[str, ComponentFactory]:
    """Bind fourteen public constructors and their concrete private half helper.

    Existing components own decompositions and lifetimes. Symbolic metadata and
    the source's model-only GreaterThanConstant remain logical declarations.
    """
    prefix = "qualtran.bloqs.arithmetic.comparison."
    mapping: dict[str, ComponentFactory] = {
        "BiQubitsMixer": partial(_tree_factory, True),
        "SingleQubitCompare": partial(_tree_factory, False),
        "LessThanEqual": partial(_ordering_factory, False),
        "GreaterThan": partial(_ordering_factory, True),
        "LinearDepthGreaterThan": _linear_factory,
        "Equals": partial(_typed_factory, "Equals", None),
        "CLinearDepthGreaterThan": partial(_typed_factory, "controlled", None),
        "_HalfLinearDepthGreaterThan": partial(_typed_factory, "half", None),
        "LinearDepthHalfGreaterThan": partial(_typed_factory, "half", "gt"),
        "LinearDepthHalfGreaterThanEqual": partial(_typed_factory, "half", "ge"),
        "LinearDepthHalfLessThan": partial(_typed_factory, "half", "lt"),
        "LinearDepthHalfLessThanEqual": partial(_typed_factory, "half", "le"),
    }
    kinds: tuple[
        Literal["LessThanConstant", "GreaterThanConstant", "EqualsAConstant"], ...
    ] = ("LessThanConstant", "GreaterThanConstant", "EqualsAConstant")
    mapping.update({kind: partial(_constant_factory, kind) for kind in kinds})
    return {prefix + key: value for key, value in mapping.items()}
