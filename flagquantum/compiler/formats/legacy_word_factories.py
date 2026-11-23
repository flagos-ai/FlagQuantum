# Copyright 2023 Google LLC
# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Constructors adapted from locked Qualtran v0.7.0 arithmetic/bitwise.py and OnEach.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/arithmetic/bitwise.py

"""Resolve bitwise and repetition libraries using owned word components."""

from collections.abc import Callable, Mapping
from dataclasses import replace
from functools import partial
from typing import Literal

import sympy

from ...core.ir.program import FunctionRecord, ProgramRecord
from ...core.ir.program.register_types import opaque_register_type
from ...core.ir.program.registers import Register, RegisterType
from ...core.ir.program.signatures import Signature
from ...core.value_parameters import is_symbolic_value
from ...errors import SerializationError
from ..registers.bits.bitwise_component import bitwise_word_program
from ..registers.bits.on_each import on_each_word_program
from ..registers.component_graph import _namespace_helpers
from .legacy_component_resolution import ComponentFactory, ResolvedArgument

ConstantEncoder = Callable[[RegisterType, object], tuple[int, ...]]
_Kind = Literal["Xor", "XorK", "BitwiseNot"]


def _constant_pattern(
    encode_constant: ConstantEncoder, dtype: RegisterType, constant: object
) -> int:
    bits = encode_constant(dtype, constant)
    if len(bits) != dtype.num_qubits or any(
        type(bit) is not int or bit not in (0, 1) for bit in bits
    ):
        raise SerializationError("Constant encoder must return one binary MSB word")
    pattern = 0
    for bit in bits:
        pattern = (pattern << 1) | bit
    return pattern


def _rename_program(program: ProgramRecord, name: str) -> ProgramRecord:
    program = replace(
        program,
        functions=(replace(program.functions[0], name=name), *program.functions[1:]),
    )
    return _namespace_helpers(program, name)


def _check_signature(
    signature: tuple[Register, ...], registers: tuple[Register, ...] | None
) -> None:
    if registers is not None and signature != registers:
        raise SerializationError("Invalid register signature for word constructor")


def _word_factory(
    encode_constant: ConstantEncoder,
    kind: _Kind,
    name: str,
    registers: tuple[Register, ...] | None,
    arguments: Mapping[str, ResolvedArgument],
) -> FunctionRecord | ProgramRecord:
    expected = {"dtype", "k"} if kind == "XorK" else {"dtype"}
    if set(arguments) != expected:
        raise SerializationError(f"Invalid constructor arguments for {kind}")
    dtype = arguments["dtype"]
    if not isinstance(dtype, RegisterType):
        raise SerializationError("Word constructor requires a register dtype")
    signature = tuple(
        Signature.build_from_dtypes(x=dtype, y=dtype)
        if kind == "Xor"
        else Signature.build_from_dtypes(x=dtype)
    )
    _check_signature(signature, registers)
    constant = arguments.get("k")
    if dtype.is_symbolic() or kind == "XorK" and is_symbolic_value(constant):
        return FunctionRecord(name, (), registers=signature)
    pattern = None
    if kind == "XorK":
        pattern = _constant_pattern(encode_constant, dtype, constant)
    operations: dict[_Kind, Literal["xor", "xor_constant", "bitwise_not"]] = {
        "Xor": "xor",
        "XorK": "xor_constant",
        "BitwiseNot": "bitwise_not",
    }
    return _rename_program(
        bitwise_word_program(dtype, operations[kind], pattern=pattern), name
    )


def _on_each_factory(
    name: str,
    registers: tuple[Register, ...] | None,
    arguments: Mapping[str, ResolvedArgument],
) -> FunctionRecord | ProgramRecord:
    if (
        arguments.keys() - {"n", "gate", "target_dtype"}
        or not {"n", "gate"} <= arguments.keys()
    ):
        raise SerializationError("OnEach requires n, gate and optional target_dtype")
    width, child = arguments["n"], arguments["gate"]
    if type(width) is not int and not isinstance(width, sympy.Expr):
        raise SerializationError("OnEach requires an integer or symbolic width")
    if not isinstance(child, FunctionRecord) or child.registers != tuple(
        Signature.build(q=1)
    ):
        raise SerializationError("OnEach requires one scalar QBit THRU port named q")
    if any(op.callee is not None for block in child.blocks for op in block.operations):
        raise SerializationError("OnEach requires a self-contained child definition")
    dtype = arguments.get("target_dtype", opaque_register_type(width))
    if (
        not isinstance(dtype, RegisterType)
        or dtype.num_qubits != width
        or dtype.num_cbits != 0
    ):
        raise SerializationError("OnEach target dtype must match its quantum width")
    signature = (Register("q", dtype),)
    _check_signature(signature, registers)
    if isinstance(width, sympy.Expr) or dtype.is_symbolic():
        return FunctionRecord(name, (), registers=signature)
    return _rename_program(on_each_word_program(width, child, target_dtype=dtype), name)


def word_factories(*, encode_constant: ConstantEncoder) -> dict[str, ComponentFactory]:
    """Supply bitwise constructors with a caller-owned concrete value encoder.

    Compiler consumes MSB bits and constructs typed calls. Numerical encoding
    and its dtype-specific validation remain with the injected codec owner.
    OnEach accepts the existing self-contained scalar QBit child interface.
    """
    kinds: tuple[_Kind, ...] = ("Xor", "XorK", "BitwiseNot")
    return {
        **{
            f"qualtran.bloqs.arithmetic.bitwise.{kind}": partial(
                _word_factory, encode_constant, kind
            )
            for kind in kinds
        },
        "qualtran.bloqs.basic_gates.on_each.OnEach": _on_each_factory,
    }
