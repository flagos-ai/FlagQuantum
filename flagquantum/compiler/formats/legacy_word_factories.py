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
from ...core.ir.program.register_types import opaque_register_type, quantum_bit_type
from ...core.ir.program.registers import Register, RegisterType
from ...core.ir.program.signatures import Signature
from ...core.value_parameters import is_symbolic_value
from ...errors import SerializationError
from ..registers.bits.bitwise_component import bitwise_word_program
from ..registers.bits.bitwise_component_control import controlled_bitwise_word_program
from ..registers.bits.on_each import _validate_on_each_child, on_each_word_program
from ..registers.component_graph import _namespace_helpers
from .legacy_component_arguments import ControlSpecification
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
    if not isinstance(child, FunctionRecord):
        raise SerializationError("OnEach requires a typed child function")
    try:
        _validate_on_each_child(child)
    except ValueError as error:
        raise SerializationError(str(error)) from error
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
    # The resolver retains the real child and its helpers in the enclosing module.
    declaration = replace(child, blocks=())
    program = on_each_word_program(width, declaration, target_dtype=dtype)
    program = replace(
        program,
        functions=(replace(program.functions[0], name=name), *program.functions[1:]),
    )
    return _namespace_helpers(program, name, external=(child.name,))


def word_factories(*, encode_constant: ConstantEncoder) -> dict[str, ComponentFactory]:
    """Supply bitwise constructors with a caller-owned concrete value encoder.

    Compiler consumes MSB bits and constructs typed calls. Numerical encoding
    and its dtype-specific validation remain with the injected codec owner.
    OnEach references the independently resolved scalar one-bit child closure.
    Controlled accepts the positive scalar QBit complement dependency. Its
    child must have been resolved in this catalogue, including symbolic views.
    Create a catalogue per compilation to bound retained constructor context.
    """
    complements: dict[FunctionRecord, RegisterType] = {}

    def word_factory(
        kind: _Kind,
        name: str,
        registers: tuple[Register, ...] | None,
        arguments: Mapping[str, ResolvedArgument],
    ) -> FunctionRecord | ProgramRecord:
        result = _word_factory(encode_constant, kind, name, registers, arguments)
        if kind == "BitwiseNot":
            root = result.functions[0] if isinstance(result, ProgramRecord) else result
            assert root.registers is not None
            complements[root] = root.registers[0].dtype
        return result

    def controlled_factory(
        name: str,
        registers: tuple[Register, ...] | None,
        arguments: Mapping[str, ResolvedArgument],
    ) -> FunctionRecord | ProgramRecord:
        if set(arguments) != {"subbloq", "ctrl_spec"}:
            raise SerializationError("Controlled requires subbloq and ctrl_spec")
        child, spec = arguments["subbloq"], arguments["ctrl_spec"]
        if not isinstance(child, FunctionRecord) or child not in complements:
            raise SerializationError(
                "Missing controlled complement constructor context"
            )
        if not isinstance(spec, ControlSpecification) or len(spec.clauses) != 1:
            raise SerializationError(
                "Controlled complement requires one control clause"
            )
        control_dtype, values = spec.clauses[0]
        bit = quantum_bit_type()
        if control_dtype != bit or values.shape != () or values.item() != 1:
            raise SerializationError(
                "Controlled complement requires positive scalar QBit"
            )
        dtype = complements[child]
        signature = tuple(Signature.build_from_dtypes(ctrl=bit, x=dtype))
        _check_signature(signature, registers)
        if dtype.is_symbolic():
            return FunctionRecord(name, (), registers=signature)
        return _rename_program(
            controlled_bitwise_word_program(dtype, "bitwise_not"), name
        )

    kinds: tuple[_Kind, ...] = ("Xor", "XorK", "BitwiseNot")
    return {
        **{
            f"qualtran.bloqs.arithmetic.bitwise.{kind}": partial(word_factory, kind)
            for kind in kinds
        },
        "qualtran.bloqs.basic_gates.on_each.OnEach": _on_each_factory,
        "qualtran._infra.controlled.Controlled": controlled_factory,
    }
