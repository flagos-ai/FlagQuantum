# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Constructor controls follow locked Qualtran v0.7.0 addition.py and bitwise.py.

"""Retain constant constructor identity through scalar control and adjoint views."""

from collections.abc import Mapping
from functools import partial
from typing import Literal

import numpy as np
import sympy

from ...core.ir.program import FunctionRecord, ProgramRecord
from ...core.ir.program.register_types import quantum_bit_type
from ...core.ir.program.registers import Register, RegisterType
from ...core.ir.program.signatures import Signature
from ...errors import SerializationError
from ..registers.arithmetic.addition_component import addition_component_program
from ..registers.arithmetic.constant_addition_component import constant_addition_program
from ..registers.arithmetic.nested_controlled_constant_component import (
    nested_controlled_constant_addition_program,
)
from ..registers.bits.bitwise_component import bitwise_word_program
from ..registers.bits.bitwise_component_control import (
    controlled_bitwise_word_program,
    nested_controlled_xor_constant_program,
)
from ..registers.component_control import (
    ControlAdder,
    _stored_control_adder,
    quantum_control_signature,
)
from .legacy_arithmetic_factories import arithmetic_factories
from .legacy_component_arguments import ControlSpecification
from .legacy_component_resolution import ComponentFactory, ResolvedArgument
from .legacy_controlled_factories import controlled_via_and_factories
from .legacy_word_factories import ConstantEncoder, _rename_program, word_factories

_Kind = Literal["add", "xor"]
_Recipe = tuple[_Kind, RegisterType, int | None, int, bool]


def unresolved_constant_addition_program(
    dtype: RegisterType,
    constant: sympy.Expr,
    *,
    name: str,
    controlled: bool = False,
    adjoint: bool = False,
) -> ProgramRecord:
    """Retain an actual unresolved AddK, exact adjoint and positive control."""
    return _unresolved_constant_program(
        "add", dtype, constant, name=name, controlled=controlled, adjoint=adjoint
    )


def unresolved_xor_constant_program(
    dtype: RegisterType,
    constant: sympy.Expr,
    *,
    name: str,
    controlled: bool = False,
) -> ProgramRecord:
    """Retain an actual unresolved XorK and its positive control without encoding."""
    return _unresolved_constant_program(
        "xor", dtype, constant, name=name, controlled=controlled
    )


def _unresolved_constant_program(
    kind: _Kind,
    dtype: RegisterType,
    constant: sympy.Expr,
    *,
    name: str,
    controlled: bool = False,
    adjoint: bool = False,
) -> ProgramRecord:
    """Retain an actual unresolved AddK constructor at its source refusal.

    Reuse constructor validation and control/adjoint provenance without a
    numeric encoder. The source classifies every SymPy expression as symbolic,
    including Integer constants. Python integers must use the numerical path.
    """
    if not isinstance(constant, sympy.Expr):
        raise SerializationError(
            "A source symbolic constant requires a SymPy expression"
        )

    def forbidden(word: RegisterType, value: object) -> tuple[int, ...]:
        raise SerializationError("An unresolved source constant has no bit encoding")

    factories = constant_control_factories(encode_constant=forbidden)
    child = factories[
        (
            "qualtran.bloqs.arithmetic.addition.AddK"
            if kind == "add"
            else "qualtran.bloqs.arithmetic.bitwise.XorK"
        )
    ](
        name if not (controlled or adjoint) else name + "_constant",
        None,
        {"dtype": dtype, "k": constant},
    )
    assert isinstance(child, FunctionRecord) and not child.blocks
    if adjoint:
        child = factories["qualtran._infra.adjoint.Adjoint"](
            name + "_adjoint" if controlled else name, None, {"subbloq": child}
        )
        assert isinstance(child, FunctionRecord) and not child.blocks
    if controlled:
        child = factories["qualtran._infra.controlled.Controlled"](
            name,
            None,
            {
                "subbloq": child,
                "ctrl_spec": ControlSpecification(
                    ((quantum_bit_type(), np.asarray(1)),)
                ),
            },
        )
    assert isinstance(child, FunctionRecord) and not child.blocks
    return ProgramRecord((child,))


def constant_control_factories(
    *,
    encode_constant: ConstantEncoder,
    controlled_factory: ComponentFactory | None = None,
    adjoint_factory: ComponentFactory | None = None,
) -> dict[str, ComponentFactory]:
    """Resolve known AddK/XorK owners and nested positive controls.

    Context is private to one compilation. Encoding and negative-constant
    normalization remain with the existing constructor and injected codec.
    Unknown children require an explicit fallback. Unbound constants retain
    declarations; they are neither encoded as zero nor priced as free gates.
    """
    recipes: dict[FunctionRecord, _Recipe] = {}
    sums: dict[FunctionRecord, tuple[RegisterType, RegisterType, bool]] = {}
    arithmetic = arithmetic_factories(encode_constant=encode_constant)

    def addition(
        name: str,
        registers: tuple[Register, ...] | None,
        arguments: Mapping[str, ResolvedArgument],
    ) -> FunctionRecord | ProgramRecord:
        result = arithmetic["qualtran.bloqs.arithmetic.addition.Add"](
            name, registers, arguments
        )
        root = result.functions[0] if isinstance(result, ProgramRecord) else result
        left = arguments["a_dtype"]
        right = arguments.get("b_dtype", left)
        assert isinstance(left, RegisterType) and isinstance(right, RegisterType)
        sums[root] = left, right, False
        return result

    def remember(
        result: FunctionRecord | ProgramRecord, recipe: _Recipe
    ) -> FunctionRecord | ProgramRecord:
        root = result.functions[0] if isinstance(result, ProgramRecord) else result
        recipes[root] = recipe
        return result

    def forward(
        kind: _Kind,
        name: str,
        registers: tuple[Register, ...] | None,
        arguments: Mapping[str, ResolvedArgument],
    ) -> FunctionRecord | ProgramRecord:
        captured: tuple[int, ...] | None = None

        def encode(dtype: RegisterType, value: object) -> tuple[int, ...]:
            nonlocal captured
            captured = encode_constant(dtype, value)
            return captured

        factory = (
            arithmetic_factories(encode_constant=encode)[
                "qualtran.bloqs.arithmetic.addition.AddK"
            ]
            if kind == "add"
            else word_factories(encode_constant=encode)[
                "qualtran.bloqs.arithmetic.bitwise.XorK"
            ]
        )
        result = factory(name, registers, arguments)
        dtype = arguments["dtype"]
        assert isinstance(dtype, RegisterType)
        # The existing factory has already validated every captured MSB bit.
        pattern = (
            None
            if captured is None
            else sum(bit << (len(captured) - i - 1) for i, bit in enumerate(captured))
        )
        return remember(result, (kind, dtype, pattern, 0, False))

    def render(
        name: str,
        registers: tuple[Register, ...] | None,
        signature: Signature,
        recipe: _Recipe,
    ) -> FunctionRecord | ProgramRecord:
        if registers is not None and registers != tuple(signature):
            raise SerializationError("Invalid constant control or adjoint signature")
        kind, dtype, pattern, count, inverse = recipe
        if pattern is None:
            return remember(
                FunctionRecord(name, (), registers=tuple(signature)), recipe
            )
        if kind == "add":
            program = (
                nested_controlled_constant_addition_program(
                    dtype, pattern, adjoint=inverse, control_count=count
                )
                if count >= 2
                else constant_addition_program(
                    dtype, pattern, adjoint=inverse, control_value=1 if count else None
                )
            )
        else:
            program = (
                nested_controlled_xor_constant_program(
                    dtype, pattern, control_count=count
                )
                if count >= 2
                else (
                    controlled_bitwise_word_program(
                        dtype, "xor_constant", pattern=pattern
                    )
                    if count
                    else bitwise_word_program(dtype, "xor_constant", pattern=pattern)
                )
            )
        assert program.functions[0].registers == tuple(signature)
        return remember(_rename_program(program, name), recipe)

    def controlled(
        name: str,
        registers: tuple[Register, ...] | None,
        arguments: Mapping[str, ResolvedArgument],
    ) -> FunctionRecord | ProgramRecord:
        child = arguments.get("subbloq")
        if not isinstance(child, FunctionRecord) or child not in recipes:
            if controlled_factory is not None:
                return controlled_factory(name, registers, arguments)
            raise SerializationError("Missing constant controlled constructor context")
        spec = arguments.get("ctrl_spec")
        if (
            set(arguments) != {"subbloq", "ctrl_spec"}
            or not isinstance(spec, ControlSpecification)
            or len(spec.clauses) != 1
        ):
            raise SerializationError(
                "Constant control requires one scalar positive QBit"
            )
        dtype, values = spec.clauses[0]
        if dtype != quantum_bit_type() or values.shape or values.item() != 1:
            raise SerializationError(
                "Constant control requires one scalar positive QBit"
            )
        kind, word, pattern, count, inverse = recipes[child]
        assert child.registers is not None
        signature = quantum_control_signature(
            Signature(child.registers), ((dtype, ()),)
        )
        return render(
            name, registers, signature, (kind, word, pattern, count + 1, inverse)
        )

    def adjoint(
        name: str,
        registers: tuple[Register, ...] | None,
        arguments: Mapping[str, ResolvedArgument],
    ) -> FunctionRecord | ProgramRecord:
        child = arguments.get("subbloq")
        if isinstance(child, FunctionRecord) and child in sums:
            if set(arguments) != {"subbloq"}:
                raise SerializationError("Addition adjoint requires one child")
            assert child.registers is not None
            signature = Signature(child.registers).adjoint()
            if registers is not None and tuple(signature) != registers:
                raise SerializationError("Invalid addition adjoint signature")
            left, right, inverse = sums[child]
            result: FunctionRecord | ProgramRecord = (
                _rename_program(
                    addition_component_program(
                        left.num_qubits,
                        right.num_qubits,
                        left_dtype=left,
                        right_dtype=right,
                        adjoint=not inverse,
                    ),
                    name,
                )
                if child.blocks
                else FunctionRecord(name, (), registers=tuple(signature))
            )
            root = result.functions[0] if isinstance(result, ProgramRecord) else result
            sums[root] = left, right, not inverse
            return result
        if not isinstance(child, FunctionRecord) or child not in recipes:
            if adjoint_factory is not None:
                return adjoint_factory(name, registers, arguments)
            raise SerializationError("Missing constant adjoint constructor context")
        if set(arguments) != {"subbloq"}:
            raise SerializationError("Constant adjoint requires one child")
        assert child.registers is not None
        kind, dtype, pattern, count, inverse = recipes[child]
        return render(
            name,
            registers,
            Signature(child.registers).adjoint(),
            (kind, dtype, pattern, count, not inverse),
        )

    def positive(child: FunctionRecord) -> tuple[ProgramRecord, ControlAdder]:
        if child not in recipes or child.registers is None:
            raise SerializationError("Missing constant predicate constructor context")
        kind, dtype, pattern, count, inverse = recipes[child]
        signature = quantum_control_signature(
            Signature(child.registers), ((quantum_bit_type(), ()),)
        )
        result = render(
            child.name + "_positive",
            None,
            signature,
            (kind, dtype, pattern, count + 1, inverse),
        )
        module = (
            result if isinstance(result, ProgramRecord) else ProgramRecord((result,))
        )
        return module, _stored_control_adder(module.functions[0])

    return {
        **controlled_via_and_factories(control_system=positive),
        "qualtran.bloqs.arithmetic.addition.Add": addition,
        "qualtran.bloqs.arithmetic.addition.AddK": partial(forward, "add"),
        "qualtran.bloqs.arithmetic.bitwise.XorK": partial(forward, "xor"),
        "qualtran._infra.controlled.Controlled": controlled,
        "qualtran._infra.adjoint.Adjoint": adjoint,
    }
