# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Constructors follow locked Qualtran v0.7.0 block_encoding/unitary.py.

"""Resolve packed unitary blocks with explicitly owned exact child pairs."""

from collections.abc import Callable, Mapping
from sys import modules
from typing import cast

from ...core.ir.program import FunctionRecord, ProgramRecord
from ...core.ir.program.registers import Register
from ...core.value_math import SymbolicInt
from ...errors import SerializationError
from ..registers.layout.packed_block_component import unitary_block_component_program
from .legacy_component_resolution import ComponentFactory, ResolvedArgument
from .legacy_fixed_gate_factories import fixed_gate_factories
from .legacy_rotation_factories import _real_argument
from .legacy_word_factories import _rename_program

UnitaryPairOwner = Callable[[FunctionRecord], tuple[ProgramRecord, ProgramRecord]]
_Recipe = tuple[ProgramRecord, ProgramRecord, SymbolicInt, SymbolicInt, bool]


def _width(arguments: Mapping[str, ResolvedArgument], name: str) -> SymbolicInt:
    value = arguments.get(name, 0)
    if type(value) is int:
        return value
    symbolic = modules.get("sympy")
    if symbolic is not None and isinstance(value, symbolic.Expr):
        return cast(SymbolicInt, value)
    raise SerializationError("Unitary auxiliary widths require integers or symbols")


def unitary_block_factories(
    *,
    child_owner: UnitaryPairOwner | None = None,
    adjoint_factory: ComponentFactory | None = None,
) -> dict[str, ComponentFactory]:
    """Keep fixed-gate pairs locally and require exact owners for other children.

    Original TensorProduct tuple-of-Bloq serialization fails upstream. Its
    direct native graph belongs to layout; this mapping invents no wire codec.
    """
    children: dict[FunctionRecord, tuple[ProgramRecord, ProgramRecord]] = {}
    recipes: dict[FunctionRecord, _Recipe] = {}

    def track(identifier: str, factory: ComponentFactory) -> ComponentFactory:
        def make_child(
            name: str,
            registers: tuple[Register, ...] | None,
            arguments: Mapping[str, ResolvedArgument],
        ) -> FunctionRecord | ProgramRecord:
            forward = factory(name, registers, arguments)
            assert isinstance(forward, FunctionRecord)
            reverse = dict(arguments)
            if identifier.endswith((".SGate", ".TGate")):
                direction = arguments.get("is_adjoint", 0)
                assert isinstance(direction, int)
                reverse["is_adjoint"] = 1 - direction
            inverse = factory(name + "_exact_inverse", registers, reverse)
            assert isinstance(inverse, FunctionRecord)
            children[forward] = ProgramRecord((forward,)), ProgramRecord((inverse,))
            return forward

        return make_child

    def make(
        name: str, registers: tuple[Register, ...] | None, recipe: _Recipe
    ) -> ProgramRecord:
        child, inverse, ancilla, resource, direction = recipe
        record = unitary_block_component_program(
            child,
            ancilla=ancilla,
            resource=resource,
            inverse=inverse,
            adjoint=direction,
        )
        if registers is not None and registers != record.functions[0].registers:
            raise SerializationError("Invalid unitary block register signature")
        record = _rename_program(record, name)
        recipes[record.functions[0]] = recipe
        return record

    def unitary(
        name: str,
        registers: tuple[Register, ...] | None,
        arguments: Mapping[str, ResolvedArgument],
    ) -> ProgramRecord:
        if "U" not in arguments or arguments.keys() - {
            "U",
            "alpha",
            "ancilla_bitsize",
            "resource_bitsize",
            "epsilon",
        }:
            raise SerializationError("Unitary block requires its original U child")
        child = arguments["U"]
        if not isinstance(child, FunctionRecord):
            raise SerializationError("Unitary block requires a native child function")
        _real_argument(arguments, "alpha", 1)
        _real_argument(arguments, "epsilon", 0)
        if child in children:
            pair = children[child]
        elif child_owner is not None:
            pair = child_owner(child)
        else:
            raise SerializationError("Missing exact unitary child owner")
        return make(
            name,
            registers,
            (
                *pair,
                _width(arguments, "ancilla_bitsize"),
                _width(arguments, "resource_bitsize"),
                False,
            ),
        )

    def adjoint(
        name: str,
        registers: tuple[Register, ...] | None,
        arguments: Mapping[str, ResolvedArgument],
    ) -> FunctionRecord | ProgramRecord:
        child = arguments.get("subbloq")
        if (
            arguments.keys() == {"subbloq"}
            and isinstance(child, FunctionRecord)
            and child in recipes
        ):
            forward, inverse, ancilla, resource, direction = recipes[child]
            return make(
                name, registers, (forward, inverse, ancilla, resource, not direction)
            )
        if adjoint_factory is not None:
            return adjoint_factory(name, registers, arguments)
        raise SerializationError("Missing exact unitary block adjoint context")

    return {
        **{key: track(key, factory) for key, factory in fixed_gate_factories().items()},
        "qualtran.bloqs.block_encoding.unitary.Unitary": unitary,
        "qualtran._infra.adjoint.Adjoint": adjoint,
    }
