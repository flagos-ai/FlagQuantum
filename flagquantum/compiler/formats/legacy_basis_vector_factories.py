# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Constructors follow Qualtran v0.7.0 basic_gates/z_basis.py.

"""Resolve computational kets and checked covectors using shared word owners."""

from collections.abc import Mapping
from dataclasses import replace
from functools import partial
from typing import Literal

import sympy

from ...core.ir.program import FunctionRecord, ProgramRecord
from ...core.ir.program.register_types import opaque_register_type, quantum_bit_type
from ...core.ir.program.registers import Register, RegisterSide
from ...errors import SerializationError
from ..registers.component_basis_vectors import (
    integer_basis_function,
    integer_basis_program,
)
from .legacy_component_resolution import ComponentFactory, ResolvedArgument
from .legacy_modular_addition_factories import _integer

_Kind = Literal[
    "ZeroState", "ZeroEffect", "OneState", "OneEffect", "IntState", "IntEffect"
]


def _basis_factory(
    kind: _Kind,
    name: str,
    registers: tuple[Register, ...] | None,
    arguments: Mapping[str, ResolvedArgument],
) -> FunctionRecord | ProgramRecord:
    state = kind.endswith("State")
    side = RegisterSide.RIGHT if state else RegisterSide.LEFT
    if kind.startswith("Int"):
        if set(arguments) != {"val", "bitsize"}:
            raise SerializationError(
                "Integer basis constructor requires val and bitsize"
            )
        value, width = _integer(arguments, "val"), _integer(arguments, "bitsize")
        dtype = quantum_bit_type() if width == 1 else opaque_register_type(width)
        signature = (Register("val", dtype, side=side),)
        if isinstance(value, sympy.Integer):
            value = int(value)
        if type(value) is int and type(width) is int and not 0 <= value < 2**width:
            raise SerializationError("Integer basis value does not fit its width")
        if isinstance(width, sympy.Expr) or isinstance(value, sympy.Expr) or width == 0:
            result: FunctionRecord | ProgramRecord = FunctionRecord(
                name, (), registers=signature
            )
        else:
            result = integer_basis_program(name, dtype, value, state=state)
    else:
        if arguments.keys() - {"n"}:
            raise SerializationError("Bit basis constructor only accepts n")
        width = _integer({"n": arguments.get("n", 1)}, "n")
        if width != 1:
            raise NotImplementedError("Come back later.")
        dtype = quantum_bit_type()
        signature = (Register("q", dtype, side=side),)
        leaf = integer_basis_function(
            name, dtype, int(kind.startswith("One")), state=state
        )
        result = replace(leaf, registers=signature)
    if registers is not None and registers != signature:
        raise SerializationError("Invalid computational basis register signature")
    return result


def basis_vector_factories() -> dict[str, ComponentFactory]:
    """Provide six explicit constructor mappings without SDK or tensor execution.

    Effects use the existing checked clean-release promise and explicit input
    ownership. They never reset or discard an arbitrary unknown quantum input.
    Symbolic and empty integer declarations retain their source body boundaries.
    """
    kinds: tuple[_Kind, ...] = (
        "ZeroState",
        "ZeroEffect",
        "OneState",
        "OneEffect",
        "IntState",
        "IntEffect",
    )
    return {
        "qualtran.bloqs.basic_gates.z_basis." + kind: partial(_basis_factory, kind)
        for kind in kinds
    }
