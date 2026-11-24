# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Constructors follow locked Qualtran v0.7.0 swap_network/cswap_approx.py.

"""Resolve relative exchanges while preserving exact generic inverse calls."""

from collections.abc import Mapping

from ...core.ir.program import FunctionRecord, ProgramRecord
from ...core.ir.program.registers import Register
from ...core.value_math import SymbolicInt
from ...errors import SerializationError
from ..registers.primitives.relative_controlled_swap import relative_swap_signature
from ..registers.primitives.relative_swap_component import relative_swap_program
from .legacy_component_resolution import ComponentFactory, ResolvedArgument
from .legacy_modular_addition_factories import _integer
from .legacy_word_factories import _rename_program


def relative_swap_factories(
    *, adjoint_factory: ComponentFactory | None = None
) -> dict[str, ComponentFactory]:
    """Own source widths and exact inverse recipes within one compilation.

    Symbols retain the existing logical declaration. The finite inverse uses
    specialized S/T inverse calls and source-ordered fanout adjoints rather
    than substituting the logically self-adjoint forward stream.
    """
    recipes: dict[FunctionRecord, tuple[SymbolicInt, bool]] = {}

    def make(
        name: str,
        registers: tuple[Register, ...] | None,
        width: SymbolicInt,
        inverse: bool,
    ) -> ProgramRecord:
        ports = tuple(relative_swap_signature(width))
        if registers is not None and registers != ports:
            raise SerializationError("Invalid relative swap register signature")
        record = (
            relative_swap_program(width, adjoint=inverse)
            if type(width) is int
            else ProgramRecord((FunctionRecord("main", (), registers=ports),))
        )
        record = _rename_program(record, name)
        recipes[record.functions[0]] = width, inverse
        return record

    def forward(
        name: str,
        registers: tuple[Register, ...] | None,
        arguments: Mapping[str, ResolvedArgument],
    ) -> ProgramRecord:
        if set(arguments) != {"bitsize"}:
            raise SerializationError("Relative swap requires only bitsize")
        return make(name, registers, _integer(arguments, "bitsize"), False)

    def adjoint(
        name: str,
        registers: tuple[Register, ...] | None,
        arguments: Mapping[str, ResolvedArgument],
    ) -> FunctionRecord | ProgramRecord:
        child = arguments.get("subbloq")
        if (
            set(arguments) == {"subbloq"}
            and isinstance(child, FunctionRecord)
            and child in recipes
        ):
            width, inverse = recipes[child]
            return make(name, registers, width, not inverse)
        if adjoint_factory is not None:
            return adjoint_factory(name, registers, arguments)
        raise SerializationError("Missing relative swap adjoint constructor context")

    return {
        "qualtran.bloqs.swap_network.cswap_approx.CSwapApprox": forward,
        "qualtran._infra.adjoint.Adjoint": adjoint,
    }
