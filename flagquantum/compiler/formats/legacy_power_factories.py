# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Constructor adaptation from Qualtran v0.7.0 basic_gates/power.py.

"""Resolve literal repetition through the existing typed component owner."""

from collections.abc import Mapping
from dataclasses import replace

import sympy

from ...core.ir.program import FunctionRecord
from ...core.ir.program.registers import Register, RegisterSide
from ...errors import SerializationError
from ..registers.component_power import power_word_program
from .legacy_component_resolution import ComponentFactory, ResolvedArgument


def _power_factory(
    name: str,
    registers: tuple[Register, ...] | None,
    arguments: Mapping[str, ResolvedArgument],
) -> FunctionRecord:
    if set(arguments) != {"bloq", "power"}:
        raise SerializationError("Power requires only bloq and power")
    child, repetitions = arguments["bloq"], arguments["power"]
    if (
        not isinstance(child, FunctionRecord)
        or child.registers is None
        or any(register.side != RegisterSide.THRU for register in child.registers)
    ):
        raise SerializationError(
            "Power requires a typed child with only THRU registers"
        )
    if registers is not None and registers != child.registers:
        raise SerializationError("Invalid register signature for Power")
    if isinstance(repetitions, sympy.Expr):
        return FunctionRecord(name, (), registers=child.registers)
    if type(repetitions) is not int or repetitions < 1:
        raise SerializationError("Power requires a positive integer repetition count")
    # The enclosing resolver owns the real child body and its helper closure.
    declaration = replace(child, blocks=())
    root = power_word_program(declaration, repetitions).functions[0]
    return replace(root, name=name)


def power_factories() -> dict[str, ComponentFactory]:
    """Resolve Power against independently resolved THRU children and helpers."""
    return {"qualtran.bloqs.basic_gates.power.Power": _power_factory}
