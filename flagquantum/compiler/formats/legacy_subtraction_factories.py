# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Constructors adapted from locked Qualtran v0.7.0 arithmetic/subtraction.py.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/arithmetic/subtraction.py

"""Resolve subtraction libraries using existing integer declarations and graphs."""

from collections.abc import Mapping
from functools import partial

from ...core.ir.program import FunctionRecord, ProgramRecord
from ...core.ir.program.registers import Register, RegisterType
from ...errors import SerializationError
from ..registers.arithmetic.subtraction_declaration import (
    SubtractFromDeclaration,
    SubtractionDeclaration,
)
from .legacy_component_resolution import ComponentFactory, ResolvedArgument
from .legacy_word_factories import _rename_program


def _subtraction_factory(
    from_target: bool,
    name: str,
    registers: tuple[Register, ...] | None,
    arguments: Mapping[str, ResolvedArgument],
) -> FunctionRecord | ProgramRecord:
    allowed = {"dtype"} if from_target else {"a_dtype", "b_dtype"}
    key = "dtype" if from_target else "a_dtype"
    if arguments.keys() - allowed or key not in arguments:
        raise SerializationError("Invalid constructor arguments for subtraction")
    dtype = arguments[key]
    other = arguments.get("b_dtype", dtype)
    if not isinstance(dtype, RegisterType) or not isinstance(other, RegisterType):
        raise SerializationError("Subtraction requires register dtype metadata")
    declaration = (
        SubtractFromDeclaration(dtype)
        if from_target
        else SubtractionDeclaration(dtype, other)
    )
    signature = tuple(declaration.signature)
    if registers is not None and registers != signature:
        raise SerializationError("Invalid register signature for subtraction")
    if dtype.is_symbolic() or other.is_symbolic():
        return FunctionRecord(name, (), registers=signature)
    return _rename_program(declaration.program(), name)


def subtraction_factories() -> dict[str, ComponentFactory]:
    """Supply both constructors with the existing dtype and prefix/carry owners."""
    return {
        "qualtran.bloqs.arithmetic.subtraction.Subtract": partial(
            _subtraction_factory, False
        ),
        "qualtran.bloqs.arithmetic.subtraction.SubtractFrom": partial(
            _subtraction_factory, True
        ),
    }
