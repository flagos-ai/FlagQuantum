# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Constructors adapted from Qualtran v0.7.0 arithmetic/permutation.py.

"""Bind literal permutation constructors through the existing cycle owners."""

from collections.abc import Mapping
from functools import partial
from typing import Literal

import numpy as np

from ...core.ir.program import FunctionRecord, ProgramRecord
from ...core.ir.program.registers import Register
from ...errors import SerializationError
from ..registers.sorting.permutation_declaration import (
    PermutationCycleDeclaration,
    PermutationDeclaration,
)
from .legacy_component_resolution import ComponentFactory, ResolvedArgument
from .legacy_modular_addition_factories import _integer
from .legacy_word_factories import _rename_program


def _literal_cycle(value: object) -> tuple[int, ...]:
    if isinstance(value, np.ndarray):
        if value.ndim != 1 or value.dtype.kind not in "iu":
            raise SerializationError("Permutation cycle requires an integer vector")
        value = value.tolist()
    if not isinstance(value, (tuple, list)) or any(
        type(entry) is not int for entry in value
    ):
        raise SerializationError("Permutation cycle requires literal integers")
    return tuple(value)


def _permutation_factory(
    kind: Literal["cycle", "permutation"],
    name: str,
    registers: tuple[Register, ...] | None,
    arguments: Mapping[str, ResolvedArgument],
) -> FunctionRecord | ProgramRecord:
    field = "cycle" if kind == "cycle" else "cycles"
    if not {"N", field} <= arguments.keys() or arguments.keys() - {
        "N",
        field,
        "bitsize",
    }:
        raise SerializationError("Invalid permutation constructor arguments")
    size = _integer(arguments, "N")
    width = _integer(arguments, "bitsize") if "bitsize" in arguments else None
    declaration: PermutationCycleDeclaration | PermutationDeclaration
    if kind == "cycle":
        declaration = PermutationCycleDeclaration(
            size, _literal_cycle(arguments[field]), width
        )
    else:
        cycles = arguments[field]
        if isinstance(cycles, np.ndarray):
            if cycles.dtype.kind not in "iu" or cycles.ndim != 2:
                # NumPy encodes an empty outer cycle sequence as shape (0,).
                if cycles.shape != (0,):
                    raise SerializationError("Permutation cycles require integer rows")
            cycles = cycles.tolist()
        if not isinstance(cycles, (tuple, list)):
            raise SerializationError("Permutation cycles require a literal sequence")
        declaration = PermutationDeclaration(
            size, tuple(_literal_cycle(cycle) for cycle in cycles), width
        )
    ports = tuple(declaration.signature)
    if registers is not None and registers != ports:
        raise SerializationError("Invalid permutation register signature")
    if type(size) is not int or type(declaration.width) is not int:
        return FunctionRecord(name, (), registers=ports)
    return _rename_program(declaration.program(), name)


def permutation_factories() -> dict[str, ComponentFactory]:
    """Supply both constructors for data admitted by the original array codec."""
    prefix = "qualtran.bloqs.arithmetic.permutation."
    return {
        prefix + "PermutationCycle": partial(_permutation_factory, "cycle"),
        prefix + "Permutation": partial(_permutation_factory, "permutation"),
    }
