# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Constructors adapted from Qualtran v0.7.0 arithmetic/sorting.py.

"""Bind sorting constructors to the authoritative comparison network owners."""

from collections.abc import Mapping
from functools import partial
from typing import Literal

import sympy

from ...core.ir.program import FunctionRecord, ProgramRecord
from ...core.ir.program.registers import Register
from ...core.value_math import SymbolicInt
from ...errors import SerializationError
from ..registers.comparison.ordering_declaration import (
    ComparatorDeclaration,
    ParallelComparatorsDeclaration,
)
from ..registers.sorting.bitonic_declaration import (
    BitonicMergeDeclaration,
    BitonicSortDeclaration,
)
from .legacy_component_resolution import ComponentFactory, ResolvedArgument
from .legacy_modular_addition_factories import _integer
from .legacy_word_factories import _rename_program

_Kind = Literal["Comparator", "ParallelComparators", "BitonicMerge", "BitonicSort"]


def _integer_dimension_ports(
    ports: tuple[Register, ...], dimensions: tuple[SymbolicInt, ...]
) -> tuple[Register, ...]:
    # Original serialization loses integer assumptions on dimension symbols.
    replacements = {sympy.floor(value): value for value in dimensions}
    return tuple(
        Register(
            port.name,
            port.dtype,
            tuple(
                value.xreplace(replacements) if isinstance(value, sympy.Expr) else value
                for value in port.shape_symbolic
            ),
            side=port.side,
        )
        for port in ports
    )


def _sorting_factory(
    kind: _Kind,
    name: str,
    registers: tuple[Register, ...] | None,
    arguments: Mapping[str, ResolvedArgument],
) -> FunctionRecord | ProgramRecord:
    fields = {
        "Comparator": ("bitsize",),
        "ParallelComparators": ("k", "offset", "bitsize"),
        "BitonicMerge": ("half_length", "bitsize"),
        "BitonicSort": ("k", "bitsize"),
    }[kind]
    if arguments.keys() != set(fields):
        raise SerializationError(f"Invalid {kind} constructor arguments")
    dimensions = tuple(_integer(arguments, field) for field in fields)
    declaration = {
        "Comparator": ComparatorDeclaration,
        "ParallelComparators": ParallelComparatorsDeclaration,
        "BitonicMerge": BitonicMergeDeclaration,
        "BitonicSort": BitonicSortDeclaration,
    }[kind](*dimensions)
    ports = tuple(declaration.signature)
    if registers is not None:
        if _integer_dimension_ports(registers, dimensions) != _integer_dimension_ports(
            ports, dimensions
        ):
            raise SerializationError(f"Invalid {kind} register signature")
        ports = registers
    if any(type(value) is not int for value in dimensions):
        return FunctionRecord(name, (), registers=ports)
    # The source declares these domains but cannot supply their physical bodies.
    if dimensions[-1] == 0 or (
        kind == "BitonicMerge" and dimensions[0] & (dimensions[0] - 1)
    ):
        return FunctionRecord(name, (), registers=ports)
    return _rename_program(declaration.program(), name)


def sorting_factories() -> dict[str, ComponentFactory]:
    """Supply all four source sorting constructors with owned history outputs."""
    kinds: tuple[_Kind, ...] = (
        "Comparator",
        "ParallelComparators",
        "BitonicMerge",
        "BitonicSort",
    )
    return {
        "qualtran.bloqs.arithmetic.sorting." + kind: partial(_sorting_factory, kind)
        for kind in kinds
    }
