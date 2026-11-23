# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Constructor adapted from Qualtran v0.7.0 arithmetic/lists/sort_in_place.py.

"""Retain the source in-place-sort declaration independently of native synthesis."""

from collections.abc import Mapping

from ...core.ir.program import FunctionRecord
from ...core.ir.program.registers import Register, RegisterType
from ...errors import SerializationError
from ..registers.sorting.in_place_sort_declaration import in_place_sort_signature
from .legacy_component_resolution import ComponentFactory, ResolvedArgument
from .legacy_modular_addition_factories import _integer


def _in_place_sort_factory(
    name: str,
    registers: tuple[Register, ...] | None,
    arguments: Mapping[str, ResolvedArgument],
) -> FunctionRecord:
    if arguments.keys() != {"l", "dtype"}:
        raise SerializationError("SortInPlace requires length and dtype")
    dtype = arguments["dtype"]
    if not isinstance(dtype, RegisterType):
        raise SerializationError("SortInPlace requires a register dtype")
    ports = tuple(in_place_sort_signature(_integer(arguments, "l"), dtype))
    if registers is not None and registers != ports:
        raise SerializationError("Invalid SortInPlace register signature")
    # The frozen source has a call-count model and no decomposition body.
    return FunctionRecord(name, (), registers=ports)


def in_place_sort_factories() -> dict[str, ComponentFactory]:
    """Supply the original sort declaration using the native signature owner."""
    return {
        "qualtran.bloqs.arithmetic.lists.sort_in_place.SortInPlace": _in_place_sort_factory
    }
