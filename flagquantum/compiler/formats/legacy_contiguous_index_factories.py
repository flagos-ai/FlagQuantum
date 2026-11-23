# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Constructor adaptation from Qualtran v0.7.0 conversions/contiguous_index.py.

"""Preserve the original triangular-index declaration and its no-body boundary."""

from collections.abc import Mapping

from ...core.ir.program import FunctionRecord
from ...core.ir.program.register_types import unsigned_register_type
from ...core.ir.program.registers import Register
from ...errors import SerializationError
from .legacy_component_resolution import ComponentFactory, ResolvedArgument
from .legacy_modular_addition_factories import _integer


def _index_factory(
    name: str,
    registers: tuple[Register, ...] | None,
    arguments: Mapping[str, ResolvedArgument],
) -> FunctionRecord:
    if arguments.keys() != {"bitsize", "s_bitsize"}:
        raise SerializationError("Invalid contiguous-index constructor arguments")
    width = _integer(arguments, "bitsize")
    result_width = _integer(arguments, "s_bitsize")
    dtype = unsigned_register_type(width)
    ports = (
        Register("mu", dtype),
        Register("nu", dtype),
        Register("s", unsigned_register_type(result_width)),
    )
    if registers is not None and registers != ports:
        raise SerializationError("Invalid contiguous-index register signature")
    return FunctionRecord(name, (), registers=ports)


def contiguous_index_factories() -> dict[str, ComponentFactory]:
    """Supply the complete original contiguous-index constructor declaration."""
    return {
        "qualtran.bloqs.arithmetic.conversions.contiguous_index.ToContiguousIndex": _index_factory
    }
