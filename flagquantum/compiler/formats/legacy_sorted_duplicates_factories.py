# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Constructor adapted from Qualtran v0.7.0 arithmetic/lists/has_duplicates.py.

"""Resolve duplicate constructors through the literal source graph owner."""

from collections.abc import Mapping

from ...core.ir.program import FunctionRecord, ProgramRecord
from ...core.ir.program.registers import Register, RegisterType
from ...errors import SerializationError
from ..registers.sorting.sorted_duplicates import sorted_duplicates_signature
from ..registers.sorting.sorted_duplicates_component import sorted_duplicates_program
from .legacy_component_resolution import ComponentFactory, ResolvedArgument
from .legacy_modular_addition_factories import _integer
from .legacy_word_factories import _rename_program


def _sorted_duplicates_factory(
    name: str,
    registers: tuple[Register, ...] | None,
    arguments: Mapping[str, ResolvedArgument],
) -> FunctionRecord | ProgramRecord:
    if not {"l", "dtype"} <= arguments.keys() or arguments.keys() - {
        "l",
        "dtype",
        "is_controlled",
    }:
        raise SerializationError(
            "HasDuplicates requires length, dtype and optional control"
        )
    length = _integer(arguments, "l")
    dtype = arguments["dtype"]
    if not isinstance(dtype, RegisterType):
        raise SerializationError("HasDuplicates requires a register dtype")
    control = arguments.get("is_controlled", False)
    if type(control) not in (bool, int) or control not in (0, 1):
        raise SerializationError("HasDuplicates control requires a boolean")
    controlled = bool(control)
    ports = tuple(sorted_duplicates_signature(length, dtype, controlled=controlled))
    if registers is not None and registers != ports:
        raise SerializationError("Invalid HasDuplicates register signature")
    if (
        type(length) is not int
        or type(dtype.num_qubits) is not int
        or dtype.encoding not in {"uint", "signed"}
        or (length <= 1 and not controlled)
        or (length > 1 and dtype.num_qubits == 0 and dtype.encoding == "signed")
    ):
        return FunctionRecord(name, (), registers=ports)
    return _rename_program(
        sorted_duplicates_program(length, dtype, controlled=controlled), name
    )


def sorted_duplicates_factories() -> dict[str, ComponentFactory]:
    """Supply the constructor while preserving source reverse-half aliases."""
    return {
        "qualtran.bloqs.arithmetic.lists.has_duplicates.HasDuplicates": _sorted_duplicates_factory
    }
