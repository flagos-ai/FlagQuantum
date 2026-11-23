# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Constructor adapted from Qualtran v0.7.0 arithmetic/lists/symmetric_difference.py.

"""Retain original symmetric-difference declarations and independent model costs."""

from collections.abc import Mapping

from ...core.ir.program import FunctionRecord
from ...core.ir.program.registers import Register, RegisterType
from ...errors import SerializationError
from ..registers.sorting.symmetric_difference_declaration import (
    symmetric_difference_signature,
)
from .legacy_component_resolution import ComponentFactory, ResolvedArgument
from .legacy_modular_addition_factories import _integer


def _symmetric_difference_factory(
    name: str,
    registers: tuple[Register, ...] | None,
    arguments: Mapping[str, ResolvedArgument],
) -> FunctionRecord:
    if arguments.keys() != {"n_lhs", "n_rhs", "n_diff", "dtype"}:
        raise SerializationError("SymmetricDifference requires three sizes and dtype")
    dtype = arguments["dtype"]
    if not isinstance(dtype, RegisterType):
        raise SerializationError("SymmetricDifference requires a register dtype")
    ports = tuple(
        symmetric_difference_signature(
            _integer(arguments, "n_lhs"),
            _integer(arguments, "n_rhs"),
            _integer(arguments, "n_diff"),
            dtype,
        )
    )
    if registers is not None and registers != ports:
        raise SerializationError("Invalid SymmetricDifference register signature")
    # The frozen root declares child counts without an executable body.
    return FunctionRecord(name, (), registers=ports)


def symmetric_difference_factories() -> dict[str, ComponentFactory]:
    """Supply source set-oracle declarations through their existing signature."""
    return {
        "qualtran.bloqs.arithmetic.lists.symmetric_difference.SymmetricDifference": _symmetric_difference_factory
    }
