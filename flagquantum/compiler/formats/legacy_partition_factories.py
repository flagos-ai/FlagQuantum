# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Constructors adapted from locked Qualtran v0.7.0 bookkeeping/partition.py.

"""Resolve source partition constructors through the existing storage-view owner."""

from collections.abc import Mapping
from functools import partial
from typing import Literal

import sympy

from ...core.ir.program import FunctionRecord, ProgramRecord
from ...core.ir.program.register_types import opaque_register_type
from ...core.ir.program.registers import Register
from ...core.value_math import SymbolicInt, ssum
from ...errors import SerializationError
from ..registers.layout.component_partition import (
    partition_component_program,
    partition_signature,
)
from .legacy_component_resolution import ComponentFactory, ResolvedArgument


def _width(arguments: Mapping[str, ResolvedArgument], key: str) -> SymbolicInt:
    value = arguments.get(key)
    if type(value) is not int and not isinstance(value, sympy.Expr):
        raise SerializationError(
            f"Partition {key} requires integer or symbolic metadata"
        )
    assert isinstance(value, (int, sympy.Expr))
    return value


def _partition_factory(
    kind: Literal["Partition", "Split2", "Join2"],
    name: str,
    registers: tuple[Register, ...] | None,
    arguments: Mapping[str, ResolvedArgument],
) -> FunctionRecord | ProgramRecord:
    if kind == "Partition":
        if (
            arguments.keys() - {"n", "regs", "partition"}
            or not {"n", "regs"} <= arguments.keys()
        ):
            raise SerializationError(
                "Partition requires n, regs and optional partition"
            )
        width = _width(arguments, "n")
        raw = arguments["regs"]
        if (
            not isinstance(raw, tuple)
            or not raw
            or not all(isinstance(part, Register) for part in raw)
        ):
            raise SerializationError(
                "Partition regs requires a nonempty register tuple"
            )
        parts = tuple(part for part in raw if isinstance(part, Register))
        if width != ssum(part.total_qubits() for part in parts) or len(
            {part.name for part in parts}
        ) != len(parts):
            raise SerializationError(
                "Partition width and distinct register names must agree"
            )
        reverse = not bool(arguments.get("partition", True))
    else:
        if set(arguments) != {"n1", "n2"}:
            raise SerializationError("Split2 and Join2 require n1 and n2")
        parts = tuple(
            Register(f"y{index}", opaque_register_type(_width(arguments, key)))
            for index, key in enumerate(("n1", "n2"), 1)
        )
        reverse = kind == "Join2"
    signature = tuple(partition_signature(parts, reverse=reverse))
    if registers is not None and registers != signature:
        raise SerializationError("Invalid register signature for partition")
    if any(part.is_symbolic() for part in parts):
        return FunctionRecord(name, (), registers=signature)
    return partition_component_program(parts, name, reverse)


def partition_factories() -> dict[str, ComponentFactory]:
    """Supply all three partition constructors with row-major MSB storage views."""
    kinds: tuple[Literal["Partition", "Split2", "Join2"], ...] = (
        "Partition",
        "Split2",
        "Join2",
    )
    return {
        "qualtran.bloqs.bookkeeping.partition."
        + kind: partial(_partition_factory, kind)
        for kind in kinds
    }
