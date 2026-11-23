# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Constructor adapted from Qualtran v0.7.0
# arithmetic/conversions/ones_complement_to_twos_complement.py.

"""Preserve the original signed conversion declaration and model-only boundary."""

from collections.abc import Mapping

from ...core.ir.program import FunctionRecord
from ...core.ir.program.register_types import (
    ones_complement_register_type,
    signed_register_type,
)
from ...core.ir.program.registers import Register, RegisterSide
from ...errors import SerializationError
from .legacy_component_resolution import ComponentFactory, ResolvedArgument
from .legacy_modular_addition_factories import _integer


def _signed_encoding_factory(
    name: str,
    registers: tuple[Register, ...] | None,
    arguments: Mapping[str, ResolvedArgument],
) -> FunctionRecord:
    if arguments.keys() != {"bitsize"}:
        raise SerializationError("Signed encoding conversion requires only bitsize")
    width = _integer(arguments, "bitsize")
    ports = (
        Register("x", ones_complement_register_type(width), side=RegisterSide.LEFT),
        Register("y", signed_register_type(width), side=RegisterSide.RIGHT),
    )
    if registers is not None and registers != ports:
        raise SerializationError(
            "Invalid signed encoding conversion register signature"
        )
    return FunctionRecord(name, (), registers=ports)


def signed_encoding_factories() -> dict[str, ComponentFactory]:
    """Supply the original constructor without inventing a source gate body."""
    return {
        "qualtran.bloqs.arithmetic.conversions.ones_complement_to_twos_complement.SignedIntegerToTwosComplement": _signed_encoding_factory
    }
