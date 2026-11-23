# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Constructor adapted from Qualtran v0.7.0 arithmetic/hamming_weight.py.

"""Bind Hamming metadata to existing compression widths and executable trees."""

from collections.abc import Mapping

import sympy

from ...core.ir.program import FunctionRecord, ProgramRecord
from ...core.ir.program.register_types import unsigned_register_type
from ...core.ir.program.registers import Register, RegisterSide, RegisterType
from ...errors import SerializationError
from ..registers.bits.hamming_weight import hamming_weight_resource_widths
from ..registers.bits.hamming_weight_component import hamming_weight_program
from .legacy_component_resolution import ComponentFactory, ResolvedArgument
from .legacy_modular_addition_factories import _integer
from .legacy_word_factories import _rename_program


def _hamming_weight_factory(
    name: str,
    registers: tuple[Register, ...] | None,
    arguments: Mapping[str, ResolvedArgument],
) -> FunctionRecord | ProgramRecord:
    if arguments.keys() != {"bitsize"}:
        raise SerializationError("HammingWeightCompute requires bitsize")
    width = _integer(arguments, "bitsize")
    junk, output = hamming_weight_resource_widths(width)
    ports = (
        Register("x", unsigned_register_type(width)),
        Register("junk", RegisterType("opaque", junk), side=RegisterSide.RIGHT),
        Register("out", unsigned_register_type(output), side=RegisterSide.RIGHT),
    )
    if registers is not None:
        # Serialized dimension symbols lose their original integer assumptions.
        normalized = (
            output.xreplace({sympy.floor(width): width})
            if isinstance(output, sympy.Expr)
            else output
        )
        integer_ports = (
            *ports[:-1],
            Register(
                "out", unsigned_register_type(normalized), side=RegisterSide.RIGHT
            ),
        )
        if registers not in (ports, integer_ports):
            raise SerializationError("Invalid HammingWeightCompute register signature")
        ports = registers
    if type(width) is not int or width == 0:
        return FunctionRecord(name, (), registers=ports)
    return _rename_program(hamming_weight_program(width), name)


def hamming_weight_factories() -> dict[str, ComponentFactory]:
    """Supply the source constructor with explicit logical junk ownership."""
    return {
        "qualtran.bloqs.arithmetic.hamming_weight.HammingWeightCompute": _hamming_weight_factory
    }
