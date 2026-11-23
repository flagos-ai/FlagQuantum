# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Constructor adapted from locked Qualtran v0.7.0 qft/approximate_qft.py.

"""Resolve precision-limited Fourier metadata through native phase addition."""

from collections.abc import Mapping

from ...core.ir.program import FunctionRecord, ProgramRecord
from ...core.ir.program.register_types import (
    fixed_point_register_type,
    unsigned_register_type,
)
from ...core.ir.program.registers import Register
from ...core.value_math import ceil, log2
from ...errors import SerializationError
from ..registers.phase.approximate_qft_component import approximate_qft_program
from .legacy_component_resolution import ComponentFactory, ResolvedArgument
from .legacy_modular_addition_factories import _integer
from .legacy_word_factories import _rename_program


def _approximate_qft_factory(
    name: str,
    registers: tuple[Register, ...] | None,
    arguments: Mapping[str, ResolvedArgument],
) -> FunctionRecord | ProgramRecord:
    if "bitsize" not in arguments or arguments.keys() - {
        "bitsize",
        "phase_bitsize",
        "with_reverse",
    }:
        raise SerializationError(
            "Approximate QFT requires data and optional phase widths"
        )
    width = _integer(arguments, "bitsize")
    precision = (
        _integer(arguments, "phase_bitsize")
        if "phase_bitsize" in arguments
        else ceil(log2(width))
    )
    reverse = arguments.get("with_reverse", True)
    if type(reverse) not in (bool, int) or reverse not in (0, 1):
        raise SerializationError("Approximate QFT reversal requires a boolean")
    ports: tuple[Register, ...] = (Register("q", unsigned_register_type(width)),)
    if precision != 0:
        ports += (
            Register("phase_grad", fixed_point_register_type(precision, precision)),
        )
    if registers is not None and registers != ports:
        raise SerializationError("Invalid approximate QFT register signature")
    if type(width) is not int or type(precision) is not int or width == 0:
        return FunctionRecord(name, (), registers=ports)
    if precision == 0 and width > 1:
        return FunctionRecord(name, (), registers=ports)
    return _rename_program(
        approximate_qft_program(width, precision, with_reverse=bool(reverse)), name
    )


def approximate_qft_factories() -> dict[str, ComponentFactory]:
    """Supply stored Fourier construction with its controlled-addition helpers."""
    return {
        "qualtran.bloqs.qft.approximate_qft.ApproximateQFT": _approximate_qft_factory
    }
