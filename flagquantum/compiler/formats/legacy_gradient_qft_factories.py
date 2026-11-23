# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Constructor adapted from locked Qualtran v0.7.0 qft/qft_phase_gradient.py.

"""Resolve recursive gradient Fourier metadata through its native component."""

from collections.abc import Mapping

from ...core.ir.program import FunctionRecord, ProgramRecord
from ...core.ir.program.register_types import (
    fixed_point_register_type,
    unsigned_register_type,
)
from ...core.ir.program.registers import Register
from ...errors import SerializationError
from ..registers.phase.gradient_qft_component import gradient_qft_program
from .legacy_component_resolution import ComponentFactory, ResolvedArgument
from .legacy_modular_addition_factories import _integer
from .legacy_word_factories import _rename_program


def _gradient_qft_factory(
    name: str,
    registers: tuple[Register, ...] | None,
    arguments: Mapping[str, ResolvedArgument],
) -> FunctionRecord | ProgramRecord:
    if "bitsize" not in arguments or arguments.keys() - {"bitsize", "with_reverse"}:
        raise SerializationError("Gradient QFT requires bitsize and optional reversal")
    width = _integer(arguments, "bitsize")
    reverse = arguments.get("with_reverse", True)
    if type(reverse) not in (bool, int) or reverse not in (0, 1):
        raise SerializationError("Gradient QFT reversal requires a boolean")
    ports = (
        Register("q", unsigned_register_type(width)),
        Register("phase_grad", fixed_point_register_type(width, width)),
    )
    if registers is not None and registers != ports:
        raise SerializationError("Invalid gradient QFT register signature")
    if type(width) is not int or width == 0:
        return FunctionRecord(name, (), registers=ports)
    return _rename_program(
        gradient_qft_program(width, with_reverse=bool(reverse)), name
    )


def gradient_qft_factories() -> dict[str, ComponentFactory]:
    """Supply recursive Fourier construction and the full product helper closure."""
    return {
        "qualtran.bloqs.qft.qft_phase_gradient.QFTPhaseGradient": _gradient_qft_factory
    }
