# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Constructors adapted from locked Qualtran v0.7.0 phase_estimation windows.

"""Convert stored phase-window declarations through shared Compiler owners."""

from collections.abc import Mapping
from functools import partial
from typing import Literal

from ...core.ir.program import FunctionRecord, ProgramRecord
from ...core.ir.program.register_types import fixed_point_register_type
from ...core.ir.program.registers import Register, RegisterSide
from ...errors import SerializationError
from ..registers.phase.window_component import rectangular_window_program
from .legacy_component_resolution import (
    ComponentFactory,
    ResolvedArgument,
)
from .legacy_modular_addition_factories import _integer
from .legacy_rotation_factories import _real_argument
from .legacy_word_factories import _rename_program


def _window_factory(
    kind: Literal["rectangular", "kaiser"],
    name: str,
    registers: tuple[Register, ...] | None,
    arguments: Mapping[str, ResolvedArgument],
) -> FunctionRecord | ProgramRecord:
    required = {"bitsize", "alpha"} if kind == "kaiser" else {"bitsize"}
    if arguments.keys() != required:
        raise SerializationError(
            "Phase window requires its original constructor fields"
        )
    width = _integer(arguments, "bitsize")
    if kind == "kaiser":
        _real_argument(arguments, "alpha")
        ports = (
            Register(
                "qpe_reg",
                fixed_point_register_type(width, width),
                side=RegisterSide.RIGHT,
            ),
        )
        if registers is not None and registers != ports:
            raise SerializationError("Invalid Kaiser window register signature")
        # The frozen source supplies a tensor and model, without a gate body.
        return FunctionRecord(name, (), registers=ports)
    program = rectangular_window_program(width)
    if registers is not None and registers != program.functions[0].registers:
        raise SerializationError("Invalid rectangular window register signature")
    return _rename_program(program, name)


def phase_window_library_factories() -> dict[str, ComponentFactory]:
    """Supply rectangular preparation and the exact Kaiser tensor declaration."""
    return {
        "qualtran.bloqs.phase_estimation.qpe_window_state.RectangularWindowState": partial(
            _window_factory, "rectangular"
        ),
        "qualtran.bloqs.phase_estimation.kaiser_window_state.KaiserWindowState": partial(
            _window_factory, "kaiser"
        ),
    }
