# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Constructors adapted from locked Qualtran v0.7.0 phase_estimation windows.

"""Convert stored phase-window declarations through existing algorithm owners."""

from collections.abc import Mapping
from functools import partial
from typing import Literal

from ..algorithms.primitives.kaiser_window import KaiserWindowState
from ..algorithms.primitives.phase_estimation_window import rectangular_window_program
from ..compiler.formats.legacy_component_resolution import (
    ComponentFactory,
    ResolvedArgument,
)
from ..compiler.formats.legacy_modular_addition_factories import _integer
from ..compiler.formats.legacy_rotation_factories import _real_argument
from ..compiler.formats.legacy_word_factories import _rename_program
from ..core.ir.program import FunctionRecord, ProgramRecord
from ..core.ir.program.registers import Register
from ..errors import SerializationError


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
        alpha = _real_argument(arguments, "alpha")
        ports = tuple(KaiserWindowState(width, alpha).signature)
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
