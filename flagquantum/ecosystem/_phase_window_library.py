# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Constructors follow locked Qualtran v0.7.0 phase_estimation/lp_resource_state.py.

"""Translate phase-window constructors through their existing native owners."""

from collections.abc import Mapping
from functools import partial
from typing import Literal

from ..algorithms.primitives.luis_perina_state import (
    LuisPerinaInterim,
    LuisPerinaState,
    luis_perina_interim_program,
    luis_perina_state_program,
)
from ..compiler.formats.legacy_component_resolution import (
    ComponentFactory,
    ResolvedArgument,
)
from ..compiler.formats.legacy_modular_addition_factories import _integer
from ..compiler.formats.legacy_phase_window_factories import (
    phase_window_library_factories,
)
from ..compiler.formats.legacy_rotation_factories import _real_argument
from ..compiler.formats.legacy_word_factories import _rename_program
from ..core.ir.program import FunctionRecord, ProgramRecord
from ..core.ir.program.registers import Register
from ..errors import SerializationError


def _luis_perina_factory(
    kind: Literal["interim", "window"],
    name: str,
    registers: tuple[Register, ...] | None,
    arguments: Mapping[str, ResolvedArgument],
) -> FunctionRecord | ProgramRecord:
    required = {"bitsize"}
    allowed = required | ({"eps"} if kind == "interim" else set())
    if not required <= arguments.keys() or arguments.keys() - allowed:
        raise SerializationError("Invalid Luis-Perina constructor arguments")
    width = _integer(arguments, "bitsize")
    if kind == "interim":
        _real_argument(arguments, "eps", 1e-11)
    signature = (
        LuisPerinaInterim(width).signature
        if kind == "interim"
        else LuisPerinaState(width).signature
    )
    if registers is not None and registers != tuple(signature):
        raise SerializationError("Invalid Luis-Perina register signature")
    if not isinstance(width, int):
        # The source stores the symbolic declaration but cannot build its body.
        return FunctionRecord(name, (), registers=tuple(signature))
    program = (
        luis_perina_interim_program(width)
        if kind == "interim"
        else luis_perina_state_program(width)
    )
    return _rename_program(program, name)


def phase_window_factories() -> dict[str, ComponentFactory]:
    """Bind stored windows without moving Algorithms composition into Compiler."""
    prefix = "qualtran.bloqs.phase_estimation.lp_resource_state."
    return {
        **phase_window_library_factories(),
        prefix + "LPRSInterimPrep": partial(_luis_perina_factory, "interim"),
        prefix + "LPResourceState": partial(_luis_perina_factory, "window"),
    }
