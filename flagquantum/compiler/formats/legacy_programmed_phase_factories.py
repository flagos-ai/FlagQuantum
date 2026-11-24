# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Constructor adaptation from Qualtran v0.7.0 rotations/programmable_ancilla_rotation.py.

"""Resolve prepared phase states and retain the source adaptive model boundary."""

from collections.abc import Mapping
from functools import partial

import sympy

from ...core.ir.program import FunctionRecord, ProgramRecord
from ...core.ir.program.register_types import quantum_bit_type
from ...core.ir.program.registers import Register, RegisterSide
from ...errors import SerializationError
from ..registers.phase.programmed_phase_component import programmed_phase_program
from .legacy_component_resolution import ComponentFactory, ResolvedArgument
from .legacy_modular_addition_factories import _integer
from .legacy_rotation_factories import _real_argument
from .legacy_word_factories import _rename_program
from .symbolic_parameters import symbolic_parameter_value


def _programmed_factory(
    adaptive: bool,
    name: str,
    registers: tuple[Register, ...] | None,
    arguments: Mapping[str, ResolvedArgument],
) -> FunctionRecord | ProgramRecord:
    allowed = {"exponent", "eps"}
    if adaptive:
        allowed |= {"n_rounds", "apply_final_correction"}
    if "exponent" not in arguments or arguments.keys() - allowed:
        raise SerializationError("Invalid programmed phase constructor arguments")
    exponent = _real_argument(arguments, "exponent")
    _real_argument(arguments, "eps", 1e-11)
    if adaptive:
        _integer({"n_rounds": arguments.get("n_rounds", 2)}, "n_rounds")
        correction = arguments.get("apply_final_correction", False)
        if type(correction) not in (int, bool) or correction not in (0, 1):
            raise SerializationError("Programmed correction requires a boolean")
    ports = (
        Register(
            "q",
            quantum_bit_type(),
            side=RegisterSide.THRU if adaptive else RegisterSide.RIGHT,
        ),
    )
    if registers is not None and registers != ports:
        raise SerializationError("Invalid programmed phase register signature")
    if adaptive:
        return FunctionRecord(name, (), registers=ports)
    angle = (
        symbolic_parameter_value(exponent)
        if isinstance(exponent, sympy.Expr)
        else exponent
    )
    return _rename_program(programmed_phase_program(angle), name)


def programmed_phase_factories() -> dict[str, ComponentFactory]:
    """Supply both complete original constructor contracts and direct owners.

    The adaptive original has a worst-case resource model and no gate body.
    Its declaration preserves that boundary. The separately certified native
    adaptive channel is not substituted for a missing source decomposition.
    """
    prefix = "qualtran.bloqs.rotations.programmable_ancilla_rotation."
    return {
        prefix + "ZPowProgrammedAncilla": partial(_programmed_factory, False),
        prefix + "ZPowUsingProgrammedAncilla": partial(_programmed_factory, True),
    }
