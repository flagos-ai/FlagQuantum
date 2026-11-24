# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Constructor adaptation from Qualtran v0.7.0 basic_gates/global_phase.py.

"""Bind zero-port scalar phases to the existing explicit carrier recipe."""

from collections.abc import Mapping
from math import pi

import sympy

from ...core.ir.program import ProgramRecord
from ...core.ir.program.registers import Register
from ...errors import SerializationError
from ..registers.primitives.global_phase import _finite_angle
from ..registers.primitives.global_phase_component import scalar_phase_program
from .legacy_component_resolution import ComponentFactory, ResolvedArgument
from .legacy_rotation_factories import _real_argument
from .legacy_word_factories import _rename_program
from .symbolic_parameters import symbolic_parameter_value


def _global_phase_factory(
    name: str,
    registers: tuple[Register, ...] | None,
    arguments: Mapping[str, ResolvedArgument],
) -> ProgramRecord:
    if arguments.keys() - {"exponent", "eps"} or "exponent" not in arguments:
        raise SerializationError("GlobalPhase requires exponent and optional eps")
    if registers is not None and registers != ():
        raise SerializationError("GlobalPhase requires an empty register signature")
    exponent = _real_argument(arguments, "exponent")
    _real_argument(arguments, "eps", 1e-11)
    angle = (
        symbolic_parameter_value(sympy.pi * exponent)
        if isinstance(exponent, sympy.Expr)
        else _finite_angle(exponent, pi)
    )
    return _rename_program(scalar_phase_program(angle), name)


def scalar_phase_factories() -> dict[str, ComponentFactory]:
    """Resolve atomic GlobalPhase with zero logical ports and owned clean work.

    The source has no primitive decomposition. Its accepted scalar carrier
    recipe supplies exact complex phase while preserving the original zero-cost
    analytical model as separate evidence. Positive source control resolves to
    the existing ZPowGate constructor.
    """
    return {
        "qualtran.bloqs.basic_gates.global_phase.GlobalPhase": _global_phase_factory
    }
