# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Constructors adapted from locked Qualtran v0.7.0 rotations/*_via_phase_gradient.py.

"""Resolve register-angle and constant kickback through shared native graphs."""

from collections.abc import Mapping

import sympy

from ...core.ir.program import FunctionRecord, ProgramRecord
from ...core.ir.program.register_types import (
    fixed_point_register_type,
    quantum_bit_type,
)
from ...core.ir.program.registers import Register, RegisterType
from ...core.ir.program.signatures import Signature
from ...errors import SerializationError
from ..registers.phase.rz_phase_gradient_component import rz_via_phase_gradient_program
from ..registers.phase.zpow_phase_gradient_component import (
    zpow_via_phase_gradient_program,
)
from .legacy_component_resolution import ComponentFactory, ResolvedArgument
from .legacy_modular_addition_factories import _integer
from .legacy_rotation_factories import _real_argument
from .legacy_word_factories import _rename_program


def _rz_factory(
    name: str,
    registers: tuple[Register, ...] | None,
    arguments: Mapping[str, ResolvedArgument],
) -> FunctionRecord | ProgramRecord:
    if arguments.keys() != {"angle_dtype", "phasegrad_dtype"}:
        raise SerializationError("Gradient Rz requires angle and gradient descriptors")
    angle, gradient = arguments["angle_dtype"], arguments["phasegrad_dtype"]
    if not isinstance(angle, RegisterType) or not isinstance(gradient, RegisterType):
        raise SerializationError("Gradient Rz requires register descriptors")
    ports = tuple(
        Signature.build_from_dtypes(
            q=quantum_bit_type(), angle=angle, phase_grad=gradient
        )
    )
    if registers is not None and registers != ports:
        raise SerializationError("Invalid gradient Rz register signature")
    if angle.quantum_bits == 0 or gradient.quantum_bits == 0:
        return FunctionRecord(name, (), registers=ports)
    return _rename_program(rz_via_phase_gradient_program(angle, gradient), name)


def _zpow_factory(
    name: str,
    registers: tuple[Register, ...] | None,
    arguments: Mapping[str, ResolvedArgument],
) -> FunctionRecord | ProgramRecord:
    if arguments.keys() != {"exponent", "phase_grad_bitsize"}:
        raise SerializationError(
            "Gradient Z power requires exponent and gradient width"
        )
    exponent = _real_argument(arguments, "exponent")
    width = _integer(arguments, "phase_grad_bitsize")
    ports = tuple(
        Signature.build_from_dtypes(
            q=quantum_bit_type(), phase_grad=fixed_point_register_type(width, width)
        )
    )
    if registers is not None and registers != ports:
        raise SerializationError("Invalid gradient Z power register signature")
    if isinstance(exponent, sympy.Expr) or width == 0:
        return FunctionRecord(name, (), registers=ports)
    return _rename_program(zpow_via_phase_gradient_program(exponent, width), name)


def phase_rotation_factories() -> dict[str, ComponentFactory]:
    """Supply both original constructors with complete existing helper closure."""
    prefix = "qualtran.bloqs.rotations."
    return {
        prefix + "rz_via_phase_gradient.RzViaPhaseGradient": _rz_factory,
        prefix + "zpow_via_phase_gradient.ZPowConstViaPhaseGradient": _zpow_factory,
    }
