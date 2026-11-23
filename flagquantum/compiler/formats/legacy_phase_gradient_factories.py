# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Constructors adapted from locked Qualtran v0.7.0 rotations/phase_gradient.py.

"""Resolve gradient phases, preparation and fractional addition metadata."""

from collections.abc import Mapping
from functools import partial
from typing import Literal

import sympy

from ...core.ir.program import FunctionRecord, ProgramRecord
from ...core.ir.program.registers import Register, RegisterType
from ...errors import SerializationError
from ..registers.phase.phase_addition import source_phase_add_signature
from ..registers.phase.phase_gradient import (
    source_phase_gradient_state_signature,
    source_phase_gradient_unitary_signature,
)
from ..registers.phase.phase_gradient_component import (
    phase_gradient_state_program,
    phase_gradient_unitary_program,
)
from ..registers.phase.scaled_phase_component import (
    phase_add_component_program,
    scaled_phase_add_program,
)
from .legacy_component_resolution import ComponentFactory, ResolvedArgument
from .legacy_modular_addition_factories import _integer
from .legacy_rotation_factories import _real_argument
from .legacy_word_factories import _rename_program
from .symbolic_parameters import symbolic_parameter_value

_GradientKind = Literal["PhaseGradientUnitary", "PhaseGradientState"]


def _gradient_factory(
    kind: _GradientKind,
    name: str,
    registers: tuple[Register, ...] | None,
    arguments: Mapping[str, ResolvedArgument],
) -> FunctionRecord | ProgramRecord:
    allowed = {"bitsize", "exponent", "eps"}
    if kind == "PhaseGradientUnitary":
        allowed.add("is_controlled")
    if "bitsize" not in arguments or arguments.keys() - allowed:
        raise SerializationError(
            "Phase gradient requires width and optional phase metadata"
        )
    width = _integer(arguments, "bitsize")
    exponent = _real_argument(
        arguments, "exponent", -1 if kind == "PhaseGradientState" else 1
    )
    _real_argument(arguments, "eps", 1e-10)
    controlled = arguments.get("is_controlled", False)
    if type(controlled) not in (bool, int) or controlled not in (0, 1):
        raise SerializationError("Gradient control mode requires a boolean")
    ports = tuple(
        source_phase_gradient_state_signature(width)
        if kind == "PhaseGradientState"
        else source_phase_gradient_unitary_signature(width, controlled=bool(controlled))
    )
    if registers is not None and registers != ports:
        raise SerializationError("Invalid phase gradient register signature")
    if type(width) is not int or width == 0:
        return FunctionRecord(name, (), registers=ports)
    angle = (
        symbolic_parameter_value(exponent)
        if isinstance(exponent, sympy.Expr)
        else exponent
    )
    program = (
        phase_gradient_state_program(width, angle)
        if kind == "PhaseGradientState"
        else phase_gradient_unitary_program(width, angle, controlled=bool(controlled))
    )
    return _rename_program(program, name)


def _phase_add_factory(
    name: str,
    registers: tuple[Register, ...] | None,
    arguments: Mapping[str, ResolvedArgument],
) -> FunctionRecord | ProgramRecord:
    if not {"x_bitsize", "phase_bitsize"} <= arguments.keys() or arguments.keys() - {
        "x_bitsize",
        "phase_bitsize",
        "right_shift",
        "sign",
        "controlled_by",
    }:
        raise SerializationError("Phase addition requires input and gradient widths")
    width, gradient = _integer(arguments, "x_bitsize"), _integer(
        arguments, "phase_bitsize"
    )
    shift = _integer({"right_shift": 0, **arguments}, "right_shift")
    sign = _integer({"sign": 1, **arguments}, "sign")
    control = arguments.get("controlled_by")
    if control is not None and (type(control) is not int or control not in (0, 1)):
        raise SerializationError("Phase addition control requires zero, one or None")
    ports = tuple(
        source_phase_add_signature(width, gradient, controlled=control is not None)
    )
    if registers is not None and registers != ports:
        raise SerializationError("Invalid phase addition register signature")
    if (
        any(type(v) is not int for v in (width, gradient, shift, sign))
        or gradient == 0
        or shift < 0
        or sign not in (-1, 1)
    ):
        return FunctionRecord(name, (), registers=ports)
    assert isinstance(width, int) and isinstance(gradient, int)
    assert isinstance(shift, int) and isinstance(sign, int)
    return _rename_program(
        phase_add_component_program(
            width,
            gradient,
            right_shift=shift,
            sign=sign,
            controlled=control is not None,
            control_value=1 if control is None else control,
        ),
        name,
    )


def _scaled_phase_factory(
    name: str,
    registers: tuple[Register, ...] | None,
    arguments: Mapping[str, ResolvedArgument],
) -> FunctionRecord | ProgramRecord:
    if arguments.keys() != {"x_dtype", "phase_bitsize", "gamma", "gamma_dtype"}:
        raise SerializationError(
            "Scaled phase addition requires both fixed-point formats"
        )
    x, gamma_dtype = arguments["x_dtype"], arguments["gamma_dtype"]
    if not isinstance(x, RegisterType) or not isinstance(gamma_dtype, RegisterType):
        raise SerializationError("Scaled phase formats require register descriptors")
    if x.encoding != "fixed_point" or gamma_dtype.encoding != "fixed_point":
        raise SerializationError("Scaled phase formats require fixed-point descriptors")
    gradient = _integer(arguments, "phase_bitsize")
    gamma = _real_argument(arguments, "gamma")
    fraction = dict(x.parameters)["fractional_width"]
    gamma_fraction = dict(gamma_dtype.parameters)["fractional_width"]
    ports = ((Register("x", x),) if x.quantum_bits != 0 else ()) + tuple(
        source_phase_add_signature(0, gradient)
    )
    if registers is not None and registers != ports:
        raise SerializationError("Invalid scaled phase addition register signature")
    if (
        x.is_symbolic()
        or gamma_dtype.is_symbolic()
        or type(gradient) is not int
        or isinstance(gamma, sympy.Expr)
        or dict(x.parameters)["signed"]
        or dict(gamma_dtype.parameters)["signed"]
    ):
        return FunctionRecord(name, (), registers=ports)
    assert isinstance(x.quantum_bits, int) and isinstance(gamma_dtype.quantum_bits, int)
    assert isinstance(fraction, int) and isinstance(gamma_fraction, int)
    return _rename_program(
        scaled_phase_add_program(
            x.quantum_bits,
            fraction,
            gradient,
            gamma,
            gamma_dtype.quantum_bits,
            gamma_fraction,
        ),
        name,
    )


def phase_gradient_factories() -> dict[str, ComponentFactory]:
    """Supply all four source constructors through existing native graph owners."""
    prefix = "qualtran.bloqs.rotations.phase_gradient."
    return {
        prefix
        + "PhaseGradientUnitary": partial(_gradient_factory, "PhaseGradientUnitary"),
        prefix + "PhaseGradientState": partial(_gradient_factory, "PhaseGradientState"),
        prefix + "AddIntoPhaseGrad": _phase_add_factory,
        prefix + "AddScaledValIntoPhaseReg": _scaled_phase_factory,
    }
