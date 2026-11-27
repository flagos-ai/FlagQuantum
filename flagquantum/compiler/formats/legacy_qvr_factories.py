# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Constructors adapted from locked Qualtran v0.7.0 rotations/quantum_variable_rotation.py.

"""Resolve typed quantum variable rotations through existing phase owners."""

from collections.abc import Mapping
from functools import partial

import sympy

from ...core.ir.program import FunctionRecord, ProgramRecord
from ...core.ir.program.register_types import fixed_point_register_type
from ...core.ir.program.registers import Register, RegisterSide
from ...errors import SerializationError
from ..registers.phase.qvr_component import (
    qvr_phase_addition_program,
    qvr_phase_gradient_program,
    qvr_phase_power_program,
    qvr_zpow_program,
)
from ..registers.phase.qvr_phase_gradient import (
    qvr_phase_format,
    source_symbolic_qvr_gradient_cost,
)
from ..registers.phase.qvr_zpow import (
    _fractional_rotations,
    symbolic_qvr_zpow_schedule,
)
from .legacy_component_resolution import ComponentFactory, ResolvedArgument
from .legacy_rotation_factories import _real_argument
from .legacy_word_factories import _rename_program
from .symbolic_parameters import symbolic_parameter_value


def _qvr_factory(
    gradient: bool,
    name: str,
    registers: tuple[Register, ...] | None,
    arguments: Mapping[str, ResolvedArgument],
) -> FunctionRecord | ProgramRecord:
    if "cost_reg" not in arguments or arguments.keys() - {"cost_reg", "gamma", "eps"}:
        raise SerializationError(
            "QVR requires a cost register and optional phase metadata"
        )
    cost = arguments["cost_reg"]
    if not isinstance(cost, Register) or cost.dtype.encoding != "fixed_point":
        raise SerializationError("QVR requires a fixed-point cost register")
    gamma = _real_argument(arguments, "gamma", 1)
    epsilon = _real_argument(arguments, "eps", 1e-9)
    parameters = dict(cost.dtype.parameters)
    fraction, signed = parameters["fractional_width"], parameters["signed"]
    width = cost.bitsize
    ports: tuple[Register, ...] = (cost,)
    if gradient:
        if signed:
            raise SerializationError("Gradient QVR requires an unsigned cost register")
        if isinstance(gamma, sympy.Expr):
            assert isinstance(fraction, (int, sympy.Expr))
            sizing = source_symbolic_qvr_gradient_cost(width, fraction, gamma, epsilon)
            phase_width = sizing.logical_qubits - width
        else:
            if (
                type(width) is not int
                or type(fraction) is not int
                or isinstance(epsilon, sympy.Expr)
            ):
                raise SerializationError(
                    "Concrete QVR coefficient requires finite cost format and precision"
                )
            phase_width = qvr_phase_format(
                width, fraction, gamma, epsilon
            ).gradient_width
        phase = Register(
            "phase_grad", fixed_point_register_type(phase_width, phase_width)
        )
        ports += (phase,)
    if registers is not None and registers != ports:
        raise SerializationError("Invalid QVR register signature")
    if cost.shape_symbolic or cost.side is not RegisterSide.THRU or width == 0:
        return FunctionRecord(name, (), registers=ports)
    if gradient:
        if isinstance(gamma, sympy.Expr):
            addition = FunctionRecord(
                "scaled_add", (), registers=(Register("x", cost.dtype), phase)
            )
            return _rename_program(
                qvr_phase_addition_program(
                    cost, phase_width, ProgramRecord((addition,))
                ),
                name,
            )
        assert isinstance(epsilon, (float, int))
        return _rename_program(qvr_phase_gradient_program(cost, gamma, epsilon), name)
    if cost.dtype.is_symbolic() or gamma == 0:
        return FunctionRecord(name, (), registers=ports)
    assert isinstance(fraction, int) and isinstance(signed, bool)
    if isinstance(gamma, sympy.Expr) or isinstance(epsilon, sympy.Expr):
        assert isinstance(width, int)
        try:
            schedule = symbolic_qvr_zpow_schedule(
                tuple(range(width)),
                cost_fraction_bits=fraction,
                coefficient=gamma,
                epsilon=epsilon,
                signed=signed,
            )
        except ValueError:
            # Source symbolic-index and out-of-range paths have no finite body.
            return FunctionRecord(name, (), registers=ports)
        powers = tuple(
            (
                wire,
                (
                    symbolic_parameter_value(exponent)
                    if isinstance(exponent, sympy.Expr)
                    else exponent
                ),
            )
            for wire, exponent, _ in schedule
        )
        return _rename_program(qvr_phase_power_program(cost, powers), name)
    fractional = _fractional_rotations(fraction, gamma, epsilon)
    rotations = width - fraction + fractional + int(signed)
    if rotations == 0 or signed and rotations > 0:
        return FunctionRecord(name, (), registers=ports)
    return _rename_program(qvr_zpow_program(cost, gamma, epsilon), name)


def qvr_factories() -> dict[str, ComponentFactory]:
    """Supply both original constructors and their shared register helper closure."""
    prefix = "qualtran.bloqs.rotations.quantum_variable_rotation."
    return {
        prefix + "QvrZPow": partial(_qvr_factory, False),
        prefix + "QvrPhaseGradient": partial(_qvr_factory, True),
    }
