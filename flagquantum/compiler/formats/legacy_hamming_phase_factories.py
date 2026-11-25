# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Constructors adapted from locked Qualtran v0.7.0 rotations/hamming_weight_phasing.py.

"""Resolve both Hamming phase families through shared count and phase owners."""

from collections.abc import Mapping
from functools import partial

import sympy

from ...core.ir.program import FunctionRecord, ProgramRecord
from ...core.ir.program.register_types import (
    fixed_point_register_type,
    unsigned_register_type,
)
from ...core.ir.program.registers import Register
from ...errors import SerializationError
from ..registers.phase.hamming_phasing_component import (
    hamming_gradient_phasing_program,
    hamming_phasing_program,
)
from ..registers.phase.qvr_phase_gradient import (
    qvr_phase_format,
    source_qvr_gradient_bound,
)
from .legacy_component_resolution import ComponentFactory, ResolvedArgument
from .legacy_modular_addition_factories import _integer
from .legacy_qvr_factories import qvr_factories
from .legacy_rotation_factories import _real_argument
from .legacy_word_factories import _rename_program
from .symbolic_parameters import symbolic_parameter_value


def _hamming_phase_factory(
    gradient: bool,
    name: str,
    registers: tuple[Register, ...] | None,
    arguments: Mapping[str, ResolvedArgument],
) -> FunctionRecord | ProgramRecord:
    if "bitsize" not in arguments or arguments.keys() - {"bitsize", "exponent", "eps"}:
        raise SerializationError(
            "Hamming phasing requires width and optional phase metadata"
        )
    width = _integer(arguments, "bitsize")
    exponent = _real_argument(arguments, "exponent", 1)
    epsilon = _real_argument(arguments, "eps", 1e-10)
    ports: tuple[Register, ...] = (
        (Register("x", unsigned_register_type(width)),) if width != 0 else ()
    )
    if gradient:
        if type(width) is not int or (
            isinstance(epsilon, sympy.Expr) and not isinstance(exponent, sympy.Expr)
        ):
            raise SerializationError(
                "Gradient Hamming signature requires concrete width and precision"
            )
        count = width.bit_length()
        if isinstance(exponent, sympy.Expr):
            _, phase_width = source_qvr_gradient_bound(epsilon)
        else:
            phase_width = qvr_phase_format(
                count, 0, exponent / 2, epsilon
            ).gradient_width
        phase = Register(
            "phase_grad", fixed_point_register_type(phase_width, phase_width)
        )
        ports += (phase,)
    if registers is not None and registers != ports:
        raise SerializationError("Invalid Hamming phasing register signature")
    if type(width) is not int or width == 0:
        return FunctionRecord(name, (), registers=ports)
    if gradient and isinstance(exponent, sympy.Expr):
        qvr = qvr_factories()[
            "qualtran.bloqs.rotations.quantum_variable_rotation.QvrPhaseGradient"
        ](
            "qvr",
            None,
            {
                "cost_reg": Register(
                    "out", fixed_point_register_type(width.bit_length(), 0)
                ),
                "gamma": exponent / 2,
                "eps": epsilon,
            },
        )
        assert isinstance(qvr, ProgramRecord)
        return _rename_program(hamming_gradient_phasing_program(width, qvr), name)
    angle = (
        symbolic_parameter_value(exponent)
        if isinstance(exponent, sympy.Expr)
        else exponent
    )
    # Direct synthesis precision is model metadata and adds no executable gate.
    return _rename_program(
        hamming_phasing_program(
            width,
            angle,
            1e-10 if isinstance(epsilon, sympy.Expr) else epsilon,
            via_gradient=gradient,
        ),
        name,
    )


def hamming_phase_factories() -> dict[str, ComponentFactory]:
    """Supply direct and gradient constructors with shared count dependency closure."""
    prefix = "qualtran.bloqs.rotations.hamming_weight_phasing."
    return {
        prefix + "HammingWeightPhasing": partial(_hamming_phase_factory, False),
        prefix
        + "HammingWeightPhasingViaPhaseGradient": partial(_hamming_phase_factory, True),
    }
