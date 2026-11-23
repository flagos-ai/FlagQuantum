# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Constructor adaptation from Qualtran v0.7.0 basic_gates/global_phase.py.

"""Bind zero-port scalar phases to the existing explicit carrier recipe."""

from collections.abc import Mapping
from math import pi

import sympy

from ...core.ir import CircuitIR
from ...core.ir.program import ProgramRecord
from ...core.ir.program.register_bindings import RegisterBinding
from ...core.ir.program.registers import Register
from ...core.ir.program.signatures import Signature
from ...errors import SerializationError
from ..registers.component_builder import ComponentProgramBuilder
from ..registers.component_gate_function import bit_gate_function
from ..registers.primitives.global_phase import _finite_angle, scalar_phase_ir
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
    gates = scalar_phase_ir(0, angle)
    builder = ComponentProgramBuilder(Signature(()))
    if gates:
        carrier = builder.allocate(1)
        phase = bit_gate_function(
            "scalar_phase", tuple(Signature.build(q=1)), CircuitIR(1, gates)
        )
        result = builder.call_program(
            ProgramRecord((phase,)), {"q": carrier}, entry=phase.name
        )
        restored = result["q"]
        assert isinstance(restored, RegisterBinding)
        builder.free(restored)
    return _rename_program(builder.finalize({}), name)


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
