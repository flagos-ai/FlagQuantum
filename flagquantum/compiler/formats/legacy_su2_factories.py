# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Constructor adaptation from Qualtran v0.7.0 basic_gates/su2_rotation.py.

"""Resolve SU(2) constructors through their existing scalar and rotation owners."""

from collections.abc import Mapping
from typing import Literal

import sympy

from ...core.ir import CircuitIR, Instruction
from ...core.ir.program import FunctionRecord, ProgramRecord
from ...core.ir.program.register_types import quantum_bit_type
from ...core.ir.program.registers import Register
from ...core.ir.program.signatures import Signature
from ...core.value_math import SymbolicReal
from ...errors import SerializationError
from ..registers.component_builder import ComponentProgramBuilder
from ..registers.component_control import ControlAdder, _stored_control_adder
from ..registers.component_gate_function import bit_gate_function
from ..registers.phase.su2_rotation import _source_angles
from .legacy_component_arguments import ControlSpecification
from .legacy_component_resolution import ComponentFactory, ResolvedArgument
from .legacy_controlled_factories import controlled_via_and_factories
from .legacy_rotation_factories import _real_argument, _rotation_factory
from .legacy_scalar_factories import _global_phase_factory
from .legacy_word_factories import _rename_program
from .symbolic_parameters import symbolic_parameter_value


def _su2_parameters(
    arguments: Mapping[str, ResolvedArgument],
) -> tuple[SymbolicReal, SymbolicReal, SymbolicReal, SymbolicReal, SymbolicReal]:
    if arguments.keys() - {"theta", "phi", "lambd", "global_shift", "eps"}:
        raise SerializationError("Unexpected constructor arguments for SU2RotationGate")
    return (
        _real_argument(arguments, "theta"),
        _real_argument(arguments, "phi"),
        _real_argument(arguments, "lambd"),
        _real_argument(arguments, "global_shift", 0),
        _real_argument(arguments, "eps", 1e-11),
    )


def _su2_factory(
    name: str,
    registers: tuple[Register, ...] | None,
    arguments: Mapping[str, ResolvedArgument],
) -> ProgramRecord:
    theta, phi, lambd, shift, precision = _su2_parameters(arguments)
    signature = Signature.build(q=1)
    if registers is not None and registers != tuple(signature):
        raise SerializationError("Invalid register signature for SU2RotationGate")
    exponent, left, middle, right = _source_angles(theta, phi, lambd, shift)
    builder = ComponentProgramBuilder(signature)
    scalar = _global_phase_factory("scalar", (), {"exponent": exponent})
    builder.call_program(scalar, {}, entry="scalar")
    q = builder.inputs["q"]
    rotations: tuple[tuple[Literal["Rz", "Rx"], SymbolicReal], ...] = (
        ("Rz", left),
        ("Rx", middle),
        ("Rz", right),
    )
    for index, (kind, angle) in enumerate(rotations):
        child_name = f"rotation_{index}"
        child = _rotation_factory(
            kind, child_name, None, {"angle": angle, "eps": precision / 3}
        )
        module = ProgramRecord((child,)) if isinstance(child, FunctionRecord) else child
        q = builder.call_program(module, {"q": q}, entry=child_name)["q"]
    return _rename_program(builder.finalize({"q": q}), name)


def _positive_su2_program(
    name: str, arguments: Mapping[str, ResolvedArgument]
) -> ProgramRecord:
    """Retain CRz, controlled Rx, CRz and the phase on the actual control."""
    theta, phi, lambd, shift, precision = _su2_parameters(arguments)
    exponent, left, middle, right = _source_angles(theta, phi, lambd, shift)
    builder = ComponentProgramBuilder(Signature.build(ctrl=1, q=1), name=name)
    state = dict(builder.inputs)
    for index, angle in enumerate((left, middle, right)):
        child_name = f"controlled_rotation_{index}"
        child: FunctionRecord | ProgramRecord
        if index == 1:
            # Controlled(Rx) has no source Bloq body. This existing native atom
            # supplies its independently checked action, not a source gate stream.
            value = (
                symbolic_parameter_value(angle)
                if isinstance(angle, sympy.Expr)
                else angle
            )
            child = bit_gate_function(
                child_name,
                tuple(Signature.build(ctrl=1, q=1)),
                CircuitIR(2, (Instruction("crx", (0, 1), params={"theta": value}),)),
            )
        else:
            child = _rotation_factory(
                "CRz", child_name, None, {"angle": angle, "eps": precision / 3}
            )
        module = ProgramRecord((child,)) if isinstance(child, FunctionRecord) else child
        state = builder.call_program(module, state, entry=child_name)
    phase = _rotation_factory("ZPowGate", "control_phase", None, {"exponent": exponent})
    module = ProgramRecord((phase,)) if isinstance(phase, FunctionRecord) else phase
    state["ctrl"] = builder.call_program(
        module, {"q": state["ctrl"]}, entry="control_phase"
    )["q"]
    return builder.finalize(state)


def controlled_su2_factories() -> dict[str, ComponentFactory]:
    """Scope actual SU2 control recipes without guessing child identities."""
    recipes: dict[FunctionRecord, dict[str, ResolvedArgument]] = {}

    def ordinary(
        name: str,
        registers: tuple[Register, ...] | None,
        arguments: Mapping[str, ResolvedArgument],
    ) -> ProgramRecord:
        result = _su2_factory(name, registers, arguments)
        recipes[result.functions[0]] = dict(arguments)
        return result

    def positive(child: FunctionRecord) -> tuple[ProgramRecord, ControlAdder]:
        if child not in recipes:
            raise SerializationError("Missing explicit SU2 control recipe")
        result = _positive_su2_program(child.name + "_positive", recipes[child])
        return result, _stored_control_adder(result.functions[0])

    def controlled(
        name: str,
        registers: tuple[Register, ...] | None,
        arguments: Mapping[str, ResolvedArgument],
    ) -> ProgramRecord:
        child, spec = arguments.get("subbloq"), arguments.get("ctrl_spec")
        if (
            arguments.keys() != {"subbloq", "ctrl_spec"}
            or not isinstance(child, FunctionRecord)
            or not isinstance(spec, ControlSpecification)
            or len(spec.clauses) != 1
        ):
            raise SerializationError("Controlled SU2 requires one positive scalar bit")
        dtype, values = spec.clauses[0]
        if dtype != quantum_bit_type() or values.shape or int(values) != 1:
            raise SerializationError("Controlled SU2 requires one positive scalar bit")
        result = _rename_program(positive(child)[0], name)
        if registers is not None and registers != result.functions[0].registers:
            raise SerializationError("Invalid controlled SU2 register signature")
        return result

    return {
        "qualtran.bloqs.basic_gates.su2_rotation.SU2RotationGate": ordinary,
        "qualtran._infra.controlled.Controlled": controlled,
        **controlled_via_and_factories(control_system=positive),
    }


def su2_factories() -> dict[str, ComponentFactory]:
    """Resolve SU2RotationGate through the unchanged source's four-call chain.

    Existing geometry helpers own the source angle formulas. Existing scalar
    and rotation constructors own symbolic parameters and materialization.
    The original scalar keeps its default precision; each rotation uses eps/3.
    """
    return {"qualtran.bloqs.basic_gates.su2_rotation.SU2RotationGate": _su2_factory}
