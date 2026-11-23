# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Constructor adaptation from Qualtran v0.7.0 basic_gates/su2_rotation.py.

"""Resolve SU(2) constructors through their existing scalar and rotation owners."""

from collections.abc import Mapping
from typing import Literal

from ...core.ir.program import FunctionRecord, ProgramRecord
from ...core.ir.program.registers import Register
from ...core.ir.program.signatures import Signature
from ...core.value_math import SymbolicReal
from ...errors import SerializationError
from ..registers.component_builder import ComponentProgramBuilder
from ..registers.phase.su2_rotation import _source_angles
from .legacy_component_resolution import ComponentFactory, ResolvedArgument
from .legacy_rotation_factories import _real_argument, _rotation_factory
from .legacy_scalar_factories import _global_phase_factory
from .legacy_word_factories import _rename_program


def _su2_factory(
    name: str,
    registers: tuple[Register, ...] | None,
    arguments: Mapping[str, ResolvedArgument],
) -> ProgramRecord:
    if arguments.keys() - {"theta", "phi", "lambd", "global_shift", "eps"}:
        raise SerializationError("Unexpected constructor arguments for SU2RotationGate")
    theta = _real_argument(arguments, "theta")
    phi = _real_argument(arguments, "phi")
    lambd = _real_argument(arguments, "lambd")
    shift = _real_argument(arguments, "global_shift", 0)
    precision = _real_argument(arguments, "eps", 1e-11)
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


def su2_factories() -> dict[str, ComponentFactory]:
    """Resolve SU2RotationGate through the unchanged source's four-call chain.

    Existing geometry helpers own the source angle formulas. Existing scalar
    and rotation constructors own symbolic parameters and materialization.
    The original scalar keeps its default precision; each rotation uses eps/3.
    """
    return {"qualtran.bloqs.basic_gates.su2_rotation.SU2RotationGate": _su2_factory}
