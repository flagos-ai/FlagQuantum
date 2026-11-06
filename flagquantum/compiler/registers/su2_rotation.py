# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 SU2RotationGate.build_composite_bloq and adjoint.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/basic_gates/su2_rotation.py

"""Phase-exact SU(2) rotation in the source's gate convention."""

from math import pi
from typing import cast

from ...core.ir import Instruction
from .._component_trace.resource_generalization import RotationCall, ZeroCostCall
from .._component_trace.symbolic_math import SymbolicReal
from .global_phase import _finite_angle, global_phase_ir
from .pauli_rotation import pauli_rotation_ir


def _pi_for(value: SymbolicReal) -> SymbolicReal:
    if isinstance(value, (int, float)):
        return pi
    from sympy import pi as symbolic_pi

    return cast(SymbolicReal, symbolic_pi)


def _source_angles(
    theta: SymbolicReal, phi: SymbolicReal, lambd: SymbolicReal, shift: SymbolicReal
) -> tuple[SymbolicReal, SymbolicReal, SymbolicReal, SymbolicReal]:
    pi_lambd = _pi_for(lambd)
    pi_phi = _pi_for(phi)
    exponent = (
        0.5 + shift / _pi_for(shift) + lambd / (2 * pi_lambd) + phi / (2 * pi_phi)
    )
    return exponent, pi_lambd / 2 - lambd, 2 * theta, pi_phi / 2 - phi


def su2_rotation_ir(
    target: int,
    theta: float,
    phi: float,
    lambd: float,
    *,
    global_shift: float = 0,
    adjoint: bool = False,
) -> tuple[Instruction, ...]:
    """Emit scalar phase, Rz, Rx and Rz, preserving the source's absolute phase."""
    if type(adjoint) is not bool:
        raise ValueError("SU(2) adjoint must be boolean")
    theta = _finite_angle(theta)
    phi = _finite_angle(phi)
    lambd = _finite_angle(lambd)
    global_shift = _finite_angle(global_shift)
    if adjoint:
        phi, lambd, global_shift = -lambd, -phi, -global_shift
    exponent, left, middle, right = _source_angles(theta, phi, lambd, global_shift)
    return (
        *global_phase_ir(float(exponent), anchor=target),
        *pauli_rotation_ir("z", target, float(left)),
        *pauli_rotation_ir("x", target, float(middle)),
        *pauli_rotation_ir("z", target, float(right)),
    )


def su2_rotation_calls(
    theta: SymbolicReal,
    phi: SymbolicReal,
    lambd: SymbolicReal,
    *,
    global_shift: SymbolicReal = 0,
    precision: SymbolicReal = 1e-11,
) -> tuple[ZeroCostCall, RotationCall, RotationCall, RotationCall]:
    """Return the source call keys before synthesis, with per-rotation precision."""
    exponent, left, middle, right = _source_angles(theta, phi, lambd, global_shift)
    rotation_precision = precision / 3
    return (
        ZeroCostCall("global_phase", (exponent, 1e-11)),
        RotationCall("rz", left, rotation_precision),
        RotationCall("rx", middle, rotation_precision),
        RotationCall("rz", right, rotation_precision),
    )
