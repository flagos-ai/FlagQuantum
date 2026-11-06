# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 SU2RotationGate.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/basic_gates/su2_rotation.py

"""Phase-exact SU(2) rotation in the source's gate convention."""

from cmath import exp, isfinite, phase
from collections.abc import Sequence
from math import atan2, cos, pi, sin
from typing import cast

from ...core.ir import Instruction
from .._component_trace.resource_generalization import RotationCall, ZeroCostCall
from .._component_trace.symbolic_math import SymbolicReal
from .global_phase import _finite_angle, global_phase_ir
from .pauli_rotation import pauli_rotation_ir


def _pi_for(*values: SymbolicReal) -> SymbolicReal:
    if all(isinstance(value, (int, float)) for value in values):
        return pi
    from sympy import pi as symbolic_pi

    return cast(SymbolicReal, symbolic_pi)


def su2_angles_from_matrix(
    matrix: Sequence[Sequence[complex]],
) -> tuple[float, float, float, float]:
    """Return theta, phi, lambda and scalar shift from a finite 2x2 unitary.

    The caller supplies a unitary. The two near-zero branches retain the
    pinned source's absolute tolerance of 1e-8.
    """
    if len(matrix) != 2 or any(len(row) != 2 for row in matrix):
        raise ValueError("SU(2) angles require a 2x2 matrix")
    top_left, top_right = matrix[0]
    bottom_left, bottom_right = matrix[1]
    if not all(
        isfinite(entry) for entry in (top_left, top_right, bottom_left, bottom_right)
    ):
        raise ValueError("SU(2) angles require finite matrix entries")

    theta = atan2(abs(bottom_left), abs(top_left))
    cosine, sine = cos(theta), sin(theta)
    # NumPy isclose(x, 0) uses an absolute tolerance of 1e-8.
    if abs(cosine) <= 1e-8:
        return theta, phase(top_right / sine), phase(bottom_left / sine), 0.0
    shift = phase(-bottom_right / cosine)
    rephase = exp(-1j * shift)
    if abs(sine) <= 1e-8:
        return theta, phase(top_left / cosine * rephase), 0.0, shift
    return (
        theta,
        phase(top_right / sine * rephase),
        phase(bottom_left / sine * rephase),
        shift,
    )


def su2_angles_from_zxz(
    z_1: SymbolicReal,
    x: SymbolicReal,
    z_2: SymbolicReal,
    *,
    global_shift: SymbolicReal = 0,
) -> tuple[SymbolicReal, SymbolicReal, SymbolicReal, SymbolicReal]:
    """Convert first Rz, Rx and final Rz angles to source SU(2) parameters."""
    lambd = _pi_for(z_1) / 2 - z_1
    theta = x / 2
    phi = _pi_for(z_2) / 2 - z_2
    shift = global_shift + (z_1 + z_2) / 2 - _pi_for(global_shift, z_1, z_2)
    return theta, phi, lambd, shift


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
