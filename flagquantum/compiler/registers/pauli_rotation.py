# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Angle and phase conventions from Qualtran v0.7.0 rotation bloqs.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/basic_gates/rotation.py

"""Exact Pauli powers and axis rotations in FlagQuantum IR."""

from math import pi
from typing import Literal

from ...core.ir import Instruction
from .bitwise import _validate_wires
from .global_phase import _finite_angle, scalar_phase_ir

PauliAxis = Literal["x", "y", "z"]


def pauli_rotation_ir(
    axis: PauliAxis, target: int, angle: float
) -> tuple[Instruction, ...]:
    """Emit the radian rotation exp(-i * angle * Pauli / 2)."""
    _validate_wires((target,))
    if axis not in ("x", "y", "z"):
        raise ValueError("Rotation axis must be x, y, or z")
    return (Instruction("r" + axis, (target,), params={"theta": _finite_angle(angle)}),)


def pauli_power_ir(
    axis: PauliAxis,
    target: int,
    exponent: float,
    *,
    global_shift: float = 0.0,
    control: int | None = None,
) -> tuple[Instruction, ...]:
    """Emit a Pauli power with eigenphases pi*t*s and pi*t*(1+s).

    The source X/Y powers permit global shift s. The source Z power fixes
    it at zero. A positive control makes the shift a phase on the control.
    """
    wires = (target,) if control is None else (control, target)
    _validate_wires(wires)
    if axis not in ("x", "y", "z"):
        raise ValueError("Rotation axis must be x, y, or z")
    if axis == "z" and global_shift != 0:
        raise ValueError("Z power requires zero global shift")
    theta = _finite_angle(exponent, pi)
    shift = _finite_angle(global_shift, theta)
    before_names: tuple[str, ...]
    after_names: tuple[str, ...]
    if axis == "x":
        before_names = after_names = ("h",)
    elif axis == "y":
        before_names, after_names = ("sdg", "h"), ("h", "s")
    else:
        before_names = after_names = ()
    before = tuple(Instruction(name, (target,)) for name in before_names)
    after = tuple(Instruction(name, (target,)) for name in after_names)

    power = Instruction(
        "phase" if control is None else "cphase", wires, params={"theta": theta}
    )
    if not shift:
        return (*before, power, *after)
    if control is not None:
        scalar = Instruction("phase", (control,), params={"theta": shift})
        return (*before, power, *after, scalar)

    return (*before, power, *after, *scalar_phase_ir(target, shift))
