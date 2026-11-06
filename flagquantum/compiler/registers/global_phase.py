# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 GlobalPhase and its controlled dispatch.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/basic_gates/global_phase.py

"""Exact scalar and control-pattern phase emission."""

from __future__ import annotations

from cmath import isfinite as isfinite_complex
from cmath import phase
from math import isfinite, pi
from typing import TYPE_CHECKING, cast

from ...core.ir import Instruction
from .._component_trace.symbolic_math import SymbolicReal
from .bitwise import _validate_wires
from .multi_control import _validate_controls, multi_and_ir

if TYPE_CHECKING:
    from sympy import Expr


def global_phase_exponent(coefficient: complex | Expr) -> SymbolicReal:
    """Return the source principal argument in units of pi.

    The magnitude is ignored. A concrete zero coefficient has exponent zero.
    """
    if isinstance(coefficient, (int, float, complex)):
        if not isfinite_complex(coefficient):
            raise ValueError("Global phase coefficient must be finite")
        return phase(coefficient) / pi
    from sympy import arg
    from sympy import pi as symbolic_pi

    return cast(SymbolicReal, arg(coefficient) / symbolic_pi)


def _finite_angle(value: float, scale: float = 1.0) -> float:
    if type(value) in (int, float):
        try:
            angle = scale * value
            if isfinite(angle):
                return angle
        except OverflowError:
            pass
    raise ValueError("Rotation requires a finite phase angle")


def scalar_phase_ir(anchor: int, angle: float) -> tuple[Instruction, ...]:
    """Multiply by exp(i * angle) using one existing wire of arbitrary state."""
    _validate_wires((anchor,))
    theta = _finite_angle(angle)
    if not theta:
        return ()
    flip = Instruction("x", (anchor,))
    shift = Instruction("phase", (anchor,), params={"theta": theta})
    return (flip, shift, flip, shift)


def global_phase_ir(
    exponent: float,
    *,
    anchor: int | None = None,
    controls: tuple[int, ...] = (),
    values: tuple[int, ...] = (),
    workspace: tuple[int, ...] = (),
    classical_bits: tuple[int, ...] = (),
) -> tuple[Instruction, ...]:
    """Apply exp(i * pi * exponent) globally or on a binary control pattern.

    Without controls, supply one arbitrary existing wire as a scalar
    carrier. With controls, supply one value per wire, in tuple order.
    Each extra control after the first needs one clean work wire and one
    classical slot. The measured inverse returns all work to zero.
    """
    theta = _finite_angle(exponent, pi)
    _validate_controls(controls, values, workspace, classical_bits, measured=True)
    if not controls:
        if anchor is None:
            raise ValueError("An uncontrolled global phase requires an anchor wire")
        return scalar_phase_ir(anchor, theta)
    if anchor is not None:
        raise ValueError(
            "A controlled global phase uses its controls as phase carriers"
        )

    target = workspace[-1] if workspace else controls[0]
    shift = Instruction("phase", (target,), params={"theta": theta})
    if len(controls) == 1:
        flip = (Instruction("x", controls),) if values == (0,) else ()
        return (*flip, shift, *flip)
    return (
        *multi_and_ir(controls, values, workspace),
        shift,
        *multi_and_ir(
            controls, values, workspace, classical_bits=classical_bits, uncompute=True
        ),
    )
