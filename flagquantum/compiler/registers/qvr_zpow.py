# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 QvrZPow.
# Source: https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/rotations/quantum_variable_rotation.py

"""Direct variable rotation of a fixed-point cost register."""

from math import ceil, isfinite, ldexp, log2, pi, tau

from ...core.ir import Instruction
from .bitwise import _validate_wires


def qvr_zpow_schedule(
    cost: tuple[int, ...],
    *,
    cost_fraction_bits: int,
    coefficient: float,
    epsilon: float,
    signed: bool = False,
    adjoint: bool = False,
) -> tuple[tuple[int, float, float], ...]:
    """Return source (qubit, Z exponent, synthesis tolerance) in gate order.

    The concrete unsigned route visits retained qubits from least to most
    significant. The inverse reverses the same schedule and negates its
    exponents. A negative source rotation count skips the rotation loop,
    retaining the standalone sign phase in a signed layout. A signed
    layout with a positive rotation count exceeds the source register.
    """
    _validate_wires(cost)
    width = len(cost)
    if type(cost_fraction_bits) is not int or not 0 <= cost_fraction_bits <= width:
        raise ValueError("QVR Z power requires a valid cost fraction width")
    if type(signed) is not bool or type(adjoint) is not bool:
        raise ValueError("QVR Z power mode flags must be boolean")
    if signed and (width < 2 or cost_fraction_bits == width):
        raise ValueError("QVR Z power requires a valid signed cost format")
    if (
        type(coefficient) not in (int, float)
        or not isfinite(coefficient)
        or coefficient == 0
    ):
        raise ValueError("QVR Z power requires a finite nonzero coefficient")
    if type(epsilon) not in (int, float) or not isfinite(epsilon) or epsilon <= 0:
        raise ValueError("QVR Z power requires a finite positive epsilon")
    ratio = tau * abs(coefficient) / epsilon
    if not isfinite(ratio) or ratio == 0:
        raise ValueError(
            "QVR Z power coefficient and epsilon ratio must be finite and positive"
        )

    fractional = min(cost_fraction_bits, ceil(log2(ratio)))
    rotations = width - cost_fraction_bits + fractional + signed
    if rotations == 0:
        raise ValueError("QVR Z power source precision allocation has zero rotations")
    if signed and rotations > 0:
        raise ValueError(
            "QVR Z power source rotation indexes beyond the signed cost register"
        )
    tolerance = epsilon / rotations
    sign_phase = ((cost[0], -1.0 if adjoint else 1.0, tolerance),) if signed else ()
    offset = 1 + cost_fraction_bits - fractional
    scale = -coefficient if adjoint else coefficient
    order = reversed(range(rotations)) if adjoint else range(rotations)
    return sign_phase + tuple(
        (cost[-(i + offset)], ldexp(scale, i - fractional + 1), tolerance)
        for i in order
    )


def qvr_zpow_ir(
    cost: tuple[int, ...],
    *,
    cost_fraction_bits: int,
    coefficient: float,
    epsilon: float,
    signed: bool = False,
    adjoint: bool = False,
) -> tuple[Instruction, ...]:
    """Emit native phase instructions without synthesizing the rotations."""
    return tuple(
        Instruction("phase", (wire,), params={"theta": pi * exponent})
        for wire, exponent, _ in qvr_zpow_schedule(
            cost,
            cost_fraction_bits=cost_fraction_bits,
            coefficient=coefficient,
            epsilon=epsilon,
            signed=signed,
            adjoint=adjoint,
        )
    )
