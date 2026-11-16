# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 ZPowConstViaPhaseGradient.
# Source: https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/rotations/zpow_via_phase_gradient.py

"""Constant Z rotations through a shared phase-gradient register."""

from math import isfinite, tau
from typing import cast

from ....core.ir import Instruction
from ..._component_trace.symbolic_math import (
    SymbolicInt,
    SymbolicReal,
    ceil,
    log2,
    smax,
)
from ..arithmetic.arithmetic_resource_model import source_phase_add_cost
from ..bits.bitwise import _validate_wires, controlled_xor_constant_ir
from ..resources.symbolic_resource_model import SymbolicResourceFormula
from .phase_addition import phase_add_ir


def zpow_phase_gradient_constant(exponent: float, width: int) -> int:
    """Encode the original unsigned, truncated and wrapping exponent/2."""
    if type(width) is not int or width < 1:
        raise ValueError("Gradient width must be a positive integer")
    if type(exponent) not in (int, float) or not isfinite(exponent) or exponent < 0:
        raise ValueError("Z power requires a finite nonnegative exponent")
    numerator, denominator = (exponent / 2).as_integer_ratio()
    return ((numerator << width) // denominator) % (1 << width)


def zpow_phase_gradient_width(epsilon: SymbolicReal) -> SymbolicInt:
    """Apply the source precision rule before binding physical wires."""
    if isinstance(epsilon, (int, float)):
        if not isfinite(epsilon) or epsilon <= 0:
            raise ValueError("Z power tolerance must be finite and positive")
        return ceil(log2(tau / epsilon))
    from sympy import pi

    return ceil(log2(2 * pi / epsilon))


def source_zpow_gradient_load_constant(
    exponent: SymbolicReal, width: SymbolicInt
) -> SymbolicReal:
    """Retain the unencoded source placeholder for a symbolic angle or width."""
    if isinstance(exponent, (int, float)) and isinstance(width, int):
        return zpow_phase_gradient_constant(exponent, width)
    return exponent / 2


def source_zpow_gradient_resource_formula(
    exponent: SymbolicReal, width: SymbolicInt
) -> SymbolicResourceFormula:
    """Count the paired controlled loads and the source phase addition."""
    constant = source_zpow_gradient_load_constant(exponent, width)
    if isinstance(width, int):
        phase_add = source_phase_add_cost(width, width)
        add_peak = phase_add.qubits
        toffoli = phase_add.gates.toffoli
        clifford = 2 * constant.bit_count() if isinstance(constant, int) else 2 * width
        # A symbolic angle queries the call graph because it cannot decompose.
        peak = (
            max(2 * width + 1, add_peak + 1)
            if isinstance(exponent, (float, int))
            else max(width + 1, add_peak)
        )
    else:
        toffoli = width - 2
        clifford = 2 * width
        peak = cast(
            SymbolicInt,
            smax(
                width + cast(SymbolicInt, smax(2, width + 1)),
                2 * width + 1,
                cast(SymbolicInt, smax(3, 2 * width)) + 1,
                width + cast(SymbolicInt, smax(0, width)) + 1,
            ),
        )
    return SymbolicResourceFormula(
        and_bloq=0,
        clifford=clifford,
        measurement=0,
        logical_qubits=peak,
        toffoli=toffoli,
    )


def zpow_via_phase_gradient_ir(
    target: int,
    gradient: tuple[int, ...],
    angle_work: tuple[int, ...],
    carries: tuple[int, ...],
    classical_bits: tuple[int, ...],
    *,
    exponent: float,
    adjoint: bool = False,
) -> tuple[Instruction, ...]:
    """Kick back a constant Z power using a negative-exponent gradient state.

    Use an n-bit clean angle word and n-1 clean carries with matching slots.
    The original load and unload are retained, with an exact measured adder
    (or its adjoint) between them. The gradient itself is caller-owned.
    """
    n = len(gradient)
    value = zpow_phase_gradient_constant(exponent, n)
    if type(adjoint) is not bool:
        raise ValueError("Z power adjoint must be boolean")
    if len(angle_work) != n:
        raise ValueError("Z power requires one clean angle wire per gradient bit")
    _validate_wires((target, *gradient, *angle_work, *carries))
    load = controlled_xor_constant_ir(target, angle_work, value)
    addition = phase_add_ir(
        angle_work, gradient, carries, classical_bits, sign=-1 if adjoint else 1
    )
    return (*load, *addition, *load)
