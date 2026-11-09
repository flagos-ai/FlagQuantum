# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 QvrPhaseGradient and find_optimal_phase_grad_size.
# Source: https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/rotations/quantum_variable_rotation.py

"""Concrete precision selection and lowering for phase-gradient variable rotations."""

from dataclasses import dataclass
from math import ceil, isfinite, log2, tau

from ....core.ir import Instruction
from .scaled_phase_addition import _coefficient_word, scaled_phase_add_ir


@dataclass(frozen=True)
class QvrPhaseFormat:
    """Concrete widths selected for the gradient and coefficient."""

    gradient_width: int
    coefficient_width: int
    coefficient_fraction_bits: int


def _optimized_width(
    word: int, fraction: int, magnitude: int, gamma_fraction: int, tolerance: float
) -> int:
    """Search the source's worst-case cost word using per-term truncation."""
    cost = word / (1 << fraction)
    expected = (magnitude / (1 << gamma_fraction) * cost) % 1
    shifts = tuple(
        bit - gamma_fraction - fraction
        for bit in reversed(range(magnitude.bit_length()))
        if (magnitude >> bit) & 1
    )
    low, high, result = 1, 100, None
    while low <= high:
        mid = (low + high) // 2
        value = 0
        modulus = 1 << mid
        for offset in shifts:
            shift = mid + offset
            # The pinned fixed-point search aligns each term to the
            # accumulator. A negative scale uses floating-point alignment
            # before conversion to the raw integer word.
            if shift < 0:
                value = int(value + word * 2.0**shift) % modulus
            else:
                value = (value + (word << shift)) % modulus
        if abs(value / (1 << mid) - expected) <= tolerance:
            result = mid
            high = mid - 1
        else:
            low = mid + 1
    if result is None:
        raise ValueError("QVR precision search found no gradient width")
    return result


def qvr_phase_format(
    cost_width: int, cost_fraction_bits: int, coefficient: float, epsilon: float
) -> QvrPhaseFormat:
    """Select widths with the original concrete precision and fallback rules."""
    if (
        type(cost_width) is not int
        or type(cost_fraction_bits) is not int
        or not 0 <= cost_fraction_bits <= cost_width
        or cost_width < 1
    ):
        raise ValueError("QVR requires a valid unsigned cost format")
    if type(epsilon) not in (int, float) or not 0 < epsilon < tau:
        raise ValueError("QVR epsilon must be positive and smaller than 2π")
    if type(coefficient) not in (int, float) or (
        type(coefficient) is float and not isfinite(coefficient)
    ):
        raise ValueError("QVR requires a finite coefficient")
    ratio = tau / epsilon
    if not isfinite(ratio):
        raise ValueError("QVR epsilon is too small for concrete sizing")
    phase_bits = ceil(log2(ratio))
    gamma_fraction = cost_width - cost_fraction_bits + phase_bits
    gamma_width = int(abs(coefficient)).bit_length() + gamma_fraction
    magnitude = _coefficient_word(coefficient, gamma_width, gamma_fraction)
    terms = min(magnitude.bit_count(), (phase_bits + 2) // 2)
    if not terms:
        raise ValueError(
            "QVR source sizing is undefined for a zero quantized coefficient"
        )
    formula = ceil(log2(terms * tau / epsilon))
    width = formula
    if cost_width < 54:
        optimized = _optimized_width(
            (1 << cost_width) - 1,
            cost_fraction_bits,
            magnitude,
            gamma_fraction,
            epsilon / tau,
        )
        width = min(formula, optimized)
    return QvrPhaseFormat(width, gamma_width, gamma_fraction)


def qvr_phase_gradient_ir(
    cost: tuple[int, ...],
    gradient: tuple[int, ...],
    workspace: tuple[int, ...],
    classical_bits: tuple[int, ...],
    *,
    cost_fraction_bits: int,
    coefficient: float,
    epsilon: float,
    adjoint: bool = False,
) -> tuple[Instruction, ...]:
    """Bind the selected format to the source scaled-add graph.

    The caller owns the negative-exponent phase-gradient state and the clean
    work required by the scaled addition. The cost bits stay unchanged.
    """
    fmt = qvr_phase_format(len(cost), cost_fraction_bits, coefficient, epsilon)
    if len(gradient) != fmt.gradient_width:
        raise ValueError(f"QVR requires a {fmt.gradient_width}-bit phase gradient")
    return scaled_phase_add_ir(
        cost,
        gradient,
        workspace,
        classical_bits,
        input_fraction_bits=cost_fraction_bits,
        coefficient=coefficient,
        coefficient_width=fmt.coefficient_width,
        coefficient_fraction_bits=fmt.coefficient_fraction_bits,
        adjoint=adjoint,
    )
