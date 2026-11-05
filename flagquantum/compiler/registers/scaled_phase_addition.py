# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 AddScaledValIntoPhaseReg.
# Source: https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/rotations/phase_gradient.py

"""Schedule fixed-point phase additions in the original coefficient bit order."""

from math import isfinite

from ...core.ir import Instruction
from .bitwise import _validate_wires
from .phase_addition import phase_add_ir, phase_add_workspace_size


def scaled_phase_add_terms(
    input_width: int,
    input_fraction_bits: int,
    gradient_width: int,
    coefficient: float,
    coefficient_width: int,
    coefficient_fraction_bits: int,
) -> tuple[tuple[slice, int], ...]:
    """Return (input slice, right shift) for each retained coefficient bit.

    Both formats describe unsigned fixed-point words. The coefficient's
    magnitude is truncated and saturated to its format, and its sign
    applies uniformly to the emitted additions. Slices use big-endian
    positions in the original input register.
    """
    for width, frac in (
        (input_width, input_fraction_bits),
        (coefficient_width, coefficient_fraction_bits),
    ):
        if (
            type(width) is not int
            or type(frac) is not int
            or not 0 <= frac <= width
            or not width
        ):
            raise ValueError("Scaled phase addition requires valid fixed-point formats")
    if type(gradient_width) is not int or gradient_width < 1:
        raise ValueError("Scaled phase addition requires a positive gradient width")
    if type(coefficient) not in (int, float) or (
        type(coefficient) is float and not isfinite(coefficient)
    ):
        raise ValueError("Scaled phase addition requires a finite coefficient")

    numerator, denominator = abs(coefficient).as_integer_ratio()
    magnitude = min(
        (numerator << coefficient_fraction_bits) // denominator,
        (1 << coefficient_width) - 1,
    )
    integer_bits = input_width - input_fraction_bits
    terms: list[tuple[slice, int]] = []
    for bit in reversed(range(magnitude.bit_length())):
        if not (magnitude >> bit) & 1:
            continue
        shift = bit - coefficient_fraction_bits
        if 0 <= shift < input_fraction_bits:
            terms.append((slice(integer_bits + shift, input_width), 0))
        elif -gradient_width - integer_bits < shift < 0:
            kept = input_fraction_bits + min(-shift, integer_bits)
            start = input_width - kept
            right_shift = max(0, -shift - integer_bits)
            count = min(kept, gradient_width - right_shift)
            terms.append((slice(start, start + count), right_shift))
    return tuple(terms)


def scaled_phase_add_ir(
    source: tuple[int, ...],
    gradient: tuple[int, ...],
    workspace: tuple[int, ...],
    classical_bits: tuple[int, ...],
    *,
    input_fraction_bits: int,
    coefficient: float,
    coefficient_width: int,
    coefficient_fraction_bits: int,
    adjoint: bool = False,
) -> tuple[Instruction, ...]:
    """Accumulate the source's per-term truncated product modulo the phase word.

    Clean work and classical slots are reused across terms. Supply the maximum
    phase_add_workspace_size over the scheduled slices, or zero for an empty
    schedule. For the adjoint, visit the same terms in reverse and negate each
    addition. Only the gradient word changes.
    """
    terms = scaled_phase_add_terms(
        len(source),
        input_fraction_bits,
        len(gradient),
        coefficient,
        coefficient_width,
        coefficient_fraction_bits,
    )
    if type(adjoint) is not bool:
        raise ValueError("Scaled phase addition adjoint must be boolean")
    _validate_wires((*source, *gradient, *workspace))
    operations = tuple(
        (
            source[part],
            shift,
            phase_add_workspace_size(
                len(source[part]), len(gradient), right_shift=shift
            ),
        )
        for part, shift in terms
    )
    required = max((size for _, _, size in operations), default=0)
    if len(workspace) != required:
        raise ValueError(f"Scaled phase addition requires {required} clean work wires")
    if (
        len(classical_bits) != required
        or len(set(classical_bits)) != required
        or any(type(bit) is not int or bit < 0 for bit in classical_bits)
    ):
        raise ValueError(
            f"Scaled phase addition requires {required} distinct classical slots"
        )
    sign = -1 if (coefficient < 0) ^ adjoint else 1
    order = reversed(operations) if adjoint else iter(operations)
    return tuple(
        gate
        for word, shift, size in order
        for gate in phase_add_ir(
            word,
            gradient,
            workspace[:size],
            classical_bits[:size],
            right_shift=shift,
            sign=sign,
        )
    )
