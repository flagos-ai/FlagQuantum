# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# ArcTan.apply sign and magnitude contract from Qualtran v0.7.0:
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/arithmetic/trigonometric/arctan.py

"""Bounded ArcTan lookup on the shared unary QROM."""

from math import atan, ldexp, pi

from ...core.ir import Instruction
from .source_qrom import source_qrom_ir


def arctan_lookup_requirements(source_width: int, target_width: int) -> tuple[int, int]:
    """Return clean workspace and slots for a concrete ArcTan table."""
    if type(source_width) is not int or not 1 <= source_width <= 12:
        raise ValueError("ArcTan lookup requires 1 to 12 source bits")
    if type(target_width) is not int or not 1 <= target_width <= 32:
        raise ValueError("ArcTan lookup requires 1 to 32 target bits")
    size = source_width - 1
    return size, size


def _arctan_magnitude_words(source_width: int, target_width: int) -> tuple[int, ...]:
    # QFxp truncates positive values toward zero at the target precision.
    return tuple(
        int(ldexp(2 * atan(x) / pi, target_width)) for x in range(1 << source_width)
    )


def arctan_lookup_ir(
    source: tuple[int, ...],
    sign: int,
    target: tuple[int, ...],
    workspace: tuple[int, ...],
    classical_bits: tuple[int, ...],
) -> tuple[Instruction, ...]:
    """XOR the sign and fractional magnitude of -2*atan(x)/pi.

    Source and target are big endian. The unsigned source is preserved.
    Target and sign may be occupied, and the same circuit erases a
    prior lookup. Clean QROM scratch is restored for every outcome.
    """
    arctan_lookup_requirements(len(source), len(target))
    size = 1 << len(source)
    signs = (0,) + (1,) * (size - 1)
    magnitude = _arctan_magnitude_words(len(source), len(target))
    return source_qrom_ir(
        (),
        (source,),
        (size,),
        (((sign,),), (target,)),
        (signs, magnitude),
        workspace,
        classical_bits,
    )
