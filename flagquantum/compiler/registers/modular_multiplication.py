# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 CModMulK.build_composite_bloq.
# Source: https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/mod_arithmetic/mod_multiplication.py
# Changes: native IR and explicit work, reusing scale accumulation and exact CSwap.

"""Controlled constant modular multiplication retaining the source clear/swap."""

from math import gcd

from ...core.ir import Instruction
from .bitwise import _validate_wires
from .controlled_swap import controlled_swap_ir
from .modular_scaled_addition import (
    controlled_modular_scaled_add_ir,
    modular_scaled_requirements,
)


def modular_multiplication_requirements(width: int) -> tuple[int, int]:
    """Reserve a clean result word plus the shared scaled-add work pool."""
    q, c = modular_scaled_requirements(width)
    return width + q, c


def controlled_modular_multiply_constant_ir(
    control: int,
    target: tuple[int, ...],
    value: int,
    modulus: int,
    workspace: tuple[int, ...],
    classical_bits: tuple[int, ...],
) -> tuple[Instruction, ...]:
    """Map x to k*x modulo p on a positive control and valid residues in both branches.

    Big-endian inputs promise 0<=x<p, 0<k<p, gcd(k,p)=1. Keep both original
    scale-add stages and the final exact controlled exchange; caller work starts
    and ends zero. Do not replace that exchange by unconditional output aliases.
    """
    n = len(target)
    q, _ = modular_multiplication_requirements(n)
    if type(modulus) is not int or not 1 < modulus < 1 << n:
        raise ValueError("Multiplication modulus must fit the operand and exceed one")
    if type(value) is not int or not 0 < value < modulus or gcd(value, modulus) != 1:
        raise ValueError("Multiplication constant must be in (0,modulus) and coprime")
    if len(workspace) != q:
        raise ValueError(f"Modular multiplication requires {q} clean work wires")
    _validate_wires((control, *target, *workspace))
    result, inner = workspace[:n], workspace[n:]
    return (
        *controlled_modular_scaled_add_ir(
            control, target, result, value, modulus, inner, classical_bits
        ),
        *controlled_modular_scaled_add_ir(
            control,
            result,
            target,
            -pow(value, -1, modulus),
            modulus,
            inner,
            classical_bits,
        ),
        *controlled_swap_ir(control, target, result),
    )
