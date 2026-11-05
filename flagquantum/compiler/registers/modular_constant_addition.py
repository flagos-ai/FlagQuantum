# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 CModAddK.build_composite_bloq.
# Source: https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/mod_arithmetic/mod_addition.py
# Changes: explicit native work ownership and shared constant/controlled Add IR.

"""Controlled modular constant addition through source load/add/unload stages."""

from ...core.ir import Instruction
from .bitwise import xor_constant_ir
from .controlled_modular_addition import (
    controlled_modular_add_ir,
    controlled_modular_requirements,
)


def modular_constant_requirements(width: int) -> tuple[int, int]:
    """Return the loaded constant plus shared controlled modular work budget."""
    quantum, classical = controlled_modular_requirements(width)
    return width + quantum, classical


def controlled_modular_add_constant_ir(
    control: int,
    target: tuple[int, ...],
    value: int,
    modulus: int,
    workspace: tuple[int, ...],
    classical_bits: tuple[int, ...],
) -> tuple[Instruction, ...]:
    """Add k modulo p under one positive coherent control, retaining source gates.

    Target is big-endian and lies in [0,p) in both control branches. The source
    CModAdd clean domain requires 0 <= k <= p. Work starts/ends zero; first n
    wires store k during the shared component and are unloaded by the same Xs.
    """
    n = len(target)
    quantum, _ = modular_constant_requirements(n)
    if type(value) is not int or type(modulus) is not int or not 0 <= value <= modulus:
        raise ValueError("Modular constant must be a concrete integer in [0,modulus]")
    if len(workspace) != quantum:
        raise ValueError(
            f"Modular constant addition requires {quantum} clean work wires"
        )
    loaded, inner = workspace[:n], workspace[n:]
    gates = controlled_modular_add_ir(
        control, loaded, target, modulus, inner, classical_bits
    )
    load = xor_constant_ir(loaded, value)
    return (*load, *gates, *load)
