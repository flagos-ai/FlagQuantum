# Adapted from Qualtran v0.7.0 and the pinned reversible-circuit generator.
# Copyright 2024 Google LLC (gf2_addition.py and gf2_add_k.py)
# Copyright 2026 Google LLC (gen_gf_iadd.cc)
# SPDX-License-Identifier: Apache-2.0
# Sources:
# https://github.com/quantumlib/Qualtran/tree/v0.7.0/qualtran/bloqs/gf_arithmetic
# https://github.com/quantumlib/Qualtran/blob/9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453/kickmix/src/kickmix/gen/gf_arithmetic/gen_gf_iadd.cc
# Changes: bind polynomial-basis registers and emit native instructions.

"""Register operations over binary extension fields."""

from ...core.ir import Instruction


def gf2_add_ir(
    source: tuple[int, ...], target: tuple[int, ...], *, control: int | None = None
) -> tuple[Instruction, ...]:
    """XOR equal-width polynomial-basis registers, optionally under control."""
    if len(source) != len(target):
        raise ValueError("Binary-field addition requires equal register widths")
    wires = (*source, *target) if control is None else (*source, *target, control)
    if len(set(wires)) != len(wires):
        raise ValueError("Binary-field addition requires disjoint operands")
    pairs = zip(source, target, strict=True)
    if control is None:
        return tuple(Instruction("cx", (a, b)) for a, b in pairs)
    return tuple(Instruction("ccx", (a, control, b)) for a, b in pairs)


def gf2_add_constant_ir(target: tuple[int, ...], value: int) -> tuple[Instruction, ...]:
    """XOR a concrete polynomial coefficient word into a big-endian register."""
    if type(value) is not int or not 0 <= value < 1 << len(target):
        raise ValueError("Binary-field constant must fit the target register")
    if len(set(target)) != len(target):
        raise ValueError("Binary-field addition requires disjoint operands")
    return tuple(
        Instruction("x", (wire,))
        for index, wire in enumerate(target)
        if value & (1 << (len(target) - index - 1))
    )
