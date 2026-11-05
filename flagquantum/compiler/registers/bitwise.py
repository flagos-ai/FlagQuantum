# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 Xor, XorK and BitwiseNot.
# Source: https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/arithmetic/bitwise.py
# Changes: explicit big-endian wire tuples, concrete unsigned constants and
# one positive controlled XorK, all emitting native IR.

"""Shared register bit permutations without hidden work or dtype conversion."""

from ...core.ir import Instruction


def _validate_wires(wires: tuple[int, ...]) -> None:
    if not wires:
        raise ValueError("Bitwise operation requires nonempty registers")
    if any(type(w) is not int or w < 0 for w in wires) or len(set(wires)) != len(wires):
        raise ValueError("Bitwise operation requires distinct nonnegative wires")


def xor_ir(source: tuple[int, ...], target: tuple[int, ...]) -> tuple[Instruction, ...]:
    """Preserve source and XOR it into any equal-width target, MSB first."""
    if not source or len(source) != len(target):
        raise ValueError("XOR requires equal nonempty register widths")
    _validate_wires((*source, *target))
    return tuple(Instruction("cx", (a, b)) for a, b in zip(source, target, strict=True))


def xor_constant_ir(target: tuple[int, ...], value: int) -> tuple[Instruction, ...]:
    """XOR a concrete unsigned bit pattern into a big-endian register."""
    _validate_wires(target)
    if type(value) is not int or not 0 <= value < 1 << len(target):
        raise ValueError("XOR constant must fit the unsigned register")
    return tuple(
        Instruction("x", (wire,))
        for i, wire in enumerate(target)
        if (value >> (len(target) - i - 1)) & 1
    )


def bitwise_not_ir(target: tuple[int, ...]) -> tuple[Instruction, ...]:
    """Flip every bit, preserving the source OnEach order."""
    _validate_wires(target)
    return tuple(Instruction("x", (wire,)) for wire in target)


def controlled_xor_constant_ir(
    control: int, target: tuple[int, ...], value: int
) -> tuple[Instruction, ...]:
    """Apply a concrete XorK under one positive control, in original bit order.

    The control may be coherent. No conjunction work, measurement or T gates
    are introduced. Other control patterns are outside this concrete entry.
    """
    flips = xor_constant_ir(target, value)
    if type(control) is not int or control < 0 or control in target:
        raise ValueError("Controlled XOR requires a distinct nonnegative control")
    return tuple(Instruction("cx", (control, gate.wires[0])) for gate in flips)
