# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 LessThanConstant.decompose_from_registers.
# Source: https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/arithmetic/comparison.py
# Changes: native IR, explicit work allocation and shared And emitters.

"""Unsigned constant comparison through clean prefix flags."""

from ...core.ir import Instruction
from .and_compute import and_compute_ir, and_uncompute_ir


def less_than_constant_workspace_size(width: int, threshold: int) -> int:
    """Return clean quantum work size for the literal concrete comparison path."""
    if type(width) is not int or width < 1:
        raise ValueError("Comparison width must be a positive concrete integer")
    if type(threshold) is not int or threshold < 0:
        raise ValueError("Comparison threshold must be a nonnegative concrete integer")
    return 0 if threshold >= 1 << width else width + 1


def less_than_constant_ir(
    source: tuple[int, ...],
    threshold: int,
    target: int,
    workspace: tuple[int, ...],
    classical_bits: tuple[int, ...],
) -> tuple[Instruction, ...]:
    """XOR x<threshold into target, preserving input amplitudes and clean work.

    The source is big-endian. Normal paths require n+1 zero work qubits and n
    distinct nonnegative classical slots; the always-true path requires neither.
    Every quantum work qubit is zero at exit. Target need not start in zero.
    """
    required = less_than_constant_workspace_size(len(source), threshold)
    wires = (*source, target, *workspace)
    if any(type(w) is not int or w < 0 for w in wires) or len(set(wires)) != len(wires):
        raise ValueError("Comparison requires distinct nonnegative quantum wires")
    if len(workspace) != required:
        raise ValueError(f"Comparison requires {required} clean work wires")
    bit_count = len(source) if required else 0
    if (
        len(classical_bits) != bit_count
        or len(set(classical_bits)) != bit_count
        or any(type(bit) is not int or bit < 0 for bit in classical_bits)
    ):
        raise ValueError(
            f"Comparison requires {bit_count} distinct nonnegative classical slots"
        )
    if not required:
        return (Instruction("x", (target,)),)
    equal, *prefix = workspace
    initialize = (Instruction("x", (equal,)),)
    gates: list[Instruction] = list(initialize)
    cleanup: list[tuple[Instruction, ...]] = [initialize]
    width = len(source)
    for i, (wire, flag, bit) in enumerate(
        zip(source, prefix, classical_bits, strict=True)
    ):
        threshold_bit = threshold >> (width - i - 1) & 1
        if threshold_bit:
            flip = (Instruction("x", (wire,)),)
            gates.extend(flip)
            cleanup.append(flip)
        gates.extend(and_compute_ir(wire, equal, flag))
        cleanup.append(and_uncompute_ir(wire, equal, flag, bit))
        if threshold_bit:
            gates.append(Instruction("cx", (flag, target)))
        update = (Instruction("cx", (flag, equal)),)
        gates.extend(update)
        cleanup.append(update)
    for block in reversed(cleanup):
        gates.extend(block)
    return tuple(gates)
