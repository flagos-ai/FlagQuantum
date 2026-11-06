# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 TwoBitCSwap.to_clifford_t_circuit and CSwap.
# Source: https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/basic_gates/swap.py
# Changes: explicit native operands, retaining the literal exact seven-T stream.

"""Exact controlled bit/register exchange without relative-phase substitution."""

from ...core.ir import Instruction
from .bitwise import _validate_wires
from .control_predicate import ControlRegister
from .controlled_component import controlled_component_ir


def controlled_swap_bit_ir(
    control: int, left: int, right: int
) -> tuple[Instruction, ...]:
    """Emit the source exact Fredkin Clifford+T circuit, with no work or measures."""
    _validate_wires((control, left, right))
    return (
        Instruction("cx", (right, left)),
        Instruction("cx", (control, left)),
        Instruction("h", (right,)),
        Instruction("t", (control,)),
        Instruction("tdg", (left,)),
        Instruction("t", (right,)),
        Instruction("cx", (right, left)),
        Instruction("cx", (control, right)),
        Instruction("t", (left,)),
        Instruction("cx", (control, left)),
        Instruction("tdg", (right,)),
        Instruction("tdg", (left,)),
        Instruction("cx", (control, right)),
        Instruction("cx", (right, left)),
        Instruction("t", (left,)),
        Instruction("h", (right,)),
        Instruction("cx", (right, left)),
    )


def controlled_swap_ir(
    control: int, left: tuple[int, ...], right: tuple[int, ...]
) -> tuple[Instruction, ...]:
    """Exchange arbitrary equal-width words on a coherent positive control."""
    if not left or len(left) != len(right):
        raise ValueError("Controlled swap requires equal nonempty register widths")
    _validate_wires((control, *left, *right))
    return tuple(
        gate
        for x, y in zip(left, right, strict=True)
        for gate in controlled_swap_bit_ir(control, x, y)
    )


def controlled_swap_with_predicate_ir(
    controls: tuple[tuple[int, ...], ...],
    registers: tuple[ControlRegister, ...],
    left: tuple[int, ...],
    right: tuple[int, ...],
    workspace: tuple[int, ...],
    classical_bits: tuple[int, ...],
) -> tuple[Instruction, ...]:
    """Apply exact register SWAP under the source's negative or mixed controls.

    One control requires no scratch. For multiple controls the shared ladder
    owns one clean wire and one classical slot per prefix conjunction.
    """
    _validate_wires(
        (*left, *right, *workspace, *(q for word in controls for q in word))
    )
    return controlled_component_ir(
        controls,
        registers,
        lambda control: controlled_swap_ir(control, left, right),
        workspace,
        classical_bits,
    )
