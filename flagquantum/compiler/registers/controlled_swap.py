# Copyright 2023 Google LLC
# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 TwoBitCSwap.to_clifford_t_circuit and CSwap.
# Source: https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/basic_gates/swap.py
# Source: https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/mcmt/specialized_ctrl.py
# Changes: explicit native operands, retaining the literal exact seven-T stream.

"""Exact controlled bit/register exchange without relative-phase substitution."""

from ...core.ir import Instruction
from .and_compute import and_compute_ir, and_uncompute_ir
from .bitwise import _validate_wires
from .control_predicate import ControlRegister, bind_control_registers
from .controlled_component import controlled_component_ir
from .multi_control import multi_and_ir


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


def control_bit_cswap_ir(
    controls: tuple[tuple[int, ...], ...],
    registers: tuple[ControlRegister, ...],
    existing_controls: tuple[int, ...],
    existing_values: tuple[int, ...],
    left: int,
    right: int,
    workspace: tuple[int, ...],
    classical_bits: tuple[int, ...],
) -> tuple[Instruction, ...]:
    """Add controls to an exact Fredkin with the pinned specialization.

    A single new control prepends its activation bit to the existing
    predicate. Multiple new controls wrap an atomic Fredkin, with the last
    work wire and slot used by the inner conjunction.
    """
    outer, values = bind_control_registers(controls, registers)
    _validate_wires((*outer, *existing_controls, left, right, *workspace))
    if (
        not outer
        or not existing_controls
        or len(existing_controls) != len(existing_values)
    ):
        raise ValueError(
            "Fredkin control requires nonempty outer and existing patterns"
        )
    if type(existing_values[-1]) is not int or existing_values[-1] != 1:
        raise ValueError("The original atomic Fredkin requires a positive control")

    if len(outer) == 1:
        merged = (*outer, *existing_controls)
        activation = (*values, *existing_values)
        compute = multi_and_ir(merged, activation, workspace)
        erase = multi_and_ir(
            merged, activation, workspace, classical_bits=classical_bits, uncompute=True
        )
        return (*compute, *controlled_swap_bit_ir(workspace[-1], left, right), *erase)

    if len(existing_controls) != 1:
        raise ValueError(
            "Multiple outer controls on a merged Fredkin have no pinned decomposition"
        )
    return control_register_cswap_ir(
        controls,
        registers,
        existing_controls[0],
        (left,),
        (right,),
        workspace,
        classical_bits,
    )


def control_register_cswap_ir(
    controls: tuple[tuple[int, ...], ...],
    registers: tuple[ControlRegister, ...],
    original_control: int,
    left: tuple[int, ...],
    right: tuple[int, ...],
    workspace: tuple[int, ...],
    classical_bits: tuple[int, ...],
) -> tuple[Instruction, ...]:
    """Control a register CSwap while preserving the source's per-pair Ands.

    Reserve the last work wire and slot for each pair in turn. The other
    pairs of work wires and slots belong to the outer control predicate.
    """
    if not left or len(left) != len(right):
        raise ValueError("Controlled swap requires equal nonempty register widths")
    if not workspace or not classical_bits:
        raise ValueError("Controlling a CSwap requires a work wire and classical slot")
    if len(set(classical_bits)) != len(classical_bits):
        raise ValueError("Outer and inner classical slots must be distinct")
    _validate_wires(
        (
            original_control,
            *left,
            *right,
            *workspace,
            *(q for word in controls for q in word),
        )
    )
    work, slot = workspace[-1], classical_bits[-1]

    def emit(outer_control: int) -> tuple[Instruction, ...]:
        compute = and_compute_ir(outer_control, original_control, work)
        erase = and_uncompute_ir(outer_control, original_control, work, slot)
        return tuple(
            gate
            for x, y in zip(left, right, strict=True)
            for gate in (*compute, *controlled_swap_bit_ir(work, x, y), *erase)
        )

    return controlled_component_ir(
        controls, registers, emit, workspace[:-1], classical_bits[:-1]
    )
