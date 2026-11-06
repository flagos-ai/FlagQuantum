# Copyright 2023 Google LLC
# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 MultiAnd and MultiControlX -> ControlledViaAnd.
# Sources: https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/mcmt/and_bloq.py
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/mcmt/multi_control_pauli.py
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/mcmt/controlled_via_and.py
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/basic_gates/hadamard.py
# Changes: native IR, explicit workspace and shared measured And emitters.

"""Concrete logical-And ladders and single-target controlled gates."""

from collections.abc import Callable
from typing import Literal

from ...core.ir import Instruction
from .and_compute import and_compute_ir, and_uncompute_ir


def _validate_controls(
    controls: tuple[int, ...],
    values: tuple[int, ...],
    workspace: tuple[int, ...],
    bits: tuple[int, ...],
    *,
    measured: bool,
) -> None:
    if len(values) != len(controls) or any(
        type(v) is not int or v not in (0, 1) for v in values
    ):
        raise ValueError("Controls require one concrete zero/one value per wire")
    wires = (*controls, *workspace)
    if any(type(w) is not int or w < 0 for w in wires) or len(set(wires)) != len(wires):
        raise ValueError("Controls and work require distinct nonnegative quantum wires")
    required = max(0, len(controls) - 1)
    if len(workspace) != required:
        raise ValueError(f"Control reduction requires {required} clean work wires")
    expected = required if measured else 0
    if (
        len(bits) != expected
        or len(set(bits)) != expected
        or any(type(b) is not int or b < 0 for b in bits)
    ):
        raise ValueError(
            f"Control reduction requires {expected} distinct nonnegative classical slots"
        )


def multi_and_ir(
    controls: tuple[int, ...],
    values: tuple[int, ...],
    workspace: tuple[int, ...],
    *,
    classical_bits: tuple[int, ...] = (),
    uncompute: bool = False,
) -> tuple[Instruction, ...]:
    """Compute prefix Ands, or erase them in the literal source reverse order.

    At least two controls are required. Work starts in zero for compute; at its
    exit work[i] is the conjunction through control i+1. These are logical
    prefix outputs, not clean scratch. Erase requires those correct prefix
    values and restores every work qubit to zero using caller-owned measurements.
    """
    if len(controls) < 2 or type(uncompute) is not bool:
        raise ValueError(
            "And ladder requires at least two controls and a boolean uncompute flag"
        )
    _validate_controls(controls, values, workspace, classical_bits, measured=uncompute)
    steps = [(controls[0], controls[1], workspace[0], (values[0], values[1]))]
    steps.extend(
        (workspace[i - 2], controls[i], workspace[i - 1], (1, values[i]))
        for i in range(2, len(controls))
    )
    if uncompute:
        return tuple(
            g
            for i in reversed(range(len(steps)))
            for g in and_uncompute_ir(
                *steps[i][:3], classical_bits[i], values=steps[i][3]
            )
        )
    return tuple(
        g for a, b, t, cvs in steps for g in and_compute_ir(a, b, t, values=cvs)
    )


def multi_control_x_ir(
    controls: tuple[int, ...],
    values: tuple[int, ...],
    target: int,
    workspace: tuple[int, ...],
    classical_bits: tuple[int, ...],
) -> tuple[Instruction, ...]:
    """XOR the specified control pattern into an arbitrary target; clear all work."""
    return _multi_control_gate_ir(
        "x", controls, values, target, workspace, classical_bits
    )


def multi_control_z_ir(
    controls: tuple[int, ...],
    values: tuple[int, ...],
    target: int,
    workspace: tuple[int, ...],
    classical_bits: tuple[int, ...],
) -> tuple[Instruction, ...]:
    """Apply the source MultiControlZ ladder and restore its clean work."""
    return _multi_control_gate_ir(
        "z", controls, values, target, workspace, classical_bits
    )


def multi_control_h_ir(
    controls: tuple[int, ...],
    values: tuple[int, ...],
    target: int,
    workspace: tuple[int, ...],
    classical_bits: tuple[int, ...],
) -> tuple[Instruction, ...]:
    """Apply Hadamard through the source CHadamard and predicate dispatch."""
    return _multi_control_gate_ir(
        "h", controls, values, target, workspace, classical_bits
    )


def _multi_control_gate_ir(
    gate: Literal["x", "y", "z", "h"],
    controls: tuple[int, ...],
    values: tuple[int, ...],
    target: int,
    workspace: tuple[int, ...],
    classical_bits: tuple[int, ...],
) -> tuple[Instruction, ...]:
    if type(target) is not int or target < 0:
        raise ValueError("Controlled gate target must be a distinct nonnegative wire")
    return _controlled_target_ir(
        Instruction(gate, (target,)),
        lambda control: Instruction(f"c{gate}", (control, target)),
        controls,
        values,
        workspace,
        classical_bits,
    )


def _controlled_target_ir(
    gate: Instruction,
    positive_controlled: Callable[[int], Instruction],
    controls: tuple[int, ...],
    values: tuple[int, ...],
    workspace: tuple[int, ...],
    classical_bits: tuple[int, ...],
) -> tuple[Instruction, ...]:
    _validate_controls(controls, values, workspace, classical_bits, measured=True)
    if set(gate.wires).intersection((*controls, *workspace)):
        raise ValueError(
            "Controlled gate operands must be distinct from controls and work"
        )
    if not controls:
        return (gate,)
    if len(controls) == 1:
        flip = (Instruction("x", controls),) if values[0] == 0 else ()
        return (*flip, positive_controlled(controls[0]), *flip)
    return (
        *multi_and_ir(controls, values, workspace),
        positive_controlled(workspace[-1]),
        *multi_and_ir(
            controls, values, workspace, classical_bits=classical_bits, uncompute=True
        ),
    )
