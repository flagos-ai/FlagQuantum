# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 CNOT and Toffoli get_ctrl_system.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/basic_gates/cnot.py
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/basic_gates/toffoli.py

"""Append controls to CNOT and Toffoli in their original operand order."""

from ....core.ir import Instruction
from ..bits.bitwise import _validate_wires
from ..control.multi_control import _controlled_target_ir, multi_control_x_ir


def cnot_with_controls_ir(
    control: int,
    target: int,
    *,
    controls: tuple[int, ...] = (),
    values: tuple[int, ...] = (),
    workspace: tuple[int, ...] = (),
    classical_bits: tuple[int, ...] = (),
) -> tuple[Instruction, ...]:
    """Apply the source controlled CNOT through its Toffoli specialization."""
    _validate_wires((control, target))
    return _controlled_target_ir(
        Instruction("cx", (control, target)),
        lambda added: Instruction("ccx", (added, control, target)),
        controls,
        values,
        workspace,
        classical_bits,
    )


def toffoli_with_controls_ir(
    control_a: int,
    control_b: int,
    target: int,
    *,
    controls: tuple[int, ...] = (),
    values: tuple[int, ...] = (),
    workspace: tuple[int, ...] = (),
    classical_bits: tuple[int, ...] = (),
) -> tuple[Instruction, ...]:
    """Append controls before the original two positive controls in the AND."""
    _validate_wires((control_a, control_b, target))
    if not controls:
        if values or workspace or classical_bits:
            raise ValueError("Bare Toffoli takes no additional controls or work")
        return (Instruction("ccx", (control_a, control_b, target)),)
    if len(controls) != len(values):
        raise ValueError("Controls require one concrete zero/one value per wire")
    return multi_control_x_ir(
        (*controls, control_a, control_b),
        (*values, 1, 1),
        target,
        workspace,
        classical_bits,
    )
