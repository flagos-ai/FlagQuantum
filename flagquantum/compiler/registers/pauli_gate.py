# Copyright 2023 Google LLC
# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/mcmt/specialized_ctrl.py
# Adapted from Qualtran v0.7.0 XGate, YGate, ZGate get_ctrl_system.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/basic_gates/x_basis.py
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/basic_gates/y_gate.py
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/basic_gates/z_basis.py

"""Source control dispatch for a fixed Pauli leaf and a single control group."""

from typing import Literal

from ...core.ir import Instruction
from .bitwise import _validate_wires
from .multi_control import _multi_control_gate_ir, _validate_controls, multi_and_ir


def pauli_gate_ir(
    gate: Literal["x", "y", "z"],
    target: int,
    *,
    controls: tuple[int, ...] = (),
    values: tuple[int, ...] = (),
    workspace: tuple[int, ...] = (),
    classical_bits: tuple[int, ...] = (),
) -> tuple[Instruction, ...]:
    """Bind a Pauli gate with one group of ordered binary controls.

    A two-wire positive control group on X selects atomic CCX and has no
    work. Other mixed or multi-wire groups use the source control wrapper.
    """
    if gate not in ("x", "y", "z"):
        raise ValueError("Pauli gate must be X, Y or Z")
    if gate == "x" and len(controls) == 2 and values == (1, 1):
        if any(type(value) is not int for value in values):
            raise ValueError("Controls require concrete zero/one values")
        if workspace or classical_bits:
            raise ValueError(
                "Double-positive X requires no clean work or classical slots"
            )
        _validate_wires((*controls, target))
        return (Instruction("ccx", (*controls, target)),)
    return _multi_control_gate_ir(
        gate, controls, values, target, workspace, classical_bits
    )


def controlled_pauli_with_controls_ir(
    gate: Literal["y", "z"],
    control: int,
    target: int,
    *,
    controls: tuple[int, ...] = (),
    values: tuple[int, ...] = (),
    workspace: tuple[int, ...] = (),
    classical_bits: tuple[int, ...] = (),
) -> tuple[Instruction, ...]:
    """Bind added controls to the source CY or CZ specialized predicate.

    For multiple added controls, the outer predicate occupies the leading
    work wires and slots. The final work wire and slot belong to the inner
    AND of the outer result and the original positive control.
    """
    if gate not in ("y", "z"):
        raise ValueError("Controlled Pauli gate must be CY or CZ")
    _validate_wires((*controls, control, target, *workspace))
    if len(controls) != len(values):
        raise ValueError("Controls require one concrete zero/one value per wire")
    _validate_controls(
        (*controls, control), (*values, 1), workspace, classical_bits, measured=True
    )
    if not controls:
        return (Instruction(f"c{gate}", (control, target)),)
    if len(controls) == 1:
        return _multi_control_gate_ir(
            gate, (*controls, control), (*values, 1), target, workspace, classical_bits
        )
    outer_work, inner_work = workspace[:-1], workspace[-1:]
    outer_bits, inner_bits = classical_bits[:-1], classical_bits[-1:]
    return (
        *multi_and_ir(controls, values, outer_work),
        *_multi_control_gate_ir(
            gate, (outer_work[-1], control), (1, 1), target, inner_work, inner_bits
        ),
        *multi_and_ir(
            controls, values, outer_work, classical_bits=outer_bits, uncompute=True
        ),
    )
