# Copyright 2023 Google LLC
# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 XGate, YGate, ZGate get_ctrl_system.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/basic_gates/x_basis.py
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/basic_gates/y_gate.py
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/basic_gates/z_basis.py

"""Source control dispatch for a fixed Pauli leaf and a single control group."""

from typing import Literal

from ...core.ir import Instruction
from .bitwise import _validate_wires
from .multi_control import _multi_control_gate_ir


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
