# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 _KaliskiIterationStep1.
# Source: https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/mod_arithmetic/mod_division.py
# Changes: explicit physical registers and shared multi-control IR emission.

"""Termination flags for the original binary modular inverse iteration."""

from ...core.ir import Instruction
from .multi_control import multi_control_x_ir


def inverse_termination_ir(
    value: tuple[int, ...],
    marker: int,
    active: int,
    terminal: int,
    workspace: tuple[int, ...],
    classical_bits: tuple[int, ...],
    *,
    adjoint: bool = False,
) -> tuple[Instruction, ...]:
    """Update logical inverse flags; restore value and all clean work.

    Value has at least two bits and is big-endian. Supply len(value) clean quantum qubits
    and classical slots. Forward first XORs active & (value==0) into marker,
    then XORs marker into active and terminal. Terminal starts in zero for the
    source classical contract; the emitted gates are reversible on either bit.
    Marker, active and terminal are logical data, not disposable work.
    """
    if len(value) < 2 or type(adjoint) is not bool:
        raise ValueError(
            "Termination requires at least two value bits and boolean adjoint"
        )
    wires = (*value, marker, active, terminal, *workspace)
    if any(type(w) is not int or w < 0 for w in wires) or len(set(wires)) != len(wires):
        raise ValueError("Termination requires distinct nonnegative quantum wires")
    detection = multi_control_x_ir(
        (*value, active), (0,) * len(value) + (1,), marker, workspace, classical_bits
    )
    updates = (
        Instruction("cx", (marker, active)),
        Instruction("cx", (marker, terminal)),
    )
    return (*reversed(updates), *detection) if adjoint else (*detection, *updates)
