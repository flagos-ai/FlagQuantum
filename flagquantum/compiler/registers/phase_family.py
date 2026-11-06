# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 SGate, TGate, and their ControlledViaAnd wrappers.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/basic_gates/s_gate.py
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/basic_gates/t_gate.py

"""Exact S and T leaves, including measured register controls."""

from math import pi
from typing import Literal

from ...core.ir import Instruction
from .bitwise import _validate_wires
from .multi_control import _controlled_target_ir


def phase_family_ir(
    gate: Literal["s", "t"],
    target: int,
    *,
    adjoint: bool = False,
    controls: tuple[int, ...] = (),
    values: tuple[int, ...] = (),
    workspace: tuple[int, ...] = (),
    classical_bits: tuple[int, ...] = (),
) -> tuple[Instruction, ...]:
    """Emit a fixed S/T leaf or its exact controlled phase.

    The caller owns clean work and classical slots for the shared control
    predicate. The upstream positive controlled leaf has no decomposition.
    """
    if gate not in ("s", "t") or type(adjoint) is not bool:
        raise ValueError("Phase family requires S or T and a boolean adjoint flag")
    _validate_wires((target,))
    theta = pi / (2 if gate == "s" else 4)
    if adjoint:
        theta = -theta
    return _controlled_target_ir(
        Instruction(gate + "dg" if adjoint else gate, (target,)),
        lambda control: Instruction("cphase", (control, target), {"theta": theta}),
        controls,
        values,
        workspace,
        classical_bits,
    )
