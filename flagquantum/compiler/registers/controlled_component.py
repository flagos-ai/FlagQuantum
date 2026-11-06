# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 ControlledViaAnd.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/mcmt/controlled_via_and.py

"""Apply an existing positive-controlled component under structured controls."""

from collections.abc import Callable

from ...core.ir import Instruction
from .control_predicate import ControlRegister, bind_control_registers
from .multi_control import multi_and_ir


def controlled_component_ir(
    controls: tuple[tuple[int, ...], ...],
    registers: tuple[ControlRegister, ...],
    emit_controlled: Callable[[int], tuple[Instruction, ...]],
    workspace: tuple[int, ...],
    classical_bits: tuple[int, ...],
) -> tuple[Instruction, ...]:
    """Compute, use and erase the predicate, preserving the control ports.

    The child emitter accepts one positive control and preserves its value.
    It may use its own explicit work and classical slots. It must leave this
    wrapper's prefix junk and original control registers untouched.
    """
    wires, bits = bind_control_registers(controls, registers)
    if not wires:
        raise ValueError("Controlled component requires a nonempty control")
    if len(wires) == 1:
        if workspace or classical_bits:
            raise ValueError("Single control needs no predicate work or slots")
        (control,) = wires
        if bits == (1,):
            return emit_controlled(control)
        flip = Instruction("x", (control,))
        return (flip, *emit_controlled(control), flip)

    prepare = multi_and_ir(wires, bits, workspace)
    erase = multi_and_ir(
        wires, bits, workspace, classical_bits=classical_bits, uncompute=True
    )
    return (*prepare, *emit_controlled(workspace[-1]), *erase)
