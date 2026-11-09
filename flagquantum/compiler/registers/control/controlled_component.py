# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 ControlledViaAnd and singly-controlled specialization.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/mcmt/controlled_via_and.py
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/mcmt/specialized_ctrl.py

"""Apply an existing positive-controlled component under structured controls."""

from collections.abc import Callable

from ....core.ir import Instruction
from ..bits.bitwise import _validate_wires
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


def extend_controlled_component_ir(
    controls: tuple[tuple[int, ...], ...],
    registers: tuple[ControlRegister, ...],
    current_control: tuple[int, int],
    emit_positive: Callable[[int], tuple[Instruction, ...]],
    workspace: tuple[int, ...],
    classical_bits: tuple[int, ...],
) -> tuple[Instruction, ...]:
    """Extend a singly controlled source component without flattening predicates.

    New controls precede the existing (wire, activation). Reserve len(new bits)
    clean qubits/slots: outer reduction first, existing-control conjunction last.
    A solitary added control uses a direct And with both original polarities.
    The callback owns the positive child and preserves all live parent storage.
    """
    wires, values = bind_control_registers(controls, registers)
    if (
        not wires
        or len(current_control) != 2
        or type(current_control[1]) is not int
        or current_control[1] not in (0, 1)
    ):
        raise ValueError(
            "Control extension needs new controls and an existing zero/one activation"
        )
    if len(workspace) != len(wires) or len(classical_bits) != len(wires):
        raise ValueError(
            "Control extension needs one clean wire and result slot per new control bit"
        )
    if len(set(classical_bits)) != len(classical_bits) or any(
        type(bit) is not int or bit < 0 for bit in classical_bits
    ):
        raise ValueError(
            "Control extension result slots must be distinct nonnegative integers"
        )
    _validate_wires((*wires, current_control[0], *workspace))
    current_wire, current_value = current_control
    if len(wires) == 1:
        return controlled_component_ir(
            ((wires[0], current_wire),),
            (ControlRegister(1, (values[0], current_value)),),
            emit_positive,
            workspace,
            classical_bits,
        )
    prefix = len(wires) - 1

    def join(predicate: int) -> tuple[Instruction, ...]:
        return controlled_component_ir(
            ((predicate, current_wire),),
            (ControlRegister(1, (1, current_value)),),
            emit_positive,
            workspace[prefix:],
            classical_bits[prefix:],
        )

    return controlled_component_ir(
        controls, registers, join, workspace[:prefix], classical_bits[:prefix]
    )
