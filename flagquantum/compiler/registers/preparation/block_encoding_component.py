# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Calls follow locked Qualtran v0.7.0 block_encoding/lcu_block_encoding.py.

"""Compose typed SELECT and zero-signal LCU block encodings."""

from ....core.ir.program import ProgramRecord
from ....core.ir.program.register_types import quantum_bit_type
from ....core.ir.program.registers import Register, RegisterSide
from ....core.ir.program.signatures import Signature
from ..component_adjoint import adjoint_component_calls
from ..component_builder import BindingCollection, ComponentProgramBuilder
from ..component_control import ControlAdder


def select_block_signature(
    selection: tuple[Register, ...], target: tuple[Register, ...]
) -> Signature:
    """Expose SELECT storage while leaving PREPARE junk in the signal state."""
    return Signature((*selection, *target))


def lcu_block_signature(
    selection: tuple[Register, ...],
    junk: tuple[Register, ...],
    target: tuple[Register, ...],
    *,
    controlled: bool = False,
) -> Signature:
    """Keep source order, optional ctrl and only THRU preparation junk."""
    control = (Register("ctrl", quantum_bit_type()),) if controlled else ()
    return Signature(
        (
            *control,
            *selection,
            *(reg for reg in junk if reg.side == RegisterSide.THRU),
            *target,
        )
    )


def _call(
    owner: ComponentProgramBuilder,
    child: ProgramRecord,
    state: dict[str, BindingCollection],
) -> None:
    registers = child.functions[0].registers
    if registers is None:
        raise ValueError("Block encoding requires typed child ports")
    inputs = {reg.name: state.pop(reg.name) for reg in Signature(registers).lefts()}
    state.update(owner.call_program(child, inputs, entry=child.functions[0].name))


def select_block_component_program(
    signature: Signature,
    select: ProgramRecord,
    *,
    adjoint: bool = False,
    inverse: ProgramRecord | None = None,
) -> ProgramRecord:
    """Keep the single source SELECT call and caller-owned bystanders."""
    if type(adjoint) is not bool:
        raise ValueError("SELECT block direction must be boolean")
    builder = ComponentProgramBuilder(signature)
    state = dict(builder.inputs)
    _call(builder, select, state)
    record = builder.finalize(state)
    if adjoint:
        if inverse is None:
            raise ValueError("SELECT block requires its exact child inverse")
        record = adjoint_component_calls(record, ((select, inverse),))
    return record


def lcu_block_component_program(
    signature: Signature,
    prepare: ProgramRecord,
    unprepare: ProgramRecord,
    select: ProgramRecord,
    *,
    controlled_select: ControlAdder | None = None,
    adjoint: bool = False,
    inverse_pairs: tuple[tuple[ProgramRecord, ProgramRecord], ...] = (),
) -> ProgramRecord:
    """Keep prepare/SELECT/unprepare, including source controlled-adder calls.

    Selection, THRU junk and targets are caller-owned MSB-first words. RIGHT
    preparation outputs remain available to its exact inverse and do not become
    public block outputs. Private clean workspace belongs to each child. A
    controlled adder owns the original call chain, including polarity flips;
    it is never replaced by an equivalent opaque controlled unitary.
    """
    if type(adjoint) is not bool:
        raise ValueError("LCU block direction must be boolean")
    builder = ComponentProgramBuilder(signature)
    state = dict(builder.inputs)
    _call(builder, prepare, state)
    if controlled_select is None:
        _call(builder, select, state)
    else:
        registers = select.functions[0].registers
        if registers is None:
            raise ValueError("Controlled SELECT requires typed original ports")
        ports = Signature(registers)
        inputs = {reg.name: state.pop(reg.name) for reg in ports.lefts()}
        control, outputs = controlled_select(builder, state.pop("ctrl"), inputs)
        right = tuple(ports.rights())
        if len(outputs) != len(right):
            raise ValueError("Controlled SELECT returned the wrong number of ports")
        state["ctrl"] = control
        state.update(
            (reg.name, value) for reg, value in zip(right, outputs, strict=True)
        )
    _call(builder, unprepare, state)
    record = builder.finalize(state)
    if adjoint:
        record = adjoint_component_calls(
            record, ((prepare, unprepare), (unprepare, prepare), *inverse_pairs)
        )
    return record
