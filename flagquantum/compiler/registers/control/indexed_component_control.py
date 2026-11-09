# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 get_ctrl_system_1bit_cv and ControlledViaAnd.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/mcmt/specialized_ctrl.py

"""Preserve indexed control specialization and nested predicate ownership."""

from collections.abc import Callable

from ....core.ir import Instruction
from ..bits.bitwise import _validate_wires
from .control_predicate import ControlRegister, bind_control_registers
from .controlled_component import (
    controlled_component_ir,
    extend_controlled_component_ir,
)
from .indexed_component import indexed_component_ir, indexed_component_workspace


def _validate_control_storage(
    controls: tuple[int, ...],
    axes: tuple[tuple[int, ...], ...],
    workspace: tuple[int, ...],
    classical_bits: tuple[int, ...],
) -> None:
    required = indexed_component_workspace(tuple(map(len, axes)), len(controls))
    if len(workspace) != required:
        raise ValueError(f"Indexed control system requires {required} clean work wires")
    if (
        len(classical_bits) != required
        or len(set(classical_bits)) != required
        or any(type(bit) is not int or bit < 0 for bit in classical_bits)
    ):
        raise ValueError("Indexed control system requires distinct work result slots")
    _validate_wires((*controls, *(wire for axis in axes for wire in axis), *workspace))


def conjoined_indexed_component_ir(
    axes: tuple[tuple[int, ...], ...],
    extents: tuple[int, ...],
    emit_controlled: Callable[[int, int], tuple[Instruction, ...]],
    controls: tuple[int, ...],
    values: tuple[int, ...],
    *,
    workspace: tuple[int, ...],
    classical_bits: tuple[int, ...],
    adjoint: bool = False,
) -> tuple[Instruction, ...]:
    """Lower a source specialized root after repeated single-bit controls.

    Newest controls precede older controls in both wire and activation order.
    One And (two controls) or MultiAnd (more controls) surrounds the positive
    indexed child. Predicate work precedes traversal work; both enter/leave zero.
    For adjoint traversal the callback must supply each child adjoint.
    """
    if len(controls) < 2:
        raise ValueError(
            "Conjoined indexed specialization requires at least two controls"
        )
    _validate_control_storage(controls, axes, workspace, classical_bits)
    prefix = len(controls) - 1

    def traverse(predicate: int) -> tuple[Instruction, ...]:
        return indexed_component_ir(
            axes,
            extents,
            emit_controlled,
            controls=(predicate,),
            workspace=workspace[prefix:],
            classical_bits=classical_bits[prefix:],
            adjoint=adjoint,
        )

    return controlled_component_ir(
        (controls,),
        (ControlRegister(1, values),),
        traverse,
        workspace[:prefix],
        classical_bits[:prefix],
    )


def specialized_indexed_component_ir(
    axes: tuple[tuple[int, ...], ...],
    extents: tuple[int, ...],
    emit_controlled: Callable[[int, int], tuple[Instruction, ...]],
    controls: tuple[tuple[int, ...], ...],
    registers: tuple[ControlRegister, ...],
    *,
    current_control: tuple[int, int] | None = None,
    workspace: tuple[int, ...],
    classical_bits: tuple[int, ...],
    adjoint: bool = False,
) -> tuple[Instruction, ...]:
    """Lower the locked ApplyLthBloq control-system call chain.

    New controls precede an optional existing (wire, activation) control.
    Storage order is outer CtrlSpec reduction, existing-control conjunction,
    then address traversal. Children own their target binding and work.
    One new control without an existing control follows the original direct
    specialization: upstream ignores activation zero. This preserves that
    source defect; callers requiring negative control must not use that case.
    The callback supplies child adjoints when adjoint is true.
    """
    wires, values = bind_control_registers(controls, registers)
    if not wires:
        raise ValueError("Indexed control specialization requires new controls")
    if current_control is not None and (
        len(current_control) != 2
        or type(current_control[1]) is not int
        or current_control[1] not in (0, 1)
    ):
        raise ValueError("Existing control requires a wire and a zero/one activation")
    outer = len(wires) - 1
    inner = int(current_control is not None)
    _validate_control_storage(
        (*wires, *((current_control[0],) if current_control else ())),
        axes,
        workspace,
        classical_bits,
    )

    def traverse(predicate: int) -> tuple[Instruction, ...]:
        return indexed_component_ir(
            axes,
            extents,
            emit_controlled,
            controls=(predicate,),
            workspace=workspace[outer + inner :],
            classical_bits=classical_bits[outer + inner :],
            adjoint=adjoint,
        )

    if current_control is not None:
        return extend_controlled_component_ir(
            controls,
            registers,
            current_control,
            traverse,
            workspace[: outer + inner],
            classical_bits[: outer + inner],
        )
    if outer == 0:
        # Preserve the original solitary-negative ApplyLthBloq defect.
        return traverse(wires[0])
    return controlled_component_ir(
        controls, registers, traverse, workspace[:outer], classical_bits[:outer]
    )
