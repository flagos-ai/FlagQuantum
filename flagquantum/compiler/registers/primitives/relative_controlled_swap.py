# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 CSwapApprox.decompose_from_registers.
# Source: https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/swap_network/cswap_approx.py

"""Controlled exchange with the source's relative signs."""

from __future__ import annotations

from dataclasses import dataclass

from ....core.ir import Instruction
from ....core.ir.program.registers import Register
from ....core.ir.program.signatures import Signature
from ..._component_trace.resource_decomposition import decomposed_resource_callees
from ..._component_trace.resource_generalization import (
    CliffordCall,
    ControlledSwapCall,
    ControlPredicateCall,
    ResourceCall,
)
from ..._component_trace.symbolic_math import SymbolicInt
from ..bits.bitwise import _validate_wires, fanout_ir
from ..control.control_predicate import ControlRegister
from ..control.controlled_component import controlled_component_ir
from ..phase.phase_family import phase_family_ir

_INVERSE_PHASE = {"s": "sdg", "sdg": "s", "t": "tdg", "tdg": "t"}


def relative_swap_signature(
    width: SymbolicInt, *, extra_controls: tuple[Register, ...] = ()
) -> Signature:
    """Declare the original swap ports after any generic controls."""
    return Signature((*extra_controls, *Signature.build(ctrl=1, x=width, y=width)))


@dataclass(frozen=True)
class PositiveControlledSwapChild:
    """One child of the source call graph under a positive scalar control."""

    child: ResourceCall


def source_outer_relative_swap_calls(
    width: SymbolicInt,
    registers: tuple[ControlRegister, ...],
    *,
    scalar_control: bool = False,
    adjoint: bool = False,
) -> dict[
    PositiveControlledSwapChild | ControlPredicateCall | CliffordCall, SymbolicInt
]:
    """Declare the wrapper's direct edges before expanding opaque Cliffords."""
    values = tuple(value for register in registers for value in register.bits())
    if not values:
        raise ValueError("Outer relative swap control requires a nonempty activation")
    swap = ControlledSwapCall(width, approximate=True, adjoint=adjoint)
    if scalar_control and values == (1,):
        return {
            PositiveControlledSwapChild(child): count
            for child, count in decomposed_resource_callees(swap, None)
        }
    calls: dict[
        PositiveControlledSwapChild | ControlPredicateCall | CliffordCall, SymbolicInt
    ] = {PositiveControlledSwapChild(swap): 1}
    if values == (0,):
        calls[CliffordCall("x", 1)] = 2
    elif len(values) > 1:
        calls[ControlPredicateCall(values)] = 1
        calls[ControlPredicateCall(values, adjoint=True)] = 1
    return calls


def relative_controlled_swap_ir(
    control: int,
    left: tuple[int, ...],
    right: tuple[int, ...],
    *,
    adjoint: bool = False,
) -> tuple[Instruction, ...]:
    """Exchange equal-width words under control, retaining source relative phases.

    The fanout conjugation is self-adjoint and uses four T gates per pair.
    The generic source adjoint reverses the instruction dependencies.
    """
    _validate_wires((control, *left, *right))
    return _relative_swap_gates(control, left, right, adjoint=adjoint)


def _relative_swap_gates(
    control: int,
    left: tuple[int, ...],
    right: tuple[int, ...],
    *,
    adjoint: bool,
    layer_rotations: bool = False,
) -> tuple[Instruction, ...]:
    if type(adjoint) is not bool:
        raise ValueError("Relative controlled swap requires a boolean adjoint flag")
    before, after = _relative_swap_parts(left, right, layer_rotations=layer_rotations)
    gates = (*before, *fanout_ir(control, right), *after)
    if not adjoint:
        return gates
    return tuple(
        Instruction(_INVERSE_PHASE.get(gate.name, gate.name), gate.wires)
        for gate in reversed(gates)
    )


def _relative_swap_parts(
    left: tuple[int, ...],
    right: tuple[int, ...],
    *,
    layer_rotations: bool = False,
) -> tuple[tuple[Instruction, ...], tuple[Instruction, ...]]:
    """Share the literal instruction segments around the fanout call."""
    if not left or len(left) != len(right):
        raise ValueError("Relative controlled swap requires equal nonempty words")
    right_to_left = tuple(
        Instruction("cx", (y, x)) for x, y in zip(left, right, strict=True)
    )
    left_to_right = tuple(
        Instruction("cx", (x, y)) for x, y in zip(left, right, strict=True)
    )

    def rotate(phase: str) -> tuple[Instruction, ...]:
        sequence = ("s", "h", phase, "h", "sdg")
        if layer_rotations:
            # Qualtran's generic control first imports the Cirq moments into
            # its component graph, then threads the new control through them.
            return tuple(
                Instruction(gate, (wire,)) for gate in sequence for wire in right
            )
        return tuple(Instruction(gate, (wire,)) for wire in right for gate in sequence)

    inverse_rotation, rotation = rotate("tdg"), rotate("t")
    before = (
        *right_to_left,
        *inverse_rotation,
        *left_to_right,
        *inverse_rotation,
    )
    after = (
        *rotation,
        *left_to_right,
        *rotation,
        *right_to_left,
    )

    return before, after


def control_relative_swap_ir(
    controls: tuple[tuple[int, ...], ...],
    registers: tuple[ControlRegister, ...],
    original_control: int,
    left: tuple[int, ...],
    right: tuple[int, ...],
    workspace: tuple[int, ...],
    classical_bits: tuple[int, ...],
    *,
    adjoint: bool = False,
) -> tuple[Instruction, ...]:
    """Add source generic controls while retaining the original relative phases."""
    _validate_wires(
        (
            original_control,
            *left,
            *right,
            *workspace,
            *(q for word in controls for q in word),
        )
    )
    gates = _relative_swap_gates(
        original_control, left, right, adjoint=adjoint, layer_rotations=True
    )

    def emit(control: int) -> tuple[Instruction, ...]:
        result: list[Instruction] = []
        for gate in gates:
            if gate.name == "cx":
                result.append(Instruction("ccx", (control, *gate.wires)))
            elif gate.name == "h":
                result.append(Instruction("ch", (control, *gate.wires)))
            else:
                result.extend(
                    phase_family_ir(
                        "s" if gate.name in ("s", "sdg") else "t",
                        gate.wires[0],
                        adjoint=gate.name in ("sdg", "tdg"),
                        controls=(control,),
                        values=(1,),
                    )
                )
        return tuple(result)

    return controlled_component_ir(controls, registers, emit, workspace, classical_bits)
