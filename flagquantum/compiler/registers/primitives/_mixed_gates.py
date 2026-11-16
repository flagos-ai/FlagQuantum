# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Kickmix CircuitBuilder gate dispatch.
# https://github.com/quantumlib/Qualtran/blob/9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453/kickmix/src/kickmix/build/circuit_builder.cc

"""Shared gate dispatch for the locked comparator schedules."""

from ....core.ir import Instruction
from ..finite_field.coefficients import ClassicalFieldCoefficient
from .global_phase import ccz_ir, scalar_sign_ir
from .measurement import measure_x_clear_ir

_Control = int | bool | ClassicalFieldCoefficient


def _three_control_ir(
    first: int,
    second: int,
    third: int,
    target: int,
    *,
    phase: bool,
    scratch: int | None,
    measurement_bit: int | None,
) -> tuple[Instruction, ...]:
    """Share the source CCCX/CCCZ schedule over already admitted operands.

    Callers supply clean scratch in zero, satisfying the source's initial R.
    Otherwise borrow the lowest available wire and restore its entire state.
    """
    helper = (
        scratch
        if scratch is not None
        else next(q for q in range(5) if q not in (first, second, third, target))
    )
    pair = (
        Instruction("ccx", (first, second, helper)),
        *(
            ccz_ir((helper, third, target))
            if phase
            else (Instruction("ccx", (helper, third, target)),)
        ),
    )
    if scratch is None:
        return pair * 2
    assert measurement_bit is not None
    return (
        *pair,
        *measure_x_clear_ir(scratch, measurement_bit),
        Instruction(
            "cz", (first, second), metadata={"conditions": ((measurement_bit, 1),)}
        ),
    )


def _flip(
    target: int | None,
    *controls: _Control,
    scratch: int | None = None,
    measurement_bit: int | None = None,
) -> tuple[Instruction, ...]:
    if any(control is False for control in controls):
        return ()
    wires = tuple(control for control in controls if type(control) is int)
    bits = tuple(
        sorted(
            {
                (control.slot, 1)
                for control in controls
                if isinstance(control, ClassicalFieldCoefficient)
            }
        )
    )
    if target is None:
        if not wires:
            return scalar_sign_ir(0, conditions=bits)
        if len(wires) == 4:
            if scratch is not None and measurement_bit is None:
                raise ValueError("Measured phase requires a classical result slot")
            return _three_control_ir(
                wires[0],
                wires[1],
                wires[2],
                wires[3],
                phase=True,
                scratch=scratch,
                measurement_bit=measurement_bit,
            )
        if len(wires) == 3:
            return ccz_ir((wires[0], wires[1], wires[2]), conditions=bits)
        return (
            Instruction(
                ("z", "cz")[len(wires) - 1],
                wires,
                metadata={"conditions": bits} if bits else {},
            ),
        )
    if len(wires) < 3:
        return (
            Instruction(
                ("x", "cx", "ccx")[len(wires)],
                (*wires, target),
                metadata={"conditions": bits} if bits else {},
            ),
        )
    first, second, last = wires
    if scratch is not None and measurement_bit is None:
        raise ValueError("Measured comparison requires a classical result slot")
    return _three_control_ir(
        first,
        second,
        last,
        target,
        phase=False,
        scratch=scratch,
        measurement_bit=measurement_bit,
    )


def _parity_controlled_flip(
    target: int | None,
    first: _Control,
    second: _Control,
    control: _Control,
    *,
    phase_operand: _Control = True,
    second_control: _Control | None = None,
    scratch: int | None = None,
    measurement_bit: int | None = None,
) -> tuple[Instruction, ...]:
    """Dispatch the source parity gates with one or two additional controls."""
    if target is not None and phase_operand is not True:
        raise ValueError("A separate phase operand requires a phase target")
    phase_controls = (phase_operand,) if target is None else ()
    controls = (control,) if second_control is None else (control, second_control)

    def apply(*operands: _Control) -> tuple[Instruction, ...]:
        return _flip(
            target,
            *operands,
            *phase_controls,
            scratch=scratch,
            measurement_bit=measurement_bit,
        )

    if second_control is None:
        if type(first) is type(second) and first == second:
            return ()
        if (type(control) is type(first) and control == first) or (
            type(control) is type(second) and control == second
        ):
            # (x xor y) & x = (x & y) xor x. Do not overwrite a parity operand.
            return (*apply(first, second), *apply(control))
    for pivot, other in ((first, second), (second, first)):
        if type(pivot) is int:
            if second_control is not None and any(
                type(extra) is int and pivot == extra for extra in controls
            ):
                raise ValueError(
                    "Quantum parity operand overlaps an additional control"
                )
            parity = _flip(pivot, other)
            return (*parity, *apply(pivot, *controls), *parity)
    return (*apply(first, *controls), *apply(second, *controls))


def _broadcast_controlled_flip(
    control: _Control,
    controls: tuple[_Control, ...],
    targets: tuple[int, ...],
    *,
    phase: bool = False,
) -> tuple[Instruction, ...]:
    """Dispatch the source quantum-target broadcast CCX or CCZ in lane order.

    A quantum shared control visits the common length. Classical and true
    shared controls delegate to the equal-length CX/CZ broadcast. A false
    shared control bypasses both emission and the delegated length check.
    """
    if control is False:
        return ()
    if type(control) is not int and len(controls) != len(targets):
        raise ValueError("Broadcast controls and targets must have equal lengths")
    # The locked homogeneous quantum loop packs its lane operand first.
    lane_first = type(control) is int and all(type(lane) is int for lane in controls)
    pairs = (
        ((lane, control) if lane_first else (control, lane), target)
        for lane, target in zip(controls, targets, strict=False)
    )
    return tuple(
        gate
        for pair, target in pairs
        for gate in (_flip(None, *pair, target) if phase else _flip(target, *pair))
    )
