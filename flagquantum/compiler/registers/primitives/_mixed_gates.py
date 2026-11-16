# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Kickmix CircuitBuilder gate dispatch.
# https://github.com/quantumlib/Qualtran/blob/9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453/kickmix/src/kickmix/build/circuit_builder.cc

"""Shared gate dispatch for the locked comparator schedules."""

from ....core.ir import Instruction
from ..finite_field.coefficients import ClassicalFieldCoefficient
from .global_phase import ccz_ir, scalar_sign_ir
from .measurement import measure_x_clear_ir


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
    *controls: int | bool | ClassicalFieldCoefficient,
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
    first: int | bool | ClassicalFieldCoefficient,
    second: int | bool | ClassicalFieldCoefficient,
    control: int,
) -> tuple[Instruction, ...]:
    if type(first) is type(second) and first == second:
        return ()
    if type(first) is int:
        parity = _flip(first, second)
        return (*parity, *_flip(target, first, control), *parity)
    if type(second) is int:
        parity = _flip(second, first)
        return (*parity, *_flip(target, second, control), *parity)
    return (*_flip(target, first, control), *_flip(target, second, control))
