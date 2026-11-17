# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Kickmix CircuitBuilder gate dispatch.
# https://github.com/quantumlib/Qualtran/blob/9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453/kickmix/src/kickmix/build/circuit_builder.cc

"""Shared gate dispatch for the locked comparator schedules."""

from dataclasses import dataclass

from ....core.classical_dataflow import ClassicalXor
from ....core.ir import Instruction
from ..finite_field.coefficients import ClassicalFieldCoefficient
from .global_phase import ccz_ir, scalar_sign_ir
from .measurement import measure_x_clear_ir

_Control = int | bool | ClassicalFieldCoefficient


@dataclass(frozen=True)
class _XBasisTarget:
    """Known or measured X eigenvalue, true denoting the minus eigenstate."""

    outcome: bool | ClassicalFieldCoefficient


def _three_control_ir(
    first: int,
    second: int,
    third: int,
    target: int,
    *,
    phase: bool,
    scratch: int | None,
    measurement_bit: int | None,
    conditions: tuple[tuple[int, int], ...] = (),
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
    metadata = {"conditions": conditions} if conditions else {}
    pair = (
        Instruction("ccx", (first, second, helper), metadata=metadata),
        *(
            ccz_ir((helper, third, target), conditions=conditions)
            if phase
            else (Instruction("ccx", (helper, third, target), metadata=metadata),)
        ),
    )
    if scratch is None:
        return pair * 2
    assert measurement_bit is not None
    return (
        *pair,
        *measure_x_clear_ir(scratch, measurement_bit, conditions=conditions),
        Instruction(
            "cz",
            (first, second),
            metadata={"conditions": tuple(sorted((*conditions, (measurement_bit, 1))))},
        ),
    )


def _flip(
    target: int | _XBasisTarget | None,
    *controls: _Control,
    scratch: int | None = None,
    measurement_bit: int | None = None,
) -> tuple[Instruction, ...]:
    if isinstance(target, _XBasisTarget):
        controls = (*controls, target.outcome)
        target = None
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
    if len(wires) > (4 if target is None else 3):
        raise ValueError("Mixed control count exceeds the source schedule")
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
                conditions=bits,
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
        conditions=bits,
    )


def _delete_and_ir(
    target: int, bit: int, first: _Control, second: _Control
) -> tuple[Instruction, ...]:
    """Erase a target storing the mixed AND with X-basis phase feedback."""
    if any(
        isinstance(control, ClassicalFieldCoefficient) and control.slot == bit
        for control in (first, second)
    ):
        raise ValueError("AND erasure result cannot overwrite a control")
    return (
        *measure_x_clear_ir(target, bit),
        *_flip(_XBasisTarget(ClassicalFieldCoefficient(bit)), first, second),
    )


def _invert_operand(value: _Control) -> tuple[_Control, tuple[Instruction, ...]]:
    """Complement a literal or toggle a physical operand without rebinding it."""
    if isinstance(value, bool):
        return not value, ()
    if isinstance(value, ClassicalFieldCoefficient):
        return value, (
            ClassicalXor((value.slot,), value.slot, constant=True).instruction(),
        )
    return value, _flip(value)


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


def _is_z_operand(value: object) -> bool:
    return type(value) in (int, bool) or isinstance(value, ClassicalFieldCoefficient)


def _broadcast_cx(
    controls: tuple[_Control, ...],
    targets: tuple[int | ClassicalFieldCoefficient | _XBasisTarget, ...],
) -> tuple[Instruction, ...]:
    """Apply ordered CX lanes, with phase kickback for X-basis targets."""
    if len(controls) != len(targets):
        raise ValueError("Broadcast controls and targets must have equal lengths")
    for control, target in zip(controls, targets, strict=True):
        if not _is_z_operand(control):
            raise ValueError("CX requires a Z-basis control")
        if isinstance(target, ClassicalFieldCoefficient) and type(control) is int:
            raise ValueError("Quantum control cannot flip a classical target")
        if type(target) is bool:
            raise ValueError("CX cannot flip a compile-time Boolean target")

    gates: list[Instruction] = []
    for control, target in zip(controls, targets, strict=True):
        if isinstance(target, _XBasisTarget):
            gates.extend(_flip(target, control))
        elif isinstance(target, ClassicalFieldCoefficient):
            if control is True:
                gates.append(
                    ClassicalXor(
                        (target.slot,), target.slot, constant=True
                    ).instruction()
                )
            elif isinstance(control, ClassicalFieldCoefficient):
                gates.append(
                    ClassicalXor((target.slot, control.slot), target.slot).instruction()
                )
        else:
            gates.extend(_flip(target, control))
    return tuple(gates)


def _broadcast_cz(
    left: tuple[_Control, ...], right: tuple[_Control, ...]
) -> tuple[Instruction, ...]:
    """Apply one controlled sign per ordered pair of Z-basis controls."""
    if len(left) != len(right):
        raise ValueError("Broadcast controls and targets must have equal lengths")
    if not all(
        _is_z_operand(value) for operands in (left, right) for value in operands
    ):
        raise ValueError("CZ requires Z-basis controls")
    return tuple(
        gate
        for first, second in zip(left, right, strict=True)
        for gate in _flip(None, first, second)
    )


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
    if control is True:
        return (
            _broadcast_cz(controls, targets)
            if phase
            else _broadcast_cx(controls, targets)
        )
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
