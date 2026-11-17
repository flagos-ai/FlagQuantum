# SPDX-License-Identifier: Apache-2.0
# Adapted from the locked Kickmix left/right rotation binding.
# https://github.com/quantumlib/Qualtran/blob/9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453/kickmix/src/kickmix/py/circuit/subs/bit_rotate.pybind.cc

"""Bind caller-owned ports and flags to the shared register rotation."""

from dataclasses import replace

from ....core.classical_dataflow import ClassicalXor
from ....core.ir import Instruction
from ..arithmetic.carry_parity import _Coefficient
from ..finite_field.coefficients import ClassicalFieldCoefficient
from ..finite_field.polynomial import _validate_registers
from .bit_rotation import bit_rotation_ir


def bound_bit_rotation_ir(
    target: tuple[int, ...],
    shift: int | tuple[_Coefficient, ...] = 1,
    *,
    control: _Coefficient = True,
    right: bool = False,
    clean: tuple[int, ...] = (),
    measurement_bit: int | None = None,
    predicate_bit: int | None = None,
) -> tuple[Instruction, ...]:
    """Bind rotation ports, preserving the source entry-time enable value.

    Clean work and result slots are supplied by the caller. If a classical
    enable can be overwritten by shift parity, predicate_bit supplies a fresh
    temporary classical slot for its entry value. The slot is cleared on exit.
    """
    _validate_registers(target)
    if control is False:
        return ()
    if not isinstance(control, (int, ClassicalFieldCoefficient)):
        raise ValueError("Rotation control must be a literal, classical bit or wire")
    body = bit_rotation_ir(
        target,
        int(shift) if isinstance(shift, bool) else shift,
        control=control if type(control) is int else None,
        right=right,
        clean=clean,
        measurement_bit=measurement_bit,
    )
    if not isinstance(control, ClassicalFieldCoefficient) or not body:
        return body
    slot = control.slot
    writes_enable = any(
        gate.name == "classical_xor" and gate.metadata["classical_bit"] == slot
        for gate in body
    )
    if writes_enable:
        assert isinstance(shift, tuple)
        input_slots = {
            bit.slot for bit in shift if isinstance(bit, ClassicalFieldCoefficient)
        }
        if (
            type(predicate_bit) is not int
            or predicate_bit < 0
            or predicate_bit in input_slots
            or predicate_bit == measurement_bit
        ):
            raise ValueError("Aliased rotation control requires a fresh predicate slot")
        slot = predicate_bit

    guarded = tuple(
        replace(
            gate,
            metadata={
                **gate.metadata,
                "conditions": tuple(
                    sorted({*gate.metadata.get("conditions", ()), (slot, 1)})
                ),
            },
        )
        for gate in body
    )
    if writes_enable:
        return (
            ClassicalXor((control.slot,), slot).instruction(),
            *guarded,
            ClassicalXor((), slot).instruction(),
        )
    return guarded
