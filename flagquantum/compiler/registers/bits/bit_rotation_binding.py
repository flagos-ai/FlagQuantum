# SPDX-License-Identifier: Apache-2.0
# Adapted from the locked Kickmix left/right rotation binding.
# https://github.com/quantumlib/Qualtran/blob/9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453/kickmix/src/kickmix/py/circuit/subs/bit_rotate.pybind.cc

"""Bind caller-owned ports and flags to the shared register rotation."""

from ....core.classical_dataflow import ClassicalXor
from ....core.ir import Instruction
from ..arithmetic.carry_parity import _Coefficient
from ..finite_field.coefficients import ClassicalFieldCoefficient
from ..finite_field.polynomial import _validate_registers
from ..primitives.control_binding import apply_control_guards, bind_z_enable
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
    enable, slots = bind_z_enable(control)
    if enable is False:
        return ()
    body = bit_rotation_ir(
        target,
        int(shift) if isinstance(shift, bool) else shift,
        control=enable if type(enable) is int else None,
        right=right,
        clean=clean,
        measurement_bit=measurement_bit,
    )
    if not slots or not body:
        return body
    (slot,) = slots
    entry_slot = slot
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

    guarded = apply_control_guards(body, (slot,))
    if writes_enable:
        return (
            ClassicalXor((entry_slot,), slot).instruction(),
            *guarded,
            ClassicalXor((), slot).instruction(),
        )
    return guarded
