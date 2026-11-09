# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Kickmix gen_iadd1.cc; original carry and cleanup order.
# https://github.com/quantumlib/Qualtran/blob/9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453/kickmix/src/kickmix/gen/adders/gen_iadd1.cc

"""Measured controlled increment with explicit clean carry ownership."""

from ....core.ir import Instruction
from ..finite_field.polynomial import _validate_registers
from .inplace_addition import _erase_carry


def _controlled_increment(
    target: tuple[int, ...],
    clean: tuple[int, ...],
    control: int,
    measurement_bit: int | None,
) -> tuple[Instruction, ...]:
    n = len(target)
    if not n:
        return ()
    if n == 1:
        return (Instruction("cx", (control, target[0])),)
    if n == 2:
        return (
            Instruction("ccx", (control, target[0], target[1])),
            Instruction("cx", (control, target[0])),
        )
    assert measurement_bit is not None
    gates = [Instruction("ccx", (control, target[0], clean[0]))]
    # LoopBuilder places the target-word control before the carry control.
    for k in range(n - 3):
        gates.append(Instruction("ccx", (target[k + 1], clean[k], clean[k + 1])))
    gates.append(Instruction("ccx", (clean[n - 3], target[n - 2], target[n - 1])))
    for k in reversed(range(n - 3)):
        gates.append(Instruction("cx", (clean[k + 1], target[k + 2])))
        gates.extend(
            _erase_carry(clean[k], target[k + 1], clean[k + 1], measurement_bit)
        )
    gates.append(Instruction("cx", (clean[0], target[1])))
    gates.extend(_erase_carry(control, target[0], clean[0], measurement_bit))
    gates.append(Instruction("cx", (control, target[0])))
    return tuple(gates)


def kickmix_increment_ir(
    target: tuple[int, ...],
    *,
    clean: tuple[int, ...] = (),
    control: int | bool = True,
    measurement_bit: int | None = None,
) -> tuple[Instruction, ...]:
    """Increment a big-endian target, preserving control and zero clean carries.

    Supply max(0, n-2) clean wires with a quantum control, or max(0, n-3)
    without one. Measured paths require one reusable classical result slot.
    Extra supplied clean wires remain untouched. False is not an upstream
    QubitOrTrue control and is refused explicitly.
    """
    if control is not True and type(control) is not int:
        raise ValueError("Increment control must be True or a quantum wire")
    _validate_registers(target, clean, (control,) if type(control) is int else ())
    needed = max(0, len(target) - (3 if control is True else 2))
    if len(clean) < needed:
        raise ValueError(f"Increment requires {needed} clean carry wires")
    if measurement_bit is not None and (
        type(measurement_bit) is not int or measurement_bit < 0
    ):
        raise ValueError("Increment requires a nonnegative measurement slot")
    if needed and measurement_bit is None:
        raise ValueError("Measured increment requires a result slot")
    little = target[::-1]
    if control is True:
        if not little:
            return ()
        return (
            *_controlled_increment(little[1:], clean, little[0], measurement_bit),
            Instruction("x", (little[0],)),
        )
    return _controlled_increment(little, clean, control, measurement_bit)
