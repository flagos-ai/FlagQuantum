# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Kickmix gen_iadd_control.cc; original carry/cleanup order.
# https://github.com/quantumlib/Qualtran/blob/9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453/kickmix/src/kickmix/gen/adders/gen_iadd_control.cc

"""Controlled quantum addition with a constant, classical or quantum carry."""

from ...core.ir import Instruction
from ._kickmix_comparison_gates import _flip
from .finite_field import ClassicalFieldCoefficient
from .kickmix_addition import _erase_carry
from .measurement import measure_x_clear_ir
from .polynomial import _validate_registers


def _add_with_carry(
    target: tuple[int, ...],
    offset: tuple[int, ...],
    control: int,
    carry: int | bool | ClassicalFieldCoefficient,
    clean: tuple[int, ...],
    measurement_bit: int | None,
) -> tuple[Instruction, ...]:
    n = len(target)
    if not n:
        return ()
    if n == 1:
        return (
            *_flip(target[0], control, offset[0]),
            *_flip(target[0], control, carry),
        )
    assert measurement_bit is not None
    outcome = ClassicalFieldCoefficient(measurement_bit)
    gates = [
        *_flip(target[0], carry),
        *_flip(offset[0], carry),
        *_flip(clean[0], target[0], offset[0]),
        *_flip(clean[0], carry),
    ]
    for k in range(1, n - 1):
        gates.extend(
            (
                *_flip(target[k], clean[k - 1]),
                *_flip(offset[k], clean[k - 1]),
                *_flip(clean[k], offset[k], target[k]),
                *_flip(clean[k], clean[k - 1]),
            )
        )
    gates.extend(
        (
            *_flip(clean[n - 2], offset[n - 1]),
            *_flip(target[n - 1], control, clean[n - 2]),
            *_flip(clean[n - 2], offset[n - 1]),
        )
    )
    for k in reversed(range(1, n - 1)):
        gates.extend(
            (
                *_erase_carry(target[k], offset[k], clean[k], measurement_bit),
                *_flip(None, clean[k - 1], outcome),
                *_flip(target[k], offset[k], control),
                *_flip(target[k], clean[k - 1]),
                *_flip(offset[k], clean[k - 1]),
            )
        )
    gates.extend(
        (
            *measure_x_clear_ir(clean[0], measurement_bit),
            *_flip(None, target[0], offset[0], outcome),
            *_flip(None, carry, outcome),
            *_flip(target[0], control, offset[0]),
            *_flip(target[0], carry),
            *_flip(offset[0], carry),
        )
    )
    return tuple(gates)


def kickmix_controlled_addition_ir(
    target: tuple[int, ...],
    offset: tuple[int, ...],
    *,
    control: int,
    carry_in: int | bool | ClassicalFieldCoefficient = False,
    clean: tuple[int, ...] = (),
    measurement_bit: int | None = None,
    subtract: bool = False,
) -> tuple[Instruction, ...]:
    """Update target under a quantum enable and restore offset, carry and work.

    Target and offset are big-endian; extra high offset bits are preserved.
    Width n >= 2 needs n clean wires and a reusable, separate result slot.
    Subtraction follows the source carry-false NOT conjugation only.
    """
    if type(carry_in) not in (int, bool) and not isinstance(
        carry_in, ClassicalFieldCoefficient
    ):
        raise ValueError(
            "Carry must be a constant, quantum wire or classical coefficient"
        )
    _validate_registers(
        target, offset, clean, (control,), (carry_in,) if type(carry_in) is int else ()
    )
    if len(target) > len(offset):
        raise ValueError(
            "Controlled addition requires offset at least as wide as target"
        )
    if type(subtract) is not bool or (subtract and carry_in is not False):
        raise ValueError("Source subtraction requires a false carry input")
    n = len(target)
    if n >= 2 and len(clean) < n:
        raise ValueError(
            "Controlled addition requires one clean carry wire per target bit"
        )
    if measurement_bit is not None and (
        type(measurement_bit) is not int
        or measurement_bit < 0
        or (
            isinstance(carry_in, ClassicalFieldCoefficient)
            and measurement_bit == carry_in.slot
        )
    ):
        raise ValueError(
            "Controlled addition requires a separate nonnegative result slot"
        )
    if n >= 2 and measurement_bit is None:
        raise ValueError("Measured controlled addition requires a result slot")
    little = target[::-1]
    gates = _add_with_carry(
        little, offset[::-1], control, carry_in, clean, measurement_bit
    )
    invert = tuple(Instruction("x", (wire,)) for wire in little) if subtract else ()
    return (*invert, *gates, *invert)
