# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Kickmix gen_iadd.cc and CircuitBuilder.
# https://github.com/quantumlib/Qualtran/blob/9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453/kickmix/src/kickmix/gen/adders/gen_iadd.cc

"""In-place quantum addition and subtraction with the locked carry schedule."""

from ....core.ir import Instruction
from ..finite_field.coefficients import ClassicalFieldCoefficient
from ..finite_field.polynomial import _validate_registers
from ..primitives._mixed_gates import _flip
from ..primitives.measurement import measure_x_clear_ir


def _erase_carry(
    first: int, second: int, carry: int, measurement_bit: int
) -> tuple[Instruction, ...]:
    return (
        *measure_x_clear_ir(carry, measurement_bit),
        *_flip(None, first, second, ClassicalFieldCoefficient(measurement_bit)),
    )


def _carry_schedule(
    target: tuple[int, ...],
    offset: tuple[int, ...],
    clean: tuple[int, ...],
    carry: int,
    control: int | bool,
    measurement_bit: int | None,
    minimize_qubits: bool,
) -> tuple[Instruction, ...]:
    top = len(target) - 1
    clean_stop = max(1, min(1 if minimize_qubits else len(clean), top - 1))
    gates: list[Instruction] = []
    if not clean:
        for wire in (*target[1:], *offset[1:]):
            gates.extend(_flip(wire, carry))

    gates.extend(
        (
            *_flip(carry, target[0], offset[0]),
            *_flip(offset[1], carry),
            *_flip(target[1], carry),
        )
    )
    # The source LoopBuilder emits offset before target in its CCX operands.
    for k in range(1, top - 1):
        if k < clean_stop:
            gates.extend(_flip(clean[k], offset[k], target[k]))
            gates.extend(_flip(carry, clean[k]))
        else:
            gates.extend(_flip(carry, offset[k], target[k]))
        gates.extend(_flip(offset[k + 1], carry))
        gates.extend(_flip(target[k + 1], carry))

    gates.extend(_flip(target[top], control, carry))
    gates.extend(_flip(target[top], control, target[top - 1], offset[top - 1]))
    for k in reversed(range(1, top - 1)):
        gates.extend(_flip(target[k + 1], offset[k + 1], control))
        gates.extend(_flip(target[k + 1], carry))
        gates.extend(_flip(offset[k + 1], carry))
        if k < clean_stop:
            assert measurement_bit is not None
            gates.extend(_flip(carry, clean[k]))
            gates.extend(_erase_carry(target[k], offset[k], clean[k], measurement_bit))
        else:
            gates.extend(_flip(carry, offset[k], target[k]))

    gates.extend(_flip(target[1], control, offset[1]))
    gates.extend(_flip(target[1], carry))
    gates.extend(_flip(offset[1], carry))
    if clean:
        assert measurement_bit is not None
        gates.extend(_erase_carry(target[0], offset[0], carry, measurement_bit))
    else:
        gates.extend(_flip(carry, target[0], offset[0]))
    gates.extend(_flip(target[0], control, offset[0]))
    if not clean:
        for wire in (*offset[1:], *target[1:]):
            gates.extend(_flip(wire, carry))
        gates.extend(_flip(target[top], control, carry))
    return tuple(gates)


def _addition(
    target: tuple[int, ...],
    offset: tuple[int, ...],
    clean: tuple[int, ...],
    dirty: tuple[int, ...],
    control: int | bool,
    measurement_bit: int | None,
    minimize_qubits: bool,
) -> tuple[Instruction, ...]:
    n, m = len(target), len(offset)
    if not n or not m:
        return ()
    if n == 1:
        return _flip(target[0], control, offset[0])
    if n == 2:
        scratch = (
            clean[0] if type(control) is int and clean and not minimize_qubits else None
        )
        if scratch is not None and measurement_bit is None:
            raise ValueError("Kickmix addition requires a reusable measurement slot")
        return (
            *_flip(
                target[1],
                control,
                offset[0],
                target[0],
                scratch=scratch,
                measurement_bit=measurement_bit,
            ),
            *_flip(target[0], control, offset[0]),
            *(_flip(target[1], control, offset[1]) if m > 1 else ()),
        )

    prefix: tuple[Instruction, ...] = ()
    if n <= m:
        prefix = _flip(target[-1], control, offset[n - 1])
        # The native workspace stack borrows from the newest span first.
        dirty = offset[n - 1 :]
        offset = offset[: n - 1]
    padding = n - len(offset) - 1
    if padding > len(clean):
        raise ValueError(
            "Kickmix addition requires clean wires to zero-extend the offset"
        )
    offset = (*offset, *clean[:padding])
    clean = clean[padding:]
    if not clean and not dirty:
        raise ValueError("Kickmix addition requires a clean or dirty carry wire")
    if clean and measurement_bit is None:
        raise ValueError("Kickmix addition requires a reusable measurement slot")
    carry = clean[0] if clean else dirty[-1]
    return (
        *prefix,
        *_carry_schedule(
            target, offset, clean, carry, control, measurement_bit, minimize_qubits
        ),
    )


def kickmix_inplace_addition_ir(
    target: tuple[int, ...],
    offset: tuple[int, ...],
    *,
    clean: tuple[int, ...] = (),
    dirty: tuple[int, ...] = (),
    control: int | bool = True,
    measurement_bit: int | None = None,
    minimize_qubits: bool = False,
    subtract: bool = False,
) -> tuple[Instruction, ...]:
    """Add or subtract the big-endian offset modulo the target register size.

    The caller supplies pre-cleared padding and clean carry wires, dirty
    wires in borrowing order, and one reusable result slot for the measured
    schedule. The offset and workspace are restored. The locked no-clean
    three-control gate can also borrow the lowest non-operand wire ID, which
    must be present in the containing circuit. Subtraction conjugates the
    same schedule by bitwise NOT on the target.
    """
    _validate_registers(
        target, offset, clean, dirty, (control,) if type(control) is int else ()
    )
    if measurement_bit is not None and (
        type(measurement_bit) is not int or measurement_bit < 0
    ):
        raise ValueError("Kickmix addition requires a nonnegative measurement slot")
    if control is False:
        return ()
    left, right = target[::-1], offset[::-1]
    gates = _addition(
        left, right, clean, dirty, control, measurement_bit, minimize_qubits
    )
    invert = tuple(Instruction("x", (wire,)) for wire in left) if subtract else ()
    return (*invert, *gates, *invert)
