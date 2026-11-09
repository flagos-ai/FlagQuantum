# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Kickmix gen_bit_rotate.cc.
# https://github.com/quantumlib/Qualtran/blob/9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453/kickmix/src/kickmix/gen/bits/gen_bit_rotate.cc

"""Source-ordered reversals and adjacent-shift parity for register rotations."""

from ....core.classical_dataflow import ClassicalXor
from ....core.ir import Instruction
from ..arithmetic.carry_parity import _Coefficient
from ..finite_field.coefficients import ClassicalFieldCoefficient
from ..finite_field.polynomial import _validate_registers
from ..primitives._mixed_gates import _flip
from ..primitives.measurement import measure_x_clear_ir


def _shift_parity(
    first: _Coefficient, second: _Coefficient
) -> tuple[_Coefficient, tuple[Instruction, ...]]:
    # The source prefers the second wire when both operands are quantum.
    if type(first) is int and type(second) is int:
        return second, _flip(second, first)
    if type(first) is int:
        return first, _flip(first, second)
    if type(second) is int:
        return second, _flip(second, first)
    if isinstance(first, ClassicalFieldCoefficient):
        if isinstance(second, ClassicalFieldCoefficient):
            return second, (
                ClassicalXor((first.slot, second.slot), second.slot).instruction(),
            )
        assert isinstance(second, bool)
        return first, (
            (ClassicalXor((first.slot,), first.slot, constant=True).instruction(),)
            if second
            else ()
        )
    if isinstance(second, ClassicalFieldCoefficient):
        assert isinstance(first, bool)
        return second, (
            (ClassicalXor((second.slot,), second.slot, constant=True).instruction(),)
            if first
            else ()
        )
    return first != second, ()


def _shifted_reverse(
    target: tuple[int, ...], offset: int, control: int | None
) -> tuple[Instruction, ...]:
    if not target:
        return ()
    cut = offset % len(target)
    gates = []
    for part in (target[:cut], target[cut:]):
        for index in range(len(part) // 2):
            pair = (part[-1 - index], part[index])
            gates.append(
                Instruction("swap", pair)
                if control is None
                else Instruction("cswap", (control, *pair))
            )
    return tuple(gates)


def kickmix_bit_rotation_ir(
    target: tuple[int, ...],
    shift: int | tuple[_Coefficient, ...],
    *,
    control: int | None = None,
    right: bool = False,
    clean: tuple[int, ...] = (),
    measurement_bit: int | None = None,
) -> tuple[Instruction, ...]:
    """Rotate a big-endian word by a signed integer or big-endian shift word.

    All quantum ports are distinct. A variable quantum shift combined with a
    quantum enable uses one initially zero clean wire and one result slot.
    Classical shift inputs are defined by the caller and restored after each
    adjacent-bit parity. Shift inputs have at most 60 bits, as in the source.
    Known-zero resets on the supplied work wire are omitted.
    """
    if type(right) is not bool or (control is not None and type(control) is not int):
        raise ValueError("Rotation requires a boolean direction and a quantum control")
    if isinstance(shift, tuple):
        if len(shift) > 60:
            raise ValueError("Rotation shift word cannot exceed 60 bits")
        if any(
            not isinstance(value, (int, ClassicalFieldCoefficient)) for value in shift
        ):
            raise ValueError(
                "Rotation shift coefficients must be bits or quantum wires"
            )
        quantum = tuple(value for value in shift if type(value) is int)
    elif type(shift) is int and -(1 << 63) <= shift < (1 << 63):
        quantum = ()
    else:
        raise ValueError("Rotation shift must be a signed 64-bit integer or a word")
    if right and type(shift) is int and shift == -(1 << 63):
        raise ValueError("Right rotation cannot negate the minimum signed 64-bit shift")
    _validate_registers(target, quantum, clean, () if control is None else (control,))
    low_target = tuple(reversed(target))
    if not isinstance(shift, tuple):
        if not target:
            return ()
        amount = (-shift if right else shift) % len(target)
        if not amount:
            return ()
        return (
            *_shifted_reverse(low_target, 0, control),
            *_shifted_reverse(low_target, amount, control),
        )

    slots = {
        value.slot for value in shift if isinstance(value, ClassicalFieldCoefficient)
    }
    merged = control is not None and bool(quantum)
    if merged and (
        not clean
        or type(measurement_bit) is not int
        or measurement_bit < 0
        or measurement_bit in slots
    ):
        raise ValueError(
            "Merged rotation controls require a clean wire and a fresh result slot"
        )
    low_shift = tuple(reversed(shift))
    gates: list[Instruction] = []
    for index in range(len(shift) + 1):
        first = low_shift[index - 1] if index else False
        second = low_shift[index] if index < len(shift) else False
        parity, prefix = _shift_parity(first, second)
        gates.extend(prefix)
        active_control = control
        conditions: tuple[tuple[int, int], ...] = ()
        if type(parity) is int:
            if control is None:
                active_control = parity
            else:
                active_control = clean[0]
                gates.extend(_flip(active_control, parity, control))
        elif isinstance(parity, ClassicalFieldCoefficient):
            conditions = ((parity.slot, 1),)
        if parity is not False:
            body = _shifted_reverse(
                low_target, (-1 if right else 1) * (1 << index) + 1, active_control
            )
            gates.extend(
                (
                    Instruction(
                        gate.name, gate.wires, metadata={"conditions": conditions}
                    )
                    if conditions
                    else gate
                )
                for gate in body
            )
        if type(parity) is int and control is not None:
            assert measurement_bit is not None
            gates.extend(measure_x_clear_ir(clean[0], measurement_bit))
            gates.append(
                Instruction(
                    "cz",
                    (parity, control),
                    metadata={"conditions": ((measurement_bit, 1),)},
                )
            )
        gates.extend(prefix)
    return tuple(gates)
