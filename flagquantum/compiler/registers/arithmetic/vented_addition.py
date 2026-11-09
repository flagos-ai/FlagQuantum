# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Kickmix gen_iadd_classical.cc.
# https://github.com/quantumlib/Qualtran/blob/9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453/kickmix/src/kickmix/gen/adders/gen_iadd_classical.cc

"""Two-clean-wire mixed addition with measured intermediate carries."""

from ....core.classical_dataflow import ClassicalXor
from ....core.ir import Instruction
from ..finite_field.coefficients import ClassicalFieldCoefficient
from ..finite_field.polynomial import _validate_registers
from ..primitives._mixed_gates import _flip
from ..primitives.measurement import measure_x_clear_ir
from .carry_parity import _broadcast_offset, _carry_parity_gate, _Coefficient


def _vented_carries(
    target: tuple[int, ...],
    offset: tuple[_Coefficient, ...],
    carry_in: _Coefficient,
    control: int | bool,
    clean: tuple[int, ...],
    vents: tuple[int, ...],
    measurement_bit: int | None,
) -> tuple[Instruction, ...]:
    n = len(target)
    a, b = clean[:2]
    parity = (offset[0], carry_in)
    if n == 2:
        return (
            *_carry_parity_gate(
                target[1], parity, (control, target[0]), clean, measurement_bit
            ),
            *_flip(target[1], control, offset[0]),
            *_flip(target[0], control, carry_in),
        )
    if measurement_bit is None:
        raise ValueError("Vented addition requires a reusable measurement slot")
    outcome = ClassicalFieldCoefficient(measurement_bit)
    if n == 3:
        return (
            *_carry_parity_gate(
                a, parity, (control, target[0]), clean[1:], measurement_bit
            ),
            *_flip(a, control, offset[0]),
            *_flip(target[1], a),
            *_flip(a, control, offset[1]),
            *_flip(target[2], a, target[1]),
            *_flip(target[2], a),
            *_flip(target[2], a, offset[1]),
            *_flip(target[2], control, offset[1]),
            *measure_x_clear_ir(a, measurement_bit),
            *_carry_parity_gate(None, parity, (control, target[0], outcome), (), None),
            *_flip(None, control, offset[0], outcome),
            *_flip(None, control, offset[1], outcome),
            *_flip(target[0], control, carry_in),
        )

    gates = [
        *_carry_parity_gate(a, parity, (control, target[0]), (), None),
        *_flip(target[0], control, carry_in),
        *_flip(a, control, offset[0]),
    ]
    for k in range(1, n - 3):
        vent = vents[k]
        gates.extend(
            (
                Instruction("swap", (a, b)),
                *_flip(b, control, offset[k]),
                *_flip(a, target[k], b),
                *_flip(b, control, offset[k]),
                *_flip(target[k], b),
                *measure_x_clear_ir(b, measurement_bit),
                ClassicalXor((vent, measurement_bit), vent).instruction(),
                *_flip(a, control, offset[k]),
            )
        )

    third_last, penultimate, last = target[-3:]
    third_offset, penultimate_offset = offset[n - 3 : n - 1]
    final_vent = vents[n - 3]
    gates.extend(
        (
            *_flip(a, control, third_offset),
            *_flip(b, third_last, a),
            *_flip(a, control, third_offset),
            *_flip(third_last, a),
            *_flip(b, control, third_offset),
            *_flip(b, control, penultimate_offset),
            *_flip(last, penultimate, b),
            *_flip(b, control, penultimate_offset),
            *_flip(penultimate, b),
            *measure_x_clear_ir(b, measurement_bit),
            *_flip(None, third_offset, third_last, control, outcome),
            *_flip(None, third_offset, a, control, outcome),
            *_flip(None, third_offset, control, outcome),
            *_flip(None, third_last, a, outcome),
            *_flip(None, a, outcome),
            *_flip(last, control, penultimate_offset),
            *measure_x_clear_ir(a, measurement_bit),
            ClassicalXor((final_vent, measurement_bit), final_vent).instruction(),
        )
    )
    return tuple(gates)


def kickmix_vented_addition_ir(
    target: tuple[int, ...],
    offset: tuple[_Coefficient, ...],
    *,
    clean: tuple[int, ...],
    carry_copy: tuple[int | None, ...] = (),
    vent_bits: tuple[int, ...] = (),
    carry_in: _Coefficient = False,
    control: int | bool = True,
    measurement_bit: int | None = None,
) -> tuple[Instruction, ...]:
    """Add a mixed offset and expose the locked carry XOR and measurement effects.

    Words are big-endian. A short copy word starts at the least significant
    end of the target. Its least significant port may be None. An additional
    high copy port becomes the high sum bit. The caller owns two pre-cleared
    work wires, the reusable result slot, and the vent slots in source order.
    Used vent slots must already be defined and receive XOR updates.
    """
    if control is False:
        return ()
    work, copy = target[::-1], carry_copy[::-1]
    if None in copy[1:] or (len(copy) > len(work) and None in copy):
        raise ValueError("Only a non-promoted low carry-copy port may be omitted")
    copy_wires = tuple(wire for wire in copy if wire is not None)
    quantum_inputs = tuple(
        dict.fromkeys(value for value in (*offset, carry_in) if type(value) is int)
    )
    _validate_registers(
        work,
        copy_wires,
        clean,
        quantum_inputs,
        (control,) if type(control) is int else (),
    )
    if len(copy) > len(work):
        high = copy[len(work)]
        assert high is not None
        copy, work = copy[: len(work)], (*work, high)
    n = len(work)
    if len(clean) < 2 or len(vent_bits) < n - 2:
        raise ValueError(
            "Vented addition requires two clean wires and n - 2 vent slots"
        )
    classical_inputs = {
        value.slot
        for value in (*offset, carry_in)
        if isinstance(value, ClassicalFieldCoefficient)
    }
    slots = (*vent_bits, *((measurement_bit,) if measurement_bit is not None else ()))
    if len(set(slots)) != len(slots) or any(
        type(bit) is not int or bit < 0 or bit in classical_inputs for bit in slots
    ):
        raise ValueError(
            "Vented addition requires distinct nonnegative classical slots"
        )
    if n == 0:
        return ()
    rhs = (*reversed(offset), *((False,) * max(0, n - len(offset))))
    copy_gates = tuple(
        Instruction("cx", (wire, dest))
        for wire, dest in zip(work, copy, strict=False)
        if dest is not None
    )
    if n == 1:
        return (
            *_flip(work[0], control, rhs[0]),
            *copy_gates,
            *_flip(work[0], control, carry_in),
            *copy_gates,
        )
    if control is True and len(rhs) != n:
        raise ValueError("Locked unconditional vented addition requires equal widths")
    return (
        *_broadcast_offset(work, rhs, control, all(type(bit) is int for bit in rhs)),
        *copy_gates,
        *_vented_carries(
            work, rhs, carry_in, control, clean, vent_bits, measurement_bit
        ),
        *copy_gates,
    )
