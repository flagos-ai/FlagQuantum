# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Kickmix gen_iadd_classical.cc and CircuitBuilder.
# https://github.com/quantumlib/Qualtran/blob/9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453/kickmix/src/kickmix/gen/adders/gen_iadd_classical.cc

"""XOR mixed addition carries into a caller-owned quantum word."""

from ...core.ir import Instruction
from ._kickmix_comparison_gates import _flip
from .finite_field import ClassicalFieldCoefficient
from .polynomial import _validate_registers

_Coefficient = int | bool | ClassicalFieldCoefficient


def _carry_parity_xor(
    target: int,
    parity: tuple[_Coefficient, _Coefficient],
    control: int | bool,
    source: int,
    clean: tuple[int, ...],
    measurement_bit: int | None,
) -> tuple[Instruction, ...]:
    for first, second in (parity, parity[::-1]):
        if type(first) is int:
            update = _flip(first, second)
            scratch = clean[0] if clean and type(control) is int else None
            if scratch is not None and measurement_bit is None:
                raise ValueError("Carry XOR requires a reusable measurement slot")
            return (
                *update,
                *_flip(
                    target,
                    first,
                    control,
                    source,
                    scratch=scratch,
                    measurement_bit=measurement_bit,
                ),
                *update,
            )
    return (
        *_flip(target, parity[0], control, source),
        *_flip(target, parity[1], control, source),
    )


def _broadcast_offset(
    target: tuple[int, ...],
    offset: tuple[_Coefficient, ...],
    control: int,
    quantum_offset: bool,
) -> tuple[Instruction, ...]:
    # The native all-quantum LoopBuilder uses coefficient before enable.
    return tuple(
        gate
        for wire, coefficient in zip(target, offset, strict=False)
        for gate in _flip(
            wire,
            *((coefficient, control) if quantum_offset else (control, coefficient)),
        )
    )


def kickmix_carry_xor_ir(
    source: tuple[int, ...],
    offset: tuple[_Coefficient, ...],
    target: tuple[int, ...],
    *,
    carry_in: _Coefficient = False,
    control: int | bool = True,
    clean: tuple[int, ...] = (),
    measurement_bit: int | None = None,
) -> tuple[Instruction, ...]:
    """XOR each outgoing addition carry into its corresponding target bit.

    Words are big-endian and the target selects the least significant source
    positions. An absent offset bit is zero. Offset and carry-in may be
    constants, defined classical slots or quantum wires. The clean carry, if
    used, starts and ends at zero and reuses the caller's measurement slot.
    A coherent three-control gate may borrow the lowest non-operand wire ID.
    The locked multi-bit path requires a quantum enable because the
    unconditional broadcasts have incompatible lengths.
    """
    n = len(target)
    if len(source) < n:
        raise ValueError("Carry XOR source must cover the target width")
    if control is False or n == 0:
        return ()
    if n > 1 and control is True:
        raise ValueError("Locked multi-bit carry XOR requires a quantum enable")
    quantum_inputs = tuple(
        dict.fromkeys(value for value in (*offset, carry_in) if type(value) is int)
    )
    _validate_registers(
        source,
        target,
        clean,
        quantum_inputs,
        (control,) if type(control) is int else (),
    )
    classical_inputs = {
        value.slot
        for value in (*offset, carry_in)
        if isinstance(value, ClassicalFieldCoefficient)
    }
    if measurement_bit is not None and (
        type(measurement_bit) is not int
        or measurement_bit < 0
        or measurement_bit in classical_inputs
    ):
        raise ValueError("Carry XOR requires a separate nonnegative measurement slot")
    lhs, rhs, dst = source[-n:][::-1], offset[::-1], target[::-1]
    low_bit = rhs[0] if rhs else False
    if n == 1:
        update = _flip(lhs[0], carry_in)
        return (
            *update,
            *_carry_parity_xor(
                dst[0], (low_bit, carry_in), control, lhs[0], clean, measurement_bit
            ),
            *update,
            *_flip(dst[0], control, carry_in),
        )
    assert type(control) is int
    quantum_offset = all(type(value) is int for value in rhs)
    toggle_source = _broadcast_offset(lhs, rhs, control, quantum_offset)
    toggle_target = _broadcast_offset(dst, rhs, control, quantum_offset)
    toggle_shifted = _broadcast_offset(dst, rhs[1:], control, quantum_offset)
    cascade = tuple(
        Instruction("ccx", (dst[k], lhs[k + 1], dst[k + 1])) for k in range(n - 1)
    )
    return (
        *toggle_source,
        *reversed(cascade),
        *toggle_target,
        *toggle_shifted,
        *_carry_parity_xor(
            dst[0], (carry_in, low_bit), control, lhs[0], clean, measurement_bit
        ),
        *cascade,
        *toggle_shifted,
        *toggle_source,
    )
