# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Kickmix gen_iadd_classical_simple.
# https://github.com/quantumlib/Qualtran/blob/9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453/kickmix/src/kickmix/gen/adders/gen_iadd_classical.cc

"""Mixed-offset addition with one clean carry wire per target bit."""

from ....core.ir import Instruction
from ..finite_field.coefficients import ClassicalFieldCoefficient
from ..finite_field.polynomial import _validate_registers
from ..primitives._mixed_gates import _flip
from ..primitives.measurement import measure_x_clear_ir
from .carry_parity import _carry_parity_gate, _Coefficient


def measured_carry_addition_ir(
    target: tuple[int, ...],
    offset: tuple[_Coefficient, ...],
    *,
    clean: tuple[int, ...],
    carry_in: _Coefficient = False,
    control: int | bool = True,
    measurement_bit: int | None = None,
) -> tuple[Instruction, ...]:
    """Add a big-endian mixed offset and carry in place.

    The caller supplies one zeroed carry wire per target bit and a reusable
    result slot when the locked schedule measures. Offset, carry input and
    work wires are restored. The one- and two-bit source paths require the
    corresponding explicit offset bits; wider targets zero-extend a nonempty
    short offset on its high end.
    """
    n = len(target)
    if len(clean) < n:
        raise ValueError("Simple addition requires one clean wire per target bit")
    if control is False or n == 0:
        return ()
    if len(offset) < (2 if n == 2 else 1):
        raise ValueError("Simple addition requires explicit low offset bits")
    quantum_inputs = tuple(
        dict.fromkeys(value for value in (*offset, carry_in) if type(value) is int)
    )
    _validate_registers(
        target, clean, quantum_inputs, (control,) if type(control) is int else ()
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
        raise ValueError(
            "Simple addition requires a separate nonnegative measurement slot"
        )

    word = target[::-1]
    rhs = (*reversed(offset), *((False,) * max(0, n - len(offset))))
    if n == 1:
        return (*_flip(word[0], carry_in, control), *_flip(word[0], rhs[0], control))
    if n == 2:
        invert = Instruction("x", (word[0],))
        return (
            *_flip(word[0], control, rhs[0]),
            *_flip(word[1], control, rhs[1]),
            invert,
            *_carry_parity_gate(
                word[1], (rhs[0], carry_in), (control, word[0]), clean, measurement_bit
            ),
            invert,
            *_flip(word[0], control, carry_in),
            *_flip(word[1], control, carry_in),
        )

    if measurement_bit is None:
        raise ValueError("Simple addition requires a reusable measurement slot")
    outcome = ClassicalFieldCoefficient(measurement_bit)
    gates = list(_flip(clean[0], control, carry_in))
    for k in range(n - 1):
        gates.extend(_flip(clean[k], control, rhs[k]))
        gates.extend(_flip(word[k], control, rhs[k]))
        gates.extend(_flip(clean[k + 1], control, rhs[k]))
        gates.extend(_flip(clean[k + 1], clean[k], word[k]))
    gates.extend(_flip(word[-1], clean[n - 1]))
    gates.extend(_flip(word[-1], control, rhs[n - 1]))

    for k in reversed(range(n - 1)):
        gates.extend(measure_x_clear_ir(clean[k + 1], measurement_bit))
        if k < n - 2:
            gates.extend(_flip(None, control, rhs[k + 1], outcome))
        gates.extend(_flip(None, clean[k], word[k], outcome))
        gates.extend(_flip(None, control, rhs[k], outcome))
        gates.extend(_flip(word[k], clean[k]))
        gates.extend(_flip(word[k], control, rhs[k]))
    gates.extend(measure_x_clear_ir(clean[0], measurement_bit))
    gates.extend(_flip(None, control, rhs[0], outcome))
    gates.extend(_flip(None, control, carry_in, outcome))
    return tuple(gates)
