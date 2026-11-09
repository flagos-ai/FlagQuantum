# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Kickmix gen_iadd_classical.cc and CircuitBuilder.
# https://github.com/quantumlib/Qualtran/blob/9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453/kickmix/src/kickmix/gen/adders/gen_iadd_classical.cc

"""Mixed-offset arithmetic with the locked measured carry cleanup."""

from dataclasses import replace
from math import ceil, inf, isnan

from ....core.classical_dataflow import ClassicalXor
from ....core.ir import Instruction
from ..finite_field.coefficients import ClassicalFieldCoefficient
from ..finite_field.polynomial import _validate_registers
from ..primitives._mixed_gates import _flip
from ..primitives.measurement import measure_x_clear_ir
from .carry_parity import (
    _broadcast_offset,
    _carry_parity_gate,
    _Coefficient,
    kickmix_carry_xor_ir,
)
from .vented_addition import kickmix_vented_addition_ir


def _xor_parity(
    target: int | None,
    first: _Coefficient,
    second: _Coefficient,
    control: _Coefficient,
    outcome: _Coefficient | None = None,
) -> tuple[Instruction, ...]:
    if type(first) is type(second) and first == second:
        return ()
    extra = (outcome,) if outcome is not None else ()
    if (type(first) is type(control) and first == control) or (
        type(second) is type(control) and second == control
    ):
        return (*_flip(target, first, second, *extra), *_flip(target, control, *extra))
    return _carry_parity_gate(target, (first, second), (control, *extra), (), None)


def _complement(value: _Coefficient) -> tuple[_Coefficient, tuple[Instruction, ...]]:
    if isinstance(value, bool):
        return not value, ()
    if isinstance(value, ClassicalFieldCoefficient):
        slot = value.slot
        return value, (ClassicalXor((slot,), slot, constant=True).instruction(),)
    return value, _flip(value)


def _guard(gates: tuple[Instruction, ...], bit: int) -> tuple[Instruction, ...]:
    # The H pair around a guarded CCX cancels when the guard is false.
    return tuple(
        (
            replace(
                gate,
                metadata={
                    **gate.metadata,
                    "conditions": tuple(
                        sorted({*gate.metadata.get("conditions", ()), (bit, 1)})
                    ),
                },
            )
            if gate.name != "h"
            else gate
        )
        for gate in gates
    )


def _short_addition(
    target: tuple[int, ...],
    offset: tuple[_Coefficient, ...],
    carry: _Coefficient,
    control: int | bool,
    clean: tuple[int, ...],
    bits: tuple[int, ...],
) -> tuple[Instruction, ...]:
    if len(target) == 1:
        return _xor_parity(target[0], carry, offset[0], control)
    if not bits:
        raise ValueError("Mixed addition requires a reusable measurement slot")
    slot = bits[0]
    outcome = ClassicalFieldCoefficient(slot)
    a, b = clean[:2]
    t0, t1 = target[:2]
    o0, o1 = offset[:2]
    toggle = (*_flip(t0, o0), *_flip(t1, o1))
    if len(target) == 2:
        return (
            *toggle,
            *_flip(a, t0, control),
            *_xor_parity(t1, carry, o0, a),
            *measure_x_clear_ir(a, slot),
            *_flip(None, t0, control, outcome),
            *_xor_parity(t1, o0, o1, control),
            *_xor_parity(t0, o0, carry, control),
            *toggle,
        )
    t2, o2 = target[2], offset[2]
    return (
        *toggle,
        *_xor_parity(a, carry, o0, t0),
        *_flip(a, o0),
        *_flip(a, o1),
        *_flip(b, control, a),
        *_flip(t2, t1, b),
        *_flip(t1, b),
        *measure_x_clear_ir(b, slot),
        *_flip(None, control, a, outcome),
        *measure_x_clear_ir(a, slot),
        *_xor_parity(None, carry, o0, t0, outcome),
        *_flip(None, o0, outcome),
        *_flip(None, o1, outcome),
        *toggle,
        *_xor_parity(t2, o1, o2, control),
        *_xor_parity(t0, o0, carry, control),
    )


def _clean_prefix(
    target: tuple[int, ...],
    offset: tuple[_Coefficient, ...],
    carry: _Coefficient,
    control: int | bool,
    clean: tuple[int, ...],
    dirty: tuple[int, ...],
    bits: tuple[int, ...],
    btol: float,
) -> tuple[Instruction, ...]:
    if not bits:
        raise ValueError("Mixed addition requires a reusable measurement slot")
    slot = bits[0]
    outcome = ClassicalFieldCoefficient(slot)
    m = min(len(target) - 1, len(clean) - 2)
    low = offset[:m]
    quantum = all(type(value) is int for value in offset)
    gates = [
        *_broadcast_offset(target[:m], low, control, quantum),
        *_broadcast_offset(clean[:m], low, control, quantum),
        *_carry_parity_gate(
            clean[0], (carry, offset[0]), (control, target[0]), clean[1:], slot
        ),
    ]
    for k in range(1, m):
        gates.extend(_flip(clean[k - 1], control, offset[k]))
        gates.extend(_flip(clean[k], target[k], clean[k - 1]))
    gates.extend(
        _add(
            target[m:],
            offset[m:],
            clean[m:],
            (*clean[: m - 1], *dirty),
            clean[m - 1],
            control,
            bits,
            False,
            btol,
        )
    )
    for k in reversed(range(1, m)):
        gates.extend(measure_x_clear_ir(clean[k], slot))
        gates.extend(_flip(None, target[k], clean[k - 1], outcome))
        gates.extend(_flip(None, offset[k], control, outcome))
        gates.extend(_flip(clean[k - 1], offset[k], control))
        gates.extend(_flip(target[k], clean[k - 1]))
    gates.extend(measure_x_clear_ir(clean[0], slot))
    gates.extend(_guard(_xor_parity(None, carry, offset[0], control, target[0]), slot))
    gates.extend(_flip(None, control, offset[0], outcome))
    gates.extend(_flip(target[0], control, carry))
    return tuple(gates)


def _add(
    target: tuple[int, ...],
    offset: tuple[_Coefficient, ...],
    clean: tuple[int, ...],
    dirty: tuple[int, ...],
    carry: _Coefficient,
    control: int | bool,
    bits: tuple[int, ...],
    minimize: bool,
    btol: float,
) -> tuple[Instruction, ...]:
    n = len(target)
    if n == 0:
        return ()
    offset = (*offset, *((False,) * max(0, n - len(offset))))
    if n > 1 and len(clean) < 2:
        raise ValueError("Mixed addition requires two clean wires")
    if n <= 3:
        return _short_addition(target, offset, carry, control, clean, bits)
    if len(clean) + len(dirty) < n:
        raise ValueError("Mixed addition requires at least n work wires")
    if btol < n and isinstance(offset[n - 1], bool):
        inverted = offset[n - 1]
        width = n
        while width and offset[width - 1] is inverted:
            width -= 1
        width = min(n, width + ceil(max(btol, 0)))
        if width < n:
            low = offset[:width]
            if not inverted:
                return _add(
                    target[:width],
                    low,
                    clean,
                    (*target[width:], *dirty),
                    carry,
                    control,
                    bits,
                    minimize,
                    inf,
                )
            complemented = tuple(_complement(value) for value in low)
            inverted_carry, carry_gates = _complement(carry)
            input_gates = tuple(gate for _, gates in complemented for gate in gates)
            invert = tuple(Instruction("x", (wire,)) for wire in target[:width])
            sub = _add(
                target[:width],
                tuple(value for value, _ in complemented),
                clean,
                (*target[width:], *dirty),
                inverted_carry,
                control,
                bits,
                minimize,
                inf,
            )
            return (
                *input_gates,
                *invert,
                *carry_gates,
                *sub,
                *carry_gates,
                *invert,
                *input_gates,
            )
    if not minimize and len(clean) > 2:
        return _clean_prefix(target, offset, carry, control, clean, dirty, bits, btol)
    if len(dirty) < n - 2:
        raise ValueError("Mixed addition requires n - 2 dirty wires for vented carries")
    if control is True:
        raise ValueError("Locked vented carry cleanup requires a quantum enable")
    if len(bits) < n:
        raise ValueError("Mixed addition requires n classical slots for vented carries")
    dirty = dirty[: n - 2]
    vents, slot = bits[1 : n - 1], bits[n - 1]
    # Native clean slots start at zero. Only the used vent slots need a store.
    initialize = tuple(ClassicalXor((), bit).instruction() for bit in vents[1:])
    body = kickmix_vented_addition_ir(
        target[::-1],
        offset[::-1],
        clean=clean,
        carry_copy=(*dirty[::-1], None),
        vent_bits=vents,
        carry_in=carry,
        control=control,
        measurement_bit=slot,
    )
    invert = tuple(Instruction("x", (wire,)) for wire in target)
    correction = tuple(
        gate
        for wire, bit in zip(dirty, vents[1:], strict=False)
        for gate in _flip(None, wire, ClassicalFieldCoefficient(bit))
    )
    erase = kickmix_carry_xor_ir(
        target[:-1][::-1],
        offset[::-1],
        dirty[::-1],
        carry_in=carry,
        control=control,
        clean=clean,
        measurement_bit=slot,
    )
    return (*initialize, *body, *invert, *correction, *erase, *correction, *invert)


def kickmix_mixed_addition_ir(
    target: tuple[int, ...],
    offset: tuple[_Coefficient, ...],
    *,
    carry_in: _Coefficient = False,
    control: int | bool = True,
    clean: tuple[int, ...] = (),
    dirty: tuple[int, ...] = (),
    classical_bits: tuple[int, ...] = (),
    minimize_qubits: bool = False,
    btol: float = inf,
    subtract: bool = False,
) -> tuple[Instruction, ...]:
    """Update a big-endian quantum word with a mixed offset and carry or borrow.

    Clean wires start and end at zero. Dirty wires, the offset, and carry
    are restored. The caller supplies result slots in source allocation
    order. Vented carry slots are initialized before use.

    This source schedule reproduces the locked extra-clean prefix branch,
    which has known value and residual-phase defects for some mixed offsets.
    Select the vented schedule with sufficient dirty work wires for exact
    arithmetic in those configurations.
    """
    quantum_inputs = tuple(
        dict.fromkeys(value for value in (*offset, carry_in) if type(value) is int)
    )
    _validate_registers(
        target, quantum_inputs, clean, dirty, (control,) if type(control) is int else ()
    )
    inputs = {
        value.slot
        for value in (*offset, carry_in)
        if isinstance(value, ClassicalFieldCoefficient)
    }
    if len(set(classical_bits)) != len(classical_bits) or any(
        type(bit) is not int or bit < 0 or bit in inputs for bit in classical_bits
    ):
        raise ValueError("Mixed addition requires distinct nonnegative classical slots")
    if isnan(btol):
        raise ValueError("Mixed addition requires an ordered bit tolerance")
    if control is False:
        return ()
    word = target[::-1]
    gates = _add(
        word,
        offset[::-1],
        clean,
        dirty,
        carry_in,
        control,
        classical_bits,
        minimize_qubits,
        btol,
    )
    invert = tuple(Instruction("x", (wire,)) for wire in word) if subtract else ()
    return (*invert, *gates, *invert)
