# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Kickmix gen_iadd_classical.cc and CircuitBuilder.
# https://github.com/quantumlib/Qualtran/blob/9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453/kickmix/src/kickmix/gen/adders/gen_iadd_classical.cc

"""Mixed-offset arithmetic with the locked measured carry cleanup."""

from dataclasses import dataclass, field, replace
from math import ceil, inf, isnan

from ....core.classical_dataflow import ClassicalXor
from ....core.ir import Instruction
from ..._component_trace.trace import ComponentCall
from ..finite_field.coefficients import ClassicalFieldCoefficient
from ..finite_field.polynomial import _validate_registers
from ..layout.workspace import GenerationWorkspace
from ..primitives._mixed_gates import _delete_and_ir, _flip, _invert_operand
from ..primitives.measurement import measure_x_clear_ir, reset_x_register_ir
from .carry_parity import (
    _broadcast_offset,
    _carry_parity_gate,
    _Coefficient,
    carry_parity_ir,
)
from .vented_addition import vented_addition_ir


@dataclass
class AdditionExport:
    """Own source component ranges and feedback scopes in instruction offsets."""

    calls: list[ComponentCall] = field(default_factory=list)
    feedback: list[tuple[int, int, int]] = field(default_factory=list)
    implicit_zero_bits: set[int] = field(default_factory=set)


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
    emit_resets: bool,
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
            *(reset_x_register_ir((a,)) if emit_resets else ()),
            *_flip(a, t0, control),
            *_xor_parity(t1, carry, o0, a),
            *_delete_and_ir(a, slot, t0, control),
            *_xor_parity(t1, o0, o1, control),
            *_xor_parity(t0, o0, carry, control),
            *toggle,
        )
    t2, o2 = target[2], offset[2]
    return (
        *(reset_x_register_ir((a, b)) if emit_resets else ()),
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
    workspace: GenerationWorkspace,
    bits: tuple[int, ...],
    btol: float,
    export: AdditionExport | None,
    start: int,
) -> tuple[Instruction, ...]:
    if not bits:
        raise ValueError("Mixed addition requires a reusable measurement slot")
    slot = bits[0]
    outcome = ClassicalFieldCoefficient(slot)
    clean = workspace.clean
    m = min(len(target) - 1, len(clean) - 2)
    low = offset[:m]
    quantum = all(type(value) is int for value in offset)
    gates = [
        *(reset_x_register_ir(clean[:m]) if export is not None else ()),
        *_broadcast_offset(target[:m], low, control, quantum),
        *_broadcast_offset(clean[:m], low, control, quantum),
        *_carry_parity_gate(
            clean[0],
            (carry, offset[0]),
            (control, target[0]),
            clean[1:],
            slot,
            emit_resets=export is not None,
        ),
    ]
    for k in range(1, m):
        gates.extend(_flip(clean[k - 1], control, offset[k]))
        gates.extend(_flip(clean[k], target[k], clean[k - 1]))
    leased, remainder = workspace.take_clean(m)
    gates.extend(
        _add(
            target[m:],
            offset[m:],
            remainder.with_borrowed(leased[:-1]),
            leased[-1],
            control,
            bits,
            btol,
            export=export,
            start=start + len(gates),
        )
    )
    for k in reversed(range(1, m)):
        gates.extend(measure_x_clear_ir(clean[k], slot))
        scope = start + len(gates)
        gates.extend(_flip(None, target[k], clean[k - 1], outcome))
        gates.extend(_flip(None, offset[k], control, outcome))
        if export is not None:
            export.feedback.append((scope, start + len(gates), slot))
        gates.extend(_flip(clean[k - 1], offset[k], control))
        gates.extend(_flip(target[k], clean[k - 1]))
    gates.extend(measure_x_clear_ir(clean[0], slot))
    scope = start + len(gates)
    gates.extend(_guard(_xor_parity(None, carry, offset[0], control, target[0]), slot))
    gates.extend(_flip(None, control, offset[0], outcome))
    if export is not None:
        export.feedback.append((scope, start + len(gates), slot))
    gates.extend(_flip(target[0], control, carry))
    return tuple(gates)


def _add(
    target: tuple[int, ...],
    offset: tuple[_Coefficient, ...],
    workspace: GenerationWorkspace,
    carry: _Coefficient,
    control: int | bool,
    bits: tuple[int, ...],
    btol: float,
    *,
    export: AdditionExport | None = None,
    start: int = 0,
) -> tuple[Instruction, ...]:
    children = (
        AdditionExport(
            feedback=export.feedback, implicit_zero_bits=export.implicit_zero_bits
        )
        if export is not None
        else None
    )
    gates = _add_body(
        target, offset, workspace, carry, control, bits, btol, children, start
    )
    if export is not None and children is not None:
        name = "c_iadd_classical" if type(control) is int else "iadd_classical"
        export.calls.append(
            ComponentCall(
                name, start, start + len(gates), children=tuple(children.calls)
            )
        )
    return gates


def _add_body(
    target: tuple[int, ...],
    offset: tuple[_Coefficient, ...],
    workspace: GenerationWorkspace,
    carry: _Coefficient,
    control: int | bool,
    bits: tuple[int, ...],
    btol: float,
    export: AdditionExport | None,
    start: int,
) -> tuple[Instruction, ...]:
    n = len(target)
    if n == 0:
        return ()
    offset = (*offset, *((False,) * max(0, n - len(offset))))
    clean = workspace.clean
    if n > 1 and len(clean) < 2:
        raise ValueError("Mixed addition requires two clean wires")
    if n <= 3:
        return _short_addition(
            target, offset, carry, control, clean, bits, export is not None
        )
    if len(clean) + workspace.borrowed_count < n:
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
                    workspace.with_borrowed(target[width:]),
                    carry,
                    control,
                    bits,
                    inf,
                    export=export,
                    start=start,
                )
            complemented = tuple(_invert_operand(value) for value in low)
            inverted_carry, carry_gates = _invert_operand(carry)
            input_gates = tuple(gate for _, gates in complemented for gate in gates)
            invert = tuple(Instruction("x", (wire,)) for wire in target[:width])
            sub = _add(
                target[:width],
                tuple(value for value, _ in complemented),
                workspace.with_borrowed(target[width:]),
                inverted_carry,
                control,
                bits,
                inf,
                export=export,
                start=start + len(input_gates) + len(invert) + len(carry_gates),
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
    if len(workspace.preferred_clean) > 2:
        return _clean_prefix(
            target, offset, carry, control, workspace, bits, btol, export, start
        )
    if workspace.borrowed_count < n - 2:
        raise ValueError("Mixed addition requires n - 2 dirty wires for vented carries")
    if control is True:
        raise ValueError("Locked vented carry cleanup requires a quantum enable")
    if len(bits) < n:
        raise ValueError("Mixed addition requires n classical slots for vented carries")
    dirty = workspace.borrowed_prefix(n - 2)
    vents, slot = bits[1 : n - 1], bits[n - 1]
    # Native clean slots start at zero. Only the used vent slots need a store.
    initialize = (
        tuple(ClassicalXor((), bit).instruction() for bit in vents[1:])
        if export is None
        else ()
    )
    feedback: list[tuple[int, int, int]] | None = [] if export is not None else None
    body = vented_addition_ir(
        target[::-1],
        offset[::-1],
        clean=clean,
        carry_copy=(*dirty[::-1], None),
        vent_bits=vents,
        carry_in=carry,
        control=control,
        measurement_bit=slot,
        emit_resets=export is not None,
        feedback=feedback,
    )
    invert = tuple(Instruction("x", (wire,)) for wire in target)
    correction = tuple(
        gate
        for wire, bit in zip(dirty, vents[1:], strict=False)
        for gate in _flip(None, wire, ClassicalFieldCoefficient(bit))
    )
    erase = carry_parity_ir(
        target[:-1][::-1],
        offset[::-1],
        dirty[::-1],
        carry_in=carry,
        control=control,
        clean=clean,
        measurement_bit=slot,
        emit_resets=export is not None,
    )
    if export is not None and feedback is not None:
        export.implicit_zero_bits.update(vents[1:])
        export.feedback.extend((start + a, start + b, bit) for a, b, bit in feedback)
        stop_body = start + len(body)
        begin_erase = stop_body + len(invert) + len(correction)
        export.calls.extend(
            (
                ComponentCall("c_iadd_classical_vented", start, stop_body),
                ComponentCall("c_ixor_carries", begin_erase, begin_erase + len(erase)),
            )
        )
    return (*initialize, *body, *invert, *correction, *erase, *correction, *invert)


def mixed_addition_ir(
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
    export: AdditionExport | None = None,
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
        GenerationWorkspace(clean, minimize_qubits).with_borrowed(dirty),
        carry_in,
        control,
        classical_bits,
        btol,
        export=export,
        start=len(word) if subtract else 0,
    )
    invert = tuple(Instruction("x", (wire,)) for wire in word) if subtract else ()
    return (*invert, *gates, *invert)
