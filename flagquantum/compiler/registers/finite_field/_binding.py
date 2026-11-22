# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Kickmix gf_arithmetic.pybind.cc argument/reset helpers.
# Revision 9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453.

"""Shared owned-field ports and source init reset boundaries."""

from dataclasses import replace

from ....core._binary_field import BinaryField
from ..._component_trace.trace import ComponentCall
from ...artifacts.reversible_phase import (
    ReversiblePhaseOperation,
    ReversiblePhaseProgram,
)
from ..layout.operand_word import bind_quantum_word


def field_for_width(value: BinaryField | None, width: int) -> BinaryField:
    if value is None:
        return BinaryField(width)
    if not isinstance(value, BinaryField):
        raise ValueError("Field requires a native binary field value or None")
    if value.degree != width:
        raise ValueError("Field degree must match the register size")
    return value


def field_word(value: object, width: int) -> tuple[int, ...]:
    word = bind_quantum_word(value)
    if len(word) != width:
        raise ValueError("Register size must match the field degree")
    return word


def prefix_field_resets(
    result: tuple[ReversiblePhaseProgram, tuple[ComponentCall, ...]],
    *words: tuple[int, ...],
) -> tuple[ReversiblePhaseProgram, tuple[ComponentCall, ...]]:
    """Reset each big-endian word in source coefficient order before its call.

    Targets and retained scaffold already belong to the caller's Program.
    This emits the source resets without creating an implicit allocator.
    """
    program, calls = result
    prefix = tuple(
        ReversiblePhaseOperation("r", (wire,))
        for word in words
        for wire in reversed(word)
    )

    def shift(call: ComponentCall) -> ComponentCall:
        return replace(
            call,
            start=call.start + len(prefix),
            stop=call.stop + len(prefix),
            children=tuple(shift(child) for child in call.children),
        )

    return (
        replace(
            program,
            operations=(*prefix, *program.operations),
            n_qubits=max(
                program.n_qubits, max((q for w in words for q in w), default=0) + 1
            ),
        ),
        tuple(shift(call) for call in calls),
    )


def reserve_field_bits(
    result: tuple[ReversiblePhaseProgram, tuple[ComponentCall, ...]],
    width: int,
    *,
    external_bits: tuple[int, ...] = (),
) -> tuple[ReversiblePhaseProgram, tuple[ComponentCall, ...]]:
    """Place fresh feedback after caller storage while retaining external reads."""
    if type(width) is not int or width < 0:
        raise ValueError("Classical storage width must be a nonnegative integer")
    boundary = max(external_bits, default=-1) + 1
    reserved = max(width, boundary)
    offset = reserved - boundary
    program, calls = result

    def relocate(bit: int | None) -> int | None:
        return bit if bit is None or bit in external_bits else bit + offset

    return (
        replace(
            program,
            n_bits=max(reserved, program.n_bits + offset),
            operations=tuple(
                replace(
                    op,
                    result_bit=relocate(op.result_bit),
                    condition_bit=relocate(op.condition_bit),
                )
                for op in program.operations
            ),
            input_bits=tuple(
                sorted(
                    set(range(reserved)).union(
                        bit if bit in external_bits else bit + offset
                        for bit in program.input_bits
                    )
                )
            ),
        ),
        calls,
    )
