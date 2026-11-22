# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from the locked Kickmix scalar and register broadcast resolver.
# https://github.com/quantumlib/Qualtran/blob/9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453/kickmix/src/kickmix/py/val/broadcast_resolver.pybind.h

# Unary dispatch also follows the locked py/circuit/ops/x.pybind.cc and z.pybind.cc.

"""Resolve scalar and explicit register widths before gate emission."""

from ....core.ir import Instruction
from ..layout.operand_word import bind_operand_word, bind_z_word, is_operand_sequence
from ._mixed_gates import (
    _broadcast_cx,
    _broadcast_cz,
    _flip,
    _invert_operand,
    _XBasisTarget,
)
from .operand_binding import _Operand


def broadcast_operand_words(
    *operands: object,
) -> tuple[tuple[_Operand, ...], ...]:
    """Align columns using the first explicit sequence width.

    A scalar repeats across that width. Explicit sequences retain their
    selected lane order and must all have the same width, including zero.
    """
    words = tuple(bind_operand_word(value) for value in operands)
    sequences = tuple(is_operand_sequence(value) for value in operands)
    widths = tuple(
        len(word) for word, sequence in zip(words, sequences, strict=True) if sequence
    )
    width = widths[0] if widths else 1
    if any(count != width for count in widths):
        raise ValueError("Inconsistent broadcast sizes for explicit operand sequences")
    return tuple(
        word if sequence else word * width
        for word, sequence in zip(words, sequences, strict=True)
    )


def broadcast_cx_ir(control: object, target: object) -> tuple[Instruction, ...]:
    """Bind CX ports and delegate the typed lanes to the existing emitter."""
    controls, targets = broadcast_operand_words(control, target)
    return _broadcast_cx(bind_z_word(controls), targets)


def broadcast_cz_ir(first: object, second: object) -> tuple[Instruction, ...]:
    """Bind CZ ports and delegate the typed lanes to the existing emitter."""
    left, right = broadcast_operand_words(first, second)
    return _broadcast_cz(bind_z_word(left), bind_z_word(right))


def broadcast_x_ir(target: object) -> tuple[Instruction, ...]:
    """Emit ordered X, classical inversion or X-eigenvalue phase lanes."""
    word = bind_operand_word(target)
    if any(type(operand) is bool for operand in word):
        raise ValueError("X targets cannot be ordinary Boolean literals")
    return tuple(
        gate
        for operand in word
        for gate in (
            _flip(operand)
            if isinstance(operand, _XBasisTarget)
            else _invert_operand(operand)[1]
        )
    )

