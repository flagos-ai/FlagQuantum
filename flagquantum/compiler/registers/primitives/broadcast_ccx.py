# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Kickmix broadcast CCX and AND lifetime binding.
# https://github.com/quantumlib/Qualtran/blob/9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453/kickmix/src/kickmix/py/circuit/ops/ccx.pybind.cc

"""Bind the ordered control lanes of CCX and measured AND schedules."""

from collections.abc import Iterator

from ....core.ir import Instruction
from ..finite_field.coefficients import ClassicalFieldCoefficient
from ..layout.operand_word import bind_quantum_word, bind_z_word
from ._mixed_gates import _Control, _delete_and_ir, _flip
from .broadcast_binding import broadcast_operand_words
from .measurement import reset_x_register_ir


def _lanes(
    first: object, second: object, target: object
) -> Iterator[tuple[_Control, _Control, int]]:
    a, b, targets = broadcast_operand_words(first, second, target)
    return zip(bind_z_word(a), bind_z_word(b), bind_quantum_word(targets), strict=True)


def broadcast_ccx_ir(
    first: object, second: object, target: object
) -> tuple[Instruction, ...]:
    """Emit ordered CCX and the source's literal and classical reductions."""
    return tuple(
        gate for a, b, t in _lanes(first, second, target) for gate in _flip(t, a, b)
    )


def broadcast_init_and_ir(
    first: object, second: object, target: object
) -> tuple[Instruction, ...]:
    """Reset in the source X basis and compute each AND in lane order."""
    return tuple(
        gate
        for a, b, t in _lanes(first, second, target)
        for gate in (*reset_x_register_ir((t,)), *_flip(t, a, b))
    )


def broadcast_del_and_ir(
    first: object, second: object, target: object, *, measurement_bit: int
) -> tuple[Instruction, ...]:
    """Erase each AND and apply its feedback before reusing the result slot."""
    lanes = tuple(_lanes(first, second, target))
    if any(
        isinstance(control, ClassicalFieldCoefficient)
        and control.slot == measurement_bit
        for a, b, _ in lanes
        for control in (a, b)
    ):
        raise ValueError("AND erasure result cannot overwrite a broadcast control")
    return tuple(
        gate for a, b, t in lanes for gate in _delete_and_ir(t, measurement_bit, a, b)
    )
