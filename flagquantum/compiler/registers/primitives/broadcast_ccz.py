# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Kickmix broadcast CCZ binding.
# https://github.com/quantumlib/Qualtran/blob/9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453/kickmix/src/kickmix/py/circuit/ops/ccz.pybind.cc

"""Bind ordered CCZ lanes and their exact phase reductions."""

from ....core.ir import Instruction
from ..layout.operand_word import bind_z_word
from ._mixed_gates import _flip
from .broadcast_binding import broadcast_operand_words


def broadcast_ccz_ir(
    first: object, second: object, third: object
) -> tuple[Instruction, ...]:
    """Emit one three-input sign per lane, preserving the quantum order."""
    a, b, c = broadcast_operand_words(first, second, third)
    return tuple(
        gate
        for lane in zip(bind_z_word(a), bind_z_word(b), bind_z_word(c), strict=True)
        for gate in _flip(None, *lane)
    )
