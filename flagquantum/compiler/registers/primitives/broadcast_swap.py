# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Kickmix swap binding.
# https://github.com/quantumlib/Qualtran/blob/9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453/kickmix/src/kickmix/py/circuit/ops/swap.pybind.cc

"""Bind scalar and array operands to the existing physical swap dispatcher."""

from ....core.ir import Instruction
from .broadcast_binding import broadcast_operand_words
from .swap import _broadcast_swap_ir


def broadcast_swap_ir(left: object, right: object) -> tuple[Instruction, ...]:
    """Swap each resolved pair of matching writable operands in order."""
    first, second = broadcast_operand_words(left, right)
    return _broadcast_swap_ir(first, second)
