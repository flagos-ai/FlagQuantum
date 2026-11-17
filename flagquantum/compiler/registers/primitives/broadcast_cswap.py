# SPDX-License-Identifier: Apache-2.0
# Adapted from the locked Kickmix controlled-swap binding.
# https://github.com/quantumlib/Qualtran/blob/9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453/kickmix/src/kickmix/py/circuit/ops/cswap.pybind.cc

"""Bind ordered scalar and array controls for physical swap lanes."""

from ....core.ir import Instruction
from ..layout.operand_word import bind_quantum_word, bind_z_word
from .broadcast_binding import broadcast_operand_words
from .swap import _broadcast_controlled_swap_ir


def broadcast_cswap_ir(
    control: object, left: object, right: object
) -> tuple[Instruction, ...]:
    """Apply controlled swap using typed controls and the shared width resolver."""
    controls, first, second = broadcast_operand_words(control, left, right)
    return tuple(
        gate
        for enable, a, b in zip(
            bind_z_word(controls),
            bind_quantum_word(first),
            bind_quantum_word(second),
            strict=True,
        )
        for gate in _broadcast_controlled_swap_ir(enable, (a,), (b,))
    )
