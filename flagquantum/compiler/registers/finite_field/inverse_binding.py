# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Kickmix gf_arithmetic.pybind.cc inverse wrappers.
# Revision 9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453.

"""Bind ordinary and retained-chain field inverse source wrappers."""

from ....core._binary_field import BinaryField
from ..._component_trace.trace import ComponentCall
from ...artifacts.reversible_phase import ReversiblePhaseProgram
from ..layout.operand_word import bind_quantum_word
from ._binding import (
    field_for_width,
    field_word,
    prefix_field_resets,
    reserve_field_bits,
)
from .inverse_chain import field_inverse_chain_size, physical_field_inverse


def field_inverse_binding(
    source: object,
    target: object,
    *,
    field: BinaryField | None = None,
    workspace: tuple[int, ...] = (),
    retain_chain: bool = False,
    erase: bool = False,
    classical_width: int = 0,
) -> tuple[ReversiblePhaseProgram, tuple[ComponentCall, ...]]:
    """Initialize or erase an inverse with explicit caller-owned target/chain."""
    word = bind_quantum_word(source)
    owned = field_for_width(field, len(word))
    output = field_word(target, owned.degree)
    if retain_chain and len(workspace) != field_inverse_chain_size(owned.degree):
        raise ValueError("Retained inverse scaffold size must match the source chain")
    result = physical_field_inverse(
        word,
        output,
        workspace,
        modulus=owned.modulus,
        retain_chain=retain_chain,
        erase=erase,
    )
    result = reserve_field_bits(result, classical_width)
    if erase:
        return result
    scaffold = (
        tuple(
            workspace[start : start + owned.degree]
            for start in range(0, len(workspace), owned.degree)
        )
        if retain_chain
        else ()
    )
    return prefix_field_resets(result, output, *scaffold)
