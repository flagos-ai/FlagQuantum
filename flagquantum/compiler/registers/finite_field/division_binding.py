# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Kickmix gf_arithmetic.pybind.cc division wrappers.
# Revision 9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453.

"""Bind quotient init, accumulation and retained-scaffold erasure."""

from typing import Literal

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
from .division_chain import physical_field_division


def field_division_binding(
    dividend: object,
    divisor: object,
    target: object,
    *,
    field: BinaryField | None = None,
    workspace: tuple[int, ...],
    retain_chain: bool = False,
    mode: Literal["init", "xor", "erase"] = "xor",
    classical_width: int = 0,
) -> tuple[ReversiblePhaseProgram, tuple[ComponentCall, ...]]:
    """Preserve source reset order and caller-retained inverse/scaffold words."""
    if mode not in ("init", "xor", "erase"):
        raise ValueError("Unknown field division binding mode")
    if retain_chain and mode == "xor":
        raise ValueError("Retained quotient scaffold requires init or erase")
    first = bind_quantum_word(dividend)
    owned = field_for_width(field, len(first))
    second = field_word(divisor, owned.degree)
    output = field_word(target, owned.degree)
    result = physical_field_division(
        first,
        second,
        output,
        workspace,
        modulus=owned.modulus,
        retain_chain=retain_chain,
        erase=mode == "erase",
    )
    result = reserve_field_bits(result, classical_width)
    return prefix_field_resets(result, output) if mode == "init" else result
