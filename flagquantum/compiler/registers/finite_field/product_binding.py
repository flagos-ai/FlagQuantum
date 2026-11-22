# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Kickmix gf_arithmetic.pybind.cc product wrappers.
# Revision 9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453.

"""Bind init, XOR and measured product deletion with source control dispatch."""

from typing import Literal

from ....core._binary_field import BinaryField
from ..._component_trace.trace import ComponentCall
from ...artifacts.reversible_phase import ReversiblePhaseProgram
from ...lowering.physical_trace import scope_physical_trace
from ..layout.operand_word import bind_quantum_word
from ..primitives._mixed_gates import _Control
from ..primitives.control_binding import bind_z_enable
from ._binding import (
    field_for_width,
    field_word,
    prefix_field_resets,
    reserve_field_bits,
)
from .polynomial_product import physical_field_product


def field_product_binding(
    left: object,
    right: object,
    target: object,
    *,
    field: BinaryField | None = None,
    mode: Literal["init", "xor", "erase"] = "xor",
    control: _Control = True,
    workspace: tuple[int, ...] = (),
    classical_width: int = 0,
) -> tuple[ReversiblePhaseProgram, tuple[ComponentCall, ...]]:
    """Compose source wrappers with caller-owned output and clean work words.

    Words are MSB-first, including the leased product scratch. Init resets its
    target outside the enable scope. Quantum-controlled erase uses the same
    reversible controlled product as accumulation, retaining its source call.
    """
    if type(classical_width) is not int or classical_width < 0:
        raise ValueError("Classical storage width must be a nonnegative integer")
    if mode not in ("init", "xor", "erase"):
        raise ValueError("Unknown field product binding mode")
    first = bind_quantum_word(left)
    owned = field_for_width(field, len(first))
    second = field_word(right, owned.degree)
    output = field_word(target, owned.degree)
    enable, guards = bind_z_enable(control)
    if enable is False:
        result: tuple[ReversiblePhaseProgram, tuple[ComponentCall, ...]] = (
            ReversiblePhaseProgram(max((*first, *second, *output)) + 1, 0, ()),
            (),
        )
    else:
        program, calls = physical_field_product(
            first,
            second,
            output,
            modulus=owned.modulus,
            erase=mode == "erase",
            control=enable if type(enable) is int else None,
            workspace=workspace if len(workspace) >= owned.degree else (),
        )
        result = program, calls
    program, calls = reserve_field_bits(
        result, max(classical_width, max(guards, default=-1) + 1)
    )
    result = scope_physical_trace(program, calls, guards)
    return prefix_field_resets(result, output) if mode == "init" else result
