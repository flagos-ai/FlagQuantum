# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Kickmix gf_arithmetic.pybind.cc scalar/word dispatch.
# Revision 9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453.

"""Bind field addition, constant scaling and classical product phases."""

from ....core._binary_field import BinaryField
from ..._component_trace.trace import ComponentCall
from ...artifacts.reversible_phase import ReversiblePhaseProgram
from ..layout.operand_word import (
    bind_operand_word,
    bind_operand_word_or_integer,
    bind_quantum_word,
    bind_z_word,
)
from ..primitives._mixed_gates import _Control
from ._binding import field_for_width, field_word, reserve_field_bits
from .binary_linear_map import physical_field_constant_scale
from .coefficients import ClassicalFieldCoefficient
from .physical_field_addition import physical_field_addition
from .product_erasure import physical_field_product_phase


def field_addition_binding(
    offset: object,
    target: object,
    *,
    field: BinaryField | None = None,
    control: _Control = True,
) -> tuple[ReversiblePhaseProgram, tuple[ComponentCall, ...]]:
    """Retain source integer/typed offset conversion and outer control scopes."""
    word = bind_quantum_word(target)
    if field is not None:
        field_for_width(field, len(word))
    coefficients = bind_z_word(bind_operand_word_or_integer(offset, len(word)))
    if len(coefficients) != len(word):
        raise ValueError("Field addition requires equal register sizes")
    return physical_field_addition(coefficients, word, control=control)


def field_constant_binding(
    constant: object,
    target: object,
    *,
    field: BinaryField | None = None,
    divide: bool = False,
) -> tuple[ReversiblePhaseProgram, tuple[ComponentCall, ...]]:
    """Require a nonzero classical field element before source scale emission."""
    if not isinstance(constant, int) or isinstance(constant, bool):
        raise NotImplementedError("Field scaling supports classical constant integers")
    word = bind_quantum_word(target)
    owned = field_for_width(field, len(word))
    constant = int(constant)
    if not owned.is_element(constant):
        raise ValueError("Field constant must be an element of the field")
    return physical_field_constant_scale(
        word, constant, modulus=owned.modulus, divide=divide
    )


def field_product_phase_binding(
    left: object,
    right: object,
    mask: object,
    *,
    field: BinaryField | None = None,
    classical_width: int = 0,
) -> tuple[ReversiblePhaseProgram, tuple[ComponentCall, ...]]:
    """Bind classical-only mask slots and the source field product phase."""
    masks = bind_operand_word(mask)
    if any(not isinstance(bit, ClassicalFieldCoefficient) for bit in masks):
        raise ValueError("Product phase mask requires classical bit slots")
    first = bind_quantum_word(left)
    owned = field_for_width(field, len(first))
    second = field_word(right, owned.degree)
    if len(masks) != owned.degree:
        raise ValueError("Mask size must match the field degree")
    bits = tuple(
        bit.slot for bit in masks if isinstance(bit, ClassicalFieldCoefficient)
    )
    return reserve_field_bits(
        physical_field_product_phase(first, second, bits, modulus=owned.modulus),
        classical_width,
        external_bits=bits,
    )
