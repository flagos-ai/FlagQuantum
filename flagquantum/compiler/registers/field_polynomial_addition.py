# Copyright 2025 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 GF2PolyAdd.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/gf_poly_arithmetic/gf2_poly_add.py

"""In-place addition of polynomial registers over binary extension fields."""

from __future__ import annotations

from ...core.ir import Instruction
from .._component_trace.symbolic_math import SymbolicInt
from .field_polynomial_layout import FieldPolynomialLayout
from .finite_field import gf2_add_ir
from .symbolic_resource_model import SymbolicResourceFormula


def gf2_polynomial_add_ir(
    source: tuple[int, ...],
    target: tuple[int, ...],
    layout: FieldPolynomialLayout,
) -> tuple[Instruction, ...]:
    """XOR corresponding coefficient words into target, highest power first.

    Both registers use the same degree bound and coefficient encoding.
    The caller provides leading zero coefficients when the actual polynomial
    degree is smaller than the bound. The source is preserved and there is no
    additional workspace. Applying the same operation twice is the identity.
    """
    source_words = layout.split(source)
    target_words = layout.split(target)
    if set(source) & set(target):
        raise ValueError("Polynomial addition requires disjoint operand registers")
    return tuple(
        gate
        for left, right in zip(source_words, target_words, strict=True)
        for gate in gf2_add_ir(left, right)
    )


def source_gf2_polynomial_add_resource_formula(
    degree: SymbolicInt, coefficient_width: SymbolicInt
) -> SymbolicResourceFormula:
    """Model the locked degree-plus-one coefficient additions without lowering."""
    if isinstance(degree, (int, float)) and (type(degree) is not int or degree < 0):
        raise ValueError("Polynomial addition requires a nonnegative degree")
    if isinstance(coefficient_width, (int, float)) and (
        type(coefficient_width) is not int or coefficient_width < 1
    ):
        raise ValueError("Polynomial addition requires a positive coefficient width")
    width = (degree + 1) * coefficient_width
    return SymbolicResourceFormula(
        and_bloq=0, clifford=width, measurement=0, logical_qubits=2 * width
    )
