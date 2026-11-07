# Copyright 2025 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 GF2PolyAdd and GF2PolyAddK.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/gf_poly_arithmetic/gf2_poly_add.py
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/gf_poly_arithmetic/gf2_poly_add_k.py

"""In-place addition of polynomial registers over binary extension fields."""

from __future__ import annotations

from ...core.ir import Instruction
from .._component_trace.symbolic_math import SymbolicInt
from .field_polynomial_layout import FieldPolynomialLayout
from .finite_field import gf2_add_constant_ir, gf2_add_ir
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


def _polynomial_width(
    degree: SymbolicInt, coefficient_width: SymbolicInt
) -> SymbolicInt:
    if isinstance(degree, (int, float)) and (type(degree) is not int or degree < 0):
        raise ValueError("Polynomial addition requires a nonnegative degree")
    if isinstance(coefficient_width, (int, float)) and (
        type(coefficient_width) is not int or coefficient_width < 1
    ):
        raise ValueError("Polynomial addition requires a positive coefficient width")
    return (degree + 1) * coefficient_width


def source_gf2_polynomial_add_resource_formula(
    degree: SymbolicInt, coefficient_width: SymbolicInt
) -> SymbolicResourceFormula:
    """Model the locked degree-plus-one coefficient additions without lowering."""
    width = _polynomial_width(degree, coefficient_width)
    return SymbolicResourceFormula(
        and_bloq=0, clifford=width, measurement=0, logical_qubits=2 * width
    )


def _constant_coefficients(
    coefficients: tuple[int, ...], layout: FieldPolynomialLayout
) -> tuple[int, ...]:
    count = layout.degree + 1
    if not coefficients or any(coefficients[:-count]):
        raise ValueError("Constant polynomial must fit the declared degree bound")
    if any(
        type(value) is not int or not 0 <= value < 1 << layout.coefficient_width
        for value in coefficients
    ):
        raise ValueError("Constant coefficient must fit the declared field width")
    coefficients = coefficients[-count:]
    return (0,) * (count - len(coefficients)) + coefficients


def gf2_polynomial_add_constant_ir(
    target: tuple[int, ...],
    coefficients: tuple[int, ...],
    layout: FieldPolynomialLayout,
) -> tuple[Instruction, ...]:
    """Add an encoded constant in descending degree order, padding on the left."""
    words = layout.split(target)
    padded = _constant_coefficients(coefficients, layout)
    return tuple(
        gate
        for word, value in zip(words, padded, strict=True)
        for gate in gf2_add_constant_ir(word, value)
    )


def source_gf2_polynomial_add_constant_resource_formula(
    degree: SymbolicInt,
    coefficient_width: SymbolicInt,
    *,
    coefficients: tuple[int, ...] | None = None,
) -> SymbolicResourceFormula:
    """Count exact X gates for a concrete constant or use the locked shape bound.

    Omitting coefficients applies the source's symbolic unknown-constant bound,
    with one potential X per register bit.
    """
    width = _polynomial_width(degree, coefficient_width)
    flips = width
    if coefficients is not None:
        if type(degree) is not int or type(coefficient_width) is not int:
            raise ValueError("Exact constant costs require concrete dimensions")
        layout = FieldPolynomialLayout(degree, coefficient_width)
        flips = sum(
            value.bit_count() for value in _constant_coefficients(coefficients, layout)
        )
    return SymbolicResourceFormula(
        and_bloq=0, clifford=flips, measurement=0, logical_qubits=width
    )
