# Copyright 2025 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Assertions adapted from Qualtran v0.7.0 polynomial-addition tests.

"""Preserve original polynomial callback domains and constructor validation."""

from __future__ import annotations

import pytest
import sympy

from flagquantum.simulation.finite_field_addition import binary_field_encoding
from flagquantum.simulation.finite_field_polynomial_addition import (
    polynomial_add_classical_values,
    polynomial_add_constant_classical_value,
    validate_polynomial_constant,
)
from flagquantum.simulation.finite_field_polynomial_encoding import (
    FiniteFieldPolynomialEncoding,
)

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def test_all_original_polynomial_classical_assertions() -> None:
    import galois
    from qualtran.bloqs.gf_poly_arithmetic.gf2_poly_add import _gf2_poly_4_8_add
    from qualtran.bloqs.gf_poly_arithmetic.gf2_poly_add_k import _gf2_poly_4_8_add_k

    add = _gf2_poly_4_8_add.make()
    constant_add = _gf2_poly_4_8_add_k.make()
    encoding = FiniteFieldPolynomialEncoding(4, binary_field_encoding(3))
    field = encoding.field.gf_type
    constant = galois.Poly(field([1, 2, 3, 4, 5]))
    validate_polynomial_constant(encoding, constant)
    domain = [
        galois.Poly(field(words))
        for words in [
            [0, 0, 0, 0, 0],
            [7, 7, 7, 7, 7],
            [0, 0, 3, 5, 7],
            [2, 3, 5, 0, 0],
            [0, 1, 2, 3, 4],
            [1, 2, 3, 4, 5],
        ]
    ]
    for source in domain:
        assert (
            polynomial_add_constant_classical_value(source, constant)
            == constant_add.call_classically(f_x=source)[0]
        )
        for target in domain:
            left, right = polynomial_add_classical_values(source, target)
            assert left is source
            assert (left, right) == add.call_classically(f_x=source, g_x=target)
    # Original callbacks add values without introducing additional field guards.
    assert polynomial_add_classical_values(1, 2) == (1, 3)
    assert polynomial_add_constant_classical_value(1, 2) == 3


@pytest.mark.parametrize(
    "degree,width",
    [
        (4, 3),
        (4, sympy.Symbol("m")),
        (sympy.Symbol("n"), 3),
        (sympy.Symbol("n"), sympy.Symbol("m")),
    ],
)
def test_original_constant_admission_and_validation_order(
    degree: int | sympy.Expr, width: int | sympy.Expr
) -> None:
    import galois
    from qualtran import QGF, QGFPoly
    from qualtran.bloqs.gf_poly_arithmetic.gf2_poly_add_k import GF2PolyAddK

    encoding = FiniteFieldPolynomialEncoding(degree, binary_field_encoding(width))
    original = QGFPoly(degree, QGF(2, width))
    for field in [
        galois.GF(2),
        binary_field_encoding(2).gf_type,
        binary_field_encoding(3).gf_type,
    ]:
        for coefficients in [[0], [1, 0], [1, 1, 0, 0, 0], [1, 0, 0, 0, 0, 0]]:
            constant = galois.Poly(field(coefficients))
            try:
                GF2PolyAddK(original, constant)
            except ValueError as error:
                with pytest.raises(ValueError) as actual:
                    validate_polynomial_constant(encoding, constant)
                assert str(actual.value) == str(error)
            else:
                validate_polynomial_constant(encoding, constant)
