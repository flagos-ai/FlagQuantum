"""Unmodified source product and square value types, arithmetic and failures."""

from itertools import product

import numpy as np
import pytest
import sympy

from flagquantum.simulation.classical_product_accumulation import (
    accumulate_product_classical_values,
)

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def same_value(actual, expected):
    assert type(actual) is type(expected)
    if isinstance(expected, np.ndarray):
        assert actual.dtype == expected.dtype
        np.testing.assert_array_equal(actual, expected)
    else:
        assert actual == expected


@pytest.mark.parametrize(
    "width", [2, 8, sympy.Integer(8), sympy.Symbol("n", integer=True, positive=True)]
)
@pytest.mark.parametrize("adjoint", [False, True])
@pytest.mark.parametrize(
    "values",
    [
        (3, 2, 1),
        (-3, 2, -1),
        (2**70, 3, 2**80),
        (np.int8(100), np.int8(3), np.int8(127)),
        (np.uint8(200), np.uint8(2), np.uint8(255)),
        (np.array([1, 2]), np.array([3, 4]), np.array([5, 6])),
        (np.array([[1], [2]]), np.array([2, 3]), 7),
        (sympy.Integer(3), sympy.Integer(2), sympy.Integer(1)),
    ],
)
def test_product_evaluation_original_order_types_and_exceptions(values, adjoint, width):
    from qualtran.bloqs.arithmetic.multiplication import PlusEqualProduct

    source = PlusEqualProduct(1, 1, width, is_adjoint=adjoint)
    left, right, result = values
    try:
        expected = source.on_classical_vals(a=left, b=right, result=result)
    except Exception as error:
        with pytest.raises(type(error)):
            accumulate_product_classical_values(
                left, right, result, width, adjoint=adjoint
            )
        return
    actual = accumulate_product_classical_values(
        left, right, result, width, adjoint=adjoint
    )
    assert actual[0] is left and actual[1] is right
    for a, e in zip(actual, expected.values(), strict=True):
        same_value(a, e)
    for a, e in zip(actual, source.apply(*values), strict=True):
        same_value(a, e)


def test_product_complete_small_domain_and_inverse():
    for width in (2, 3, 4):
        for left, right, result in product(range(2), range(2), range(1 << width)):
            computed = accumulate_product_classical_values(left, right, result, width)
            restored = accumulate_product_classical_values(
                *computed, width, adjoint=True
            )
            assert restored == (left, right, result)
