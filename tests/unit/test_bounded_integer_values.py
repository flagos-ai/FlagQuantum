"""Bounded values retain source shape, type and comparison-order differences."""

import numpy as np
import pytest
import sympy

from flagquantum.simulation.bounded_integer_values import (
    bounded_integer_default_bound,
    bounded_integer_domain,
    validate_bounded_integer_array,
    validate_bounded_integer_shape,
    validate_bounded_integer_value,
)
from tests.unit.test_integer_encoding import compare_call

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
SHAPES = [
    (0, 0),
    (0, 1),
    (0, 2),
    (3, -1),
    (3, 0),
    (3, 3),
    (3, 8),
    (3, 9),
    (-1, 0),
    (-1, 1),
    (3, np.int64(3)),
    (np.int64(3), 3),
    (3, sympy.Integer(9)),
    (sympy.Integer(3), 9),
    (sympy.Symbol("n", integer=True), 3),
    (3, sympy.Symbol("b", integer=True)),
    (3, 3.5),
    (3, float("nan")),
    (3, True),
    (3, None),
]


def source(width, bound):
    from qualtran._infra.data_types import _BUInt

    return _BUInt(width, bound)


@pytest.mark.parametrize("width,bound", SHAPES)
def test_original_shape_and_domain_metadata(width, bound):
    compare_call(
        lambda _: source(width, bound) and None,
        lambda _: validate_bounded_integer_shape(width, bound),
        None,
    )
    compare_call(
        lambda _: source(width, bound).get_domain(),
        lambda _: bounded_integer_domain(width, bound),
        None,
    )


@pytest.mark.parametrize("width,bound", SHAPES)
def test_scalar_integer_type_and_negative_before_bound_validation(width, bound):
    def native(value):
        validate_bounded_integer_shape(width, bound)
        return validate_bounded_integer_value(value, bound)

    for value in (-1, 0, 1, 2, 3, 8, 9, True, np.int8(1), np.bool_(True), 1.0, None):
        compare_call(lambda x: source(width, bound).assert_valid_val(x), native, value)


@pytest.mark.parametrize("width,bound", SHAPES)
@pytest.mark.parametrize(
    "values",
    [
        np.array(0),
        np.array([-1, 0, 1]),
        np.array([0, 1]),
        np.array([0.5, 1.5]),
        np.array([np.nan]),
        np.array([True, False]),
        np.empty((2, 0)),
    ],
)
def test_array_range_only_rules_and_symbolic_short_circuits(width, bound, values):
    def native(value):
        validate_bounded_integer_shape(width, bound)
        return validate_bounded_integer_array(value, bound)

    compare_call(
        lambda x: source(width, bound).assert_valid_val_array(x), native, values
    )


@pytest.mark.parametrize("name", ["BQUInt", "BCUInt"])
@pytest.mark.parametrize(
    "width",
    [
        0,
        1,
        3,
        -1,
        np.int64(3),
        np.int64(63),
        np.int64(64),
        sympy.Integer(3),
        sympy.Symbol("n"),
    ],
)
def test_default_bound_expression_and_source_public_metadata(name, width):
    import qualtran

    def original(_):
        dtype = getattr(qualtran, name)(width)
        return dtype.bound

    def native(_):
        bound = bounded_integer_default_bound(width)
        validate_bounded_integer_shape(width, bound)
        return bound

    compare_call(original, native, None)


def test_nonpython_bound_domain_and_scalar_array_rules_are_not_normalized():
    assert bounded_integer_domain(3, True) == range(1)
    for bound in (np.int64(3), sympy.Integer(3), 3.0):
        validate_bounded_integer_shape(3, bound)
        with pytest.raises(ValueError):
            bounded_integer_domain(3, bound)
    for value in (1.0, np.bool_(True)):
        with pytest.raises(ValueError):
            validate_bounded_integer_value(value, 3)
    validate_bounded_integer_array(np.array([0.5, 1.5, np.nan]), 3)
    unknown = sympy.Symbol("b")
    with pytest.raises(ValueError, match="Negative"):
        validate_bounded_integer_value(-1, unknown)
    with pytest.raises(ValueError, match="Negative"):
        validate_bounded_integer_array(np.array([-1, 0]), unknown)
    with pytest.raises(TypeError):
        validate_bounded_integer_value(0, unknown)
