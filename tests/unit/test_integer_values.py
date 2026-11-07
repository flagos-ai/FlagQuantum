"""Pinned scalar register validation without dtype objects or coercion."""

from itertools import product

import numpy as np
import pytest
import sympy

from flagquantum.simulation.integer_values import (
    validate_integer_word_array,
    validate_integer_word_value,
)

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize(
    "name,width", tuple(product(("QUInt", "QInt", "QMontgomeryUInt"), (1, 3)))
)
def test_scalar_integer_register_bounds_match_original(name, width):
    import qualtran

    source = getattr(qualtran, name)(width)
    for value in (
        -9,
        -1,
        0,
        1,
        7,
        8,
        np.int8(1),
        np.uint8(1),
        True,
        1.0,
        np.array([1]),
        np.bool_(True),
    ):
        try:
            source.assert_valid_classical_val(value)
        except ValueError:
            with pytest.raises(ValueError):
                validate_integer_word_value(value, width, signed=name == "QInt")
        else:
            validate_integer_word_value(value, width, signed=name == "QInt")


n = sympy.Symbol("n", positive=True, integer=True)


@pytest.mark.parametrize(
    "name,width",
    tuple(
        product(("QUInt", "QInt", "QMontgomeryUInt"), (0, 1, 3, sympy.Integer(3), n))
    ),
)
def test_array_validator_matches_original_including_noninteger_diagnostic_values(
    name, width
):
    import qualtran

    source = getattr(qualtran, name)(width)
    signed = name == "QInt"
    for array in (
        np.array([0, 1]),
        np.array([-1, 0]),
        np.array([np.nan, 1.5]),
        np.array([], dtype=float),
        np.array([99]),
        np.array([1 + 2j]),
        np.ma.array([99], mask=[True]),
    ):
        try:
            source.assert_valid_classical_val_array(array)
        except (ValueError, TypeError, OverflowError) as exc:
            with pytest.raises(type(exc)):
                validate_integer_word_array(array, width, signed=signed)
        else:
            validate_integer_word_array(array, width, signed=signed)


@pytest.mark.parametrize(
    "name,width",
    tuple(product(("QUInt", "QInt", "QMontgomeryUInt"), (0, 3, sympy.Integer(3), n))),
)
def test_scalar_zero_and_symbolic_word_width_bounds_keep_original_exceptions(
    name, width
):
    import qualtran

    source = getattr(qualtran, name)(width)
    signed = name == "QInt"
    for value in (-1, 0, 1, 7, np.int64(1)):
        try:
            source.assert_valid_classical_val(value)
        except (ValueError, TypeError, OverflowError) as exc:
            with pytest.raises(type(exc)):
                validate_integer_word_value(value, width, signed=signed)
        else:
            validate_integer_word_value(value, width, signed=signed)
