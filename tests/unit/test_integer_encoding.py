"""Pinned scalar/array integer codecs, including original diagnostic boundaries."""

from copy import deepcopy

import numpy as np
import pytest
import sympy

from flagquantum.simulation.integer_encoding import (
    integer_array_from_bits,
    integer_array_to_bits,
    integer_from_bits,
    integer_to_bits,
)

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
WIDTHS = [
    0,
    1,
    3,
    8,
    64,
    65,
    -1,
    np.int64(3),
    np.int64(64),
    sympy.Integer(3),
    sympy.Symbol("n", integer=True, positive=True),
    1.0,
]


def source(width, signed):
    from qualtran import QInt, QUInt

    return (QInt if signed else QUInt)(width)


def compare_call(original, native, value):
    left, right = deepcopy(value), deepcopy(value)
    try:
        expected = original(left)
    except Exception as error:
        with pytest.raises(type(error)):
            native(right)
        return
    actual = native(right)
    assert type(actual) is type(expected)
    if isinstance(expected, np.ndarray):
        assert actual.dtype == expected.dtype
        np.testing.assert_array_equal(actual, expected)
    else:
        assert actual == expected


@pytest.mark.parametrize("signed", [False, True])
@pytest.mark.parametrize("width", WIDTHS)
def test_scalar_encode_and_decode_match_unchanged_source(width, signed):
    original = source(width, signed)
    for value in (
        -9,
        -1,
        0,
        1,
        7,
        8,
        2**63,
        2**64,
        True,
        np.int8(1),
        np.uint64(2**64 - 1),
        np.bool_(True),
        1.5,
        np.array([1]),
    ):
        compare_call(
            original.to_bits, lambda x: integer_to_bits(x, width, signed=signed), value
        )
    for bits in (
        [],
        [0],
        [1],
        [0, 1],
        [1, 0, 1],
        [1, 9],
        [-1, 1],
        [10],
        [True, False],
        ["1", "0"],
        np.array([1, 0], dtype=np.uint8),
    ):
        compare_call(
            original.from_bits,
            lambda x: integer_from_bits(x, width, signed=signed),
            bits,
        )


@pytest.mark.parametrize("signed", [False, True])
@pytest.mark.parametrize(
    "width", [0, 1, 2, 3, 8, 64, 65, sympy.Integer(3), sympy.Symbol("n")]
)
@pytest.mark.parametrize(
    "values",
    [
        np.array(1),
        np.array([0, 1]),
        np.array([-1, 1]),
        np.array([7]),
        np.array([1], dtype=np.uint8),
        np.array([1], dtype=np.uint64),
        np.array([1.5]),
        np.array([True]),
        np.array([[0, 1], [1, 0]]),
        np.arange(6).reshape(2, 3),
        np.array([], dtype=np.int64),
        np.array([0, 2**64], dtype=object),
    ],
)
def test_array_encoding_shape_dtype_and_original_failures(width, signed, values):
    compare_call(
        source(width, signed).to_bits_array,
        lambda x: integer_array_to_bits(x, width, signed=signed),
        values,
    )


@pytest.mark.parametrize("signed", [False, True])
@pytest.mark.parametrize(
    "width", [0, 1, 2, 3, 64, 65, sympy.Integer(3), sympy.Symbol("n")]
)
@pytest.mark.parametrize(
    "bits",
    [
        np.array(1),
        np.array([0, 1]),
        np.array([[0, 1], [1, 0]]),
        np.array([[2, -1]]),
        np.array([[True, False]]),
        np.array([[1.5, 0]]),
        np.empty((2, 0), dtype=np.uint8),
        np.empty((0, 3), dtype=np.uint8),
        np.zeros((2, 3, 3), dtype=np.uint8),
    ],
)
def test_array_decoding_exact_width_checks_broadcasts_and_inference(
    width, signed, bits
):
    compare_call(
        source(width, signed).from_bits_array,
        lambda x: integer_array_from_bits(x, width, signed=signed),
        bits,
    )


@pytest.mark.parametrize("signed", [False, True])
@pytest.mark.parametrize("first_large", [False, True])
def test_above_64_bits_vectorized_output_dtype_follows_first_result(
    signed, first_large
):
    original = source(65, signed)
    values = [2**63, 0] if first_large else [0, 2**63]
    if signed:
        values = [-(2**63), 0] if first_large else [0, -(2**63)]
    bits = np.asarray([original.to_bits(value) for value in values], dtype=np.uint8)
    compare_call(
        original.from_bits_array,
        lambda x: integer_array_from_bits(x, 65, signed=signed),
        bits,
    )
    compare_call(
        original.to_bits_array,
        lambda x: integer_array_to_bits(x, 65, signed=signed),
        np.asarray(values, dtype=object),
    )


def test_scalar_and_array_diagnostics_are_not_normalized_to_one_contract():
    assert integer_to_bits(0, 0) == [0]  # Original scalar format's zero-width result.
    assert integer_from_bits([1, 0, 1], 1) == 5  # Unsigned width is ignored.
    assert (
        integer_from_bits([1, 9], 1, signed=True) == -1
    )  # Signed one-bit suffix is ignored.
    np.testing.assert_array_equal(integer_array_to_bits(np.array([7]), 2), [[1, 1]])
    with pytest.raises(ValueError):
        integer_to_bits(7, 2)
