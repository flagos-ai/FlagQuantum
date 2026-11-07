"""Original default Montgomery array codecs and actual shared vectorized routes."""

import numpy as np
import pytest
import sympy

from flagquantum.simulation.integer_encoding import (
    integer_array_from_bits,
    integer_array_to_bits,
    vectorized_integer_array_from_bits,
    vectorized_integer_array_to_bits,
)
from tests.unit.test_integer_encoding import compare_call

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
WIDTHS = [
    0,
    1,
    3,
    64,
    65,
    np.int64(3),
    sympy.Integer(3),
    sympy.Symbol("n", integer=True),
]


@pytest.mark.parametrize("width", WIDTHS)
@pytest.mark.parametrize(
    "values",
    [
        np.array(1),
        np.array([0, 1]),
        np.array([7]),
        np.array([-1, 1]),
        np.array([[0, 1], [2, 3]]),
        np.empty((0, 2), dtype=np.int64),
        np.array([0, 2**64], dtype=object),
        np.array([1.5]),
        np.array([True]),
    ],
)
def test_montgomery_default_array_encoding_matches_original(width, values):
    from qualtran import QMontgomeryUInt

    compare_call(
        QMontgomeryUInt(width, 5).to_bits_array,
        lambda x: vectorized_integer_array_to_bits(x, width),
        values,
    )


@pytest.mark.parametrize("width", WIDTHS)
@pytest.mark.parametrize(
    "bits",
    [
        np.array(1),
        np.array([0, 1]),
        np.array([[0, 1], [1, 0]]),
        np.array([[2, -1]]),
        np.array([[True, False]]),
        np.empty((2, 0), dtype=np.uint8),
        np.empty((0, 3), dtype=np.uint8),
        np.zeros((2, 3, 3), dtype=np.uint8),
    ],
)
def test_montgomery_default_array_decode_is_not_the_unsigned_fast_contract(width, bits):
    from qualtran import QMontgomeryUInt

    compare_call(
        QMontgomeryUInt(width, 5).from_bits_array,
        lambda x: vectorized_integer_array_from_bits(x, width),
        bits,
    )


@pytest.mark.parametrize("signed,width", [(True, 3), (True, 65), (False, 65)])
def test_existing_signed_and_wide_unsigned_paths_use_replaceable_shared_helpers(
    monkeypatch, signed, width
):
    import flagquantum.simulation.integer_encoding as module

    marker = np.array([9], dtype=np.uint8)
    calls = []

    def encode(values, bits, *, signed=False):
        calls.append(("encode", values, bits, signed))
        return marker

    def decode(values, bits, *, signed=False):
        calls.append(("decode", values, bits, signed))
        return marker

    monkeypatch.setattr(module, "vectorized_integer_array_to_bits", encode)
    monkeypatch.setattr(module, "vectorized_integer_array_from_bits", decode)
    values = np.array([1])
    bits = np.zeros((1, width), dtype=np.uint8)
    assert integer_array_to_bits(values, width, signed=signed) is marker
    assert integer_array_from_bits(bits, width, signed=signed) is marker
    assert (
        calls[0][0] == "encode"
        and calls[0][1] is values
        and calls[0][2:] == (width, signed)
    )
    assert (
        calls[1][0] == "decode"
        and calls[1][1] is bits
        and calls[1][2:] == (width, signed)
    )


def test_default_and_optimized_unsigned_routes_preserve_their_actual_differences():
    values = np.array([[0, 1], [2, 3]])
    assert vectorized_integer_array_to_bits(values, 3).shape == (2, 2, 3)
    with pytest.raises(ValueError):
        integer_array_to_bits(values, 3)
    np.testing.assert_array_equal(
        vectorized_integer_array_from_bits(np.array([[1, 0]]), 3), [2]
    )
    with pytest.raises(ValueError):
        integer_array_from_bits(np.array([[1, 0]]), 3)
