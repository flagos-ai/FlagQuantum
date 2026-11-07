"""Unmodified ones' complement domain, endpoint diagnostics and value codecs."""

import numpy as np
import pytest
import sympy

from flagquantum.simulation.ones_complement_encoding import (
    ones_complement_array_from_bits,
    ones_complement_array_to_bits,
    ones_complement_domain,
    ones_complement_from_bits,
    ones_complement_to_bits,
    validate_ones_complement_array,
    validate_ones_complement_value,
)
from tests.unit.test_integer_encoding import compare_call

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
WIDTHS = [
    0,
    1,
    2,
    3,
    8,
    65,
    np.int64(1),
    np.int64(3),
    sympy.Integer(1),
    sympy.Integer(3),
    sympy.Symbol("n"),
    -1,
]


def source(name, width):
    import qualtran

    return getattr(qualtran, name)(width)


@pytest.mark.parametrize("name", ["QIntOnesComp", "CIntOnesComp"])
@pytest.mark.parametrize("width", WIDTHS)
@pytest.mark.parametrize("method", ["validate", "encode", "decode"])
def test_scalar_original_order_width_guards_types_and_errors(name, width, method):
    try:
        dtype = source(name, width)
    except Exception as error:
        native = {
            "validate": validate_ones_complement_value,
            "encode": ones_complement_to_bits,
            "decode": ones_complement_from_bits,
        }[method]
        with pytest.raises(type(error)):
            native([0, 0] if method == "decode" else 0, width)
        return
    native = {
        "validate": validate_ones_complement_value,
        "encode": ones_complement_to_bits,
        "decode": ones_complement_from_bits,
    }[method]
    original = {
        "validate": dtype.assert_valid_classical_val,
        "encode": dtype.to_bits,
        "decode": dtype.from_bits,
    }[method]
    values = [
        -129,
        -128,
        -4,
        -3,
        -1,
        0,
        1,
        3,
        4,
        128,
        np.int8(-128),
        np.uint8(1),
        True,
        np.bool_(True),
        1.0,
        np.array([1]),
    ]
    if method == "decode":
        values = [
            [],
            [0],
            [1],
            [0, 0],
            [1, 1],
            [1, 0, 1],
            [1, 1, 1],
            [-1, -1],
            [2, 3],
            [True, False],
            np.array([1, 0, 1], dtype=np.uint8),
        ]
    for value in values:
        compare_call(original, lambda x: native(x, width), value)


@pytest.mark.parametrize("width", WIDTHS)
@pytest.mark.parametrize("method", ["validate", "encode", "decode"])
@pytest.mark.parametrize(
    "values",
    [
        np.array(0),
        np.array([-1, 0, 1]),
        np.array([4]),
        np.array([0.0, 1.0]),
        np.array([[1, 0, 1], [1, 1, 1]], dtype=np.uint8),
        np.empty((0, 3), dtype=np.int64),
        np.empty((2, 0), dtype=np.uint8),
        np.array([0, 2**63], dtype=object),
    ],
)
def test_array_source_default_validation_and_vectorization(width, method, values):
    native = {
        "validate": validate_ones_complement_array,
        "encode": ones_complement_array_to_bits,
        "decode": ones_complement_array_from_bits,
    }[method]
    try:
        dtype = source("QIntOnesComp", width)
    except Exception as error:
        with pytest.raises(type(error)):
            native(values, width)
        return
    original = {
        "validate": dtype.assert_valid_classical_val_array,
        "encode": dtype.to_bits_array,
        "decode": dtype.from_bits_array,
    }[method]
    compare_call(original, lambda x: native(x, width), values)


@pytest.mark.parametrize("width", WIDTHS)
def test_domain_original_range_metadata_without_large_enumeration(width):
    try:
        expected = source("QIntOnesComp", width)._bit_encoding.get_domain()
    except Exception as error:
        with pytest.raises(type(error)):
            ones_complement_domain(width)
    else:
        actual = ones_complement_domain(width)
        assert type(actual) is type(expected)
        assert (actual.start, actual.stop, actual.step) == (
            expected.start,
            expected.stop,
            expected.step,
        )


def test_endpoint_validation_and_negative_zero_are_not_silently_canonicalized():
    from qualtran import QIntOnesComp

    dtype = QIntOnesComp(3)
    for value in (-4, 4):
        dtype.assert_valid_classical_val(value)
        validate_ones_complement_value(value, 3)
        with pytest.raises(ValueError):
            dtype.to_bits(value)
        with pytest.raises(ValueError):
            ones_complement_to_bits(value, 3)
    assert ones_complement_from_bits([1, 1, 1], 3) == 0
    assert ones_complement_to_bits(0, 3) == [0, 0, 0]
    assert (
        tuple(ones_complement_domain(3))
        == tuple(dtype.get_classical_domain())
        == tuple(range(-3, 4))
    )
    assert type(ones_complement_from_bits([-1, -1], 3)) is float


def test_array_codecs_use_shared_replaceable_boundary(monkeypatch):
    import flagquantum.simulation.ones_complement_encoding as native

    marker = np.array([9], dtype=np.uint8)
    calls = []

    def encode(values, codec):
        calls.append(("encode", values, codec))
        return marker

    def decode(values, codec):
        calls.append(("decode", values, codec))
        return marker

    monkeypatch.setattr(native, "encode_value_array", encode)
    monkeypatch.setattr(native, "decode_value_array", decode)
    values, bits = np.array([0, 1]), np.array([[0], [1]])
    assert native.ones_complement_array_to_bits(values, 3) is marker
    assert native.ones_complement_array_from_bits(bits, 3) is marker
    assert calls[0][0] == "encode" and calls[0][1] is values and callable(calls[0][2])
    assert calls[1][0] == "decode" and calls[1][1] is bits and callable(calls[1][2])
