"""Default source vectorization shared by distinct scalar codec consumers."""

import numpy as np
import pytest

from flagquantum.simulation.value_encoding import decode_value_array, encode_value_array
from tests.unit.test_integer_encoding import compare_call

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def codecs(kind):
    from qualtran import QBit, QInt, QIntOnesComp, QMontgomeryUInt

    source = {
        "bit": QBit(),
        "ones": QIntOnesComp(3),
        "signed": QInt(3),
        "montgomery": QMontgomeryUInt(3, 5),
    }[kind]
    return source, source.to_bits, source.from_bits


@pytest.mark.parametrize("kind", ["bit", "ones", "signed", "montgomery"])
@pytest.mark.parametrize(
    "values",
    [
        np.array(0),
        np.array([0, 1]),
        np.array([-1, 1]),
        np.array([[0, 1], [2, 3]]),
        np.empty((0, 2), dtype=np.int64),
    ],
)
def test_original_default_encoder_with_distinct_scalar_callbacks(kind, values):
    source, encode, _ = codecs(kind)
    compare_call(source.to_bits_array, lambda x: encode_value_array(x, encode), values)


@pytest.mark.parametrize("kind", ["bit", "ones", "signed", "montgomery"])
@pytest.mark.parametrize(
    "bits",
    [
        np.array(1),
        np.array([[0], [1]]),
        np.array([[True], [False]]),
        np.array([[1, 0, 1], [1, 1, 1]]),
        np.empty((0, 3), dtype=np.uint8),
    ],
)
def test_original_default_decoder_dtype_and_scalar_result_inference(kind, bits):
    source, _, decode = codecs(kind)
    compare_call(source.from_bits_array, lambda x: decode_value_array(x, decode), bits)


def test_integer_consumer_uses_replaceable_array_boundary(monkeypatch):
    import flagquantum.simulation.integer_encoding as integer

    marker = np.array([9], dtype=np.uint8)
    calls = []

    def encode(values, codec):
        calls.append(("encode", values, codec))
        return marker

    def decode(values, codec):
        calls.append(("decode", values, codec))
        return marker

    monkeypatch.setattr(integer, "encode_value_array", encode)
    monkeypatch.setattr(integer, "decode_value_array", decode)
    values, bits = np.array([0, 1]), np.array([[0], [1]])
    assert integer.vectorized_integer_array_to_bits(values, 3) is marker
    assert integer.vectorized_integer_array_from_bits(bits, 3) is marker
    assert calls[0][0] == "encode" and calls[0][1] is values and callable(calls[0][2])
    assert calls[1][0] == "decode" and calls[1][1] is bits and callable(calls[1][2])
