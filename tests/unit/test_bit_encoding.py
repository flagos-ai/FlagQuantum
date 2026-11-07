"""Original quantum/classical single-bit value, identity and array semantics."""

import numpy as np
import pytest

from flagquantum.simulation.bit_encoding import (
    bit_array_from_bits,
    bit_array_to_bits,
    bit_from_bits,
    bit_to_bits,
    bit_value_domain,
    validate_bit_array,
    validate_bit_value,
)
from tests.unit.test_integer_encoding import compare_call

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("name", ["QBit", "CBit"])
@pytest.mark.parametrize("method", ["validate", "encode", "decode"])
def test_original_bit_scalar_type_value_and_identity_rules(name, method):
    import qualtran

    source = getattr(qualtran, name)()
    native = {
        "validate": validate_bit_value,
        "encode": bit_to_bits,
        "decode": bit_from_bits,
    }[method]
    original = {
        "validate": source.assert_valid_classical_val,
        "encode": source.to_bits,
        "decode": source.from_bits,
    }[method]
    values = [
        0,
        1,
        -1,
        2,
        False,
        True,
        np.bool_(True),
        np.uint8(1),
        0.0,
        1.0,
        np.nan,
        0j,
        np.array([1]),
        np.array([0, 1]),
        None,
    ]
    if method == "decode":
        values = [
            [],
            [0],
            [np.uint8(1)],
            [True],
            [1.5],
            [None],
            ["hello"],
            [2],
            [0, 1],
            np.array([True]),
            "1",
        ]
    for value in values:
        compare_call(original, native, value)


@pytest.mark.parametrize("name", ["QBit", "CBit"])
@pytest.mark.parametrize("method", ["validate", "encode", "decode"])
@pytest.mark.parametrize(
    "values",
    [
        np.array(1),
        np.array([0, 1]),
        np.array([0.0, 1.0]),
        np.array([True, False]),
        np.array([0, 2]),
        np.array([np.nan]),
        np.array([[0], [1]], dtype=np.uint8),
        np.array([[True], [False]]),
        np.array([["x"], ["y"]]),
        np.empty((0, 1), dtype=float),
        np.array([0, 1.0, True], dtype=object),
    ],
)
def test_original_bit_array_validation_codec_dtype_shape_and_failures(
    name, method, values
):
    import qualtran

    source = getattr(qualtran, name)()
    native = {
        "validate": validate_bit_array,
        "encode": bit_array_to_bits,
        "decode": bit_array_from_bits,
    }[method]
    original = {
        "validate": source.assert_valid_classical_val_array,
        "encode": source.to_bits_array,
        "decode": source.from_bits_array,
    }[method]
    compare_call(original, native, values)


def test_identity_decode_and_source_domain_are_not_replaced_by_integer_conversion():
    from qualtran import QBit

    marker = object()
    assert bit_from_bits([marker]) is marker
    assert QBit().from_bits([marker]) is marker
    assert tuple(bit_value_domain()) == tuple(QBit().get_classical_domain()) == (0, 1)
    assert bit_from_bits([True]) is True
    assert type(bit_to_bits(True)[0]) is int
    validate_bit_value(1.0)
    validate_bit_array(np.empty((2, 0), dtype=float))


def test_all_bit_words_full_complex_cnot_tensor_and_spectator_with_no_work():
    import torch
    from qualtran.bloqs.basic_gates import CNOT
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    import flagquantum as fq

    grid = torch.arange(1, 9, dtype=torch.float64)
    amplitudes = torch.complex(grid, 3 * grid - 5)
    amplitudes /= amplitudes.norm()
    indices = []
    for control in bit_value_domain():
        for target in bit_value_domain():
            c, t = bit_from_bits(bit_to_bits(control)), bit_from_bits(
                bit_to_bits(target)
            )
            indices.append((int(c) << 1) | int(t))
    assert indices == [0, 1, 2, 3]
    expected = torch.as_tensor(CNOT().tensor_contract().copy()) @ amplitudes.reshape(
        4, 2
    )
    circuit = fq.Circuit(3, inputs=amplitudes, dtype=torch.complex128).cx(0, 1)
    torch.testing.assert_close(
        circuit.state().reshape(4, 2), expected, rtol=0, atol=2e-12
    )
    torch.testing.assert_close(
        circuit.cx(0, 1).state().flatten(), amplitudes, rtol=0, atol=2e-12
    )
    gates = get_cost_value(CNOT(), QECGatesCost())
    assert gates.clifford == 1 and gates.total_t_count() == gates.measurement == 0


def test_array_codecs_use_shared_replaceable_boundary(monkeypatch):
    import flagquantum.simulation.bit_encoding as native

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
    assert native.bit_array_to_bits(values) is marker
    assert native.bit_array_from_bits(bits) is marker
    assert calls[0][0] == "encode" and calls[0][1] is values and callable(calls[0][2])
    assert calls[1][0] == "decode" and calls[1][1] is bits and callable(calls[1][2])
