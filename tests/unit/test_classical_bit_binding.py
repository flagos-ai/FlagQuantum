"""Pinned bookkeeping values use native codecs with original conversion order."""

from functools import partial
from itertools import product

import numpy as np
import pytest

from flagquantum.simulation.classical_bit_binding import (
    allocate_classical_value,
    cast_classical_value,
    free_classical_value,
    join_classical_value,
    split_classical_value,
)
from tests.unit.test_classical_comparison import compare_calls

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def codecs(kind, width):
    from qualtran import QBit, QFxp, QInt, QMontgomeryUInt, QUInt

    from flagquantum.simulation.bit_encoding import bit_from_bits, bit_to_bits
    from flagquantum.simulation.fixed_point_encoding import (
        fixed_point_from_bits,
        fixed_point_to_bits,
    )
    from flagquantum.simulation.integer_encoding import (
        integer_from_bits,
        integer_to_bits,
    )

    if kind == "bit":
        return QBit(), bit_to_bits, bit_from_bits
    if kind == "fixed":
        return (
            QFxp(width, width // 2),
            partial(fixed_point_to_bits, width=width, fractional_width=width // 2),
            partial(fixed_point_from_bits, width=width, fractional_width=width // 2),
        )
    signed = kind == "int"
    dtype = (
        QInt(width)
        if signed
        else QMontgomeryUInt(width) if kind == "montgomery" else QUInt(width)
    )
    return (
        dtype,
        partial(integer_to_bits, width=width, signed=signed),
        partial(integer_from_bits, width=width, signed=signed),
    )


CONFIGS = [
    (kind, width)
    for kind in ("uint", "int", "fixed", "montgomery")
    for width in (1, 3, 8)
] + [("bit", 1)]


@pytest.mark.parametrize("kind,width", CONFIGS)
@pytest.mark.parametrize(
    "value", [0, 1, -1, 2, 7, 256, np.uint8(1), np.array([1]), np.array([0, 1]), 0.5]
)
def test_split_original_codec_domain_and_numpy_dtype(kind, width, value):
    from qualtran.bloqs.bookkeeping import Split

    dtype, encode, _ = codecs(kind, width)
    source = Split(dtype)
    compare_calls(
        lambda v: tuple(source.on_classical_vals(v).values()),
        lambda v: (split_classical_value(v, encode),),
        (value,),
    )


@pytest.mark.parametrize("kind,width", CONFIGS)
@pytest.mark.parametrize(
    "bits",
    [
        np.array(x)
        for x in [[], [0], [1], [0, 1, 0], [1, 1, 1], [2], [-1], [[0, 1]], [0.0, 1.0]]
    ],
)
def test_join_original_tolist_decode_shapes_types_and_failures(kind, width, bits):
    from qualtran.bloqs.bookkeeping import Join

    dtype, _, decode = codecs(kind, width)
    source = Join(dtype)
    compare_calls(
        lambda b: tuple(source.on_classical_vals(b).values()),
        lambda b: (join_classical_value(b, decode),),
        (bits,),
    )


@pytest.mark.parametrize("kind,width", CONFIGS)
@pytest.mark.parametrize("dirty", [False, True])
def test_allocate_original_zero_decode_even_when_dirty(kind, width, dirty):
    from qualtran.bloqs.bookkeeping import Allocate

    dtype, _, decode = codecs(kind, width)
    compare_calls(
        lambda: tuple(Allocate(dtype, dirty).on_classical_vals().values()),
        lambda: (allocate_classical_value(width, decode),),
        (),
    )


@pytest.mark.parametrize(
    "value",
    [0, 1, -1, None, np.int8(0), np.array([0]), np.array([1]), np.array([0, 1])],
)
@pytest.mark.parametrize("dirty", [False, True])
def test_free_original_comparison_before_dirty_short_circuit(value, dirty):
    from qualtran import QUInt
    from qualtran.bloqs.bookkeeping import Free

    compare_calls(
        lambda v: tuple(Free(QUInt(3), dirty).on_classical_vals(v).values()),
        lambda v: tuple(free_classical_value(v, dirty=dirty).values()),
        (value,),
    )


@pytest.mark.parametrize(
    "input_kind,output_kind",
    list(product(("uint", "int", "fixed", "montgomery"), repeat=2)),
)
@pytest.mark.parametrize("value", [-4, -1, 0, 1, 3, 7, np.int8(1)])
def test_cast_original_input_encode_then_output_decode(input_kind, output_kind, value):
    from qualtran.bloqs.bookkeeping import Cast

    inp, encode, _ = codecs(input_kind, 3)
    out, _, decode = codecs(output_kind, 3)
    source = Cast(inp, out)
    compare_calls(
        lambda v: tuple(source.on_classical_vals(v).values()),
        lambda v: (cast_classical_value(v, encode, decode),),
        (value,),
    )


def test_real_codec_replacement_and_conversion_object_ownership():
    marker = object()
    encoded = [1, 0, 1]
    calls = []

    def encode(value):
        assert value is marker
        calls.append("encode")
        return encoded

    def decode(bits):
        calls.append(("decode", bits is encoded, type(bits)))
        return marker

    assert cast_classical_value(marker, encode, decode) is marker
    assert calls == ["encode", ("decode", True, list)]
    calls.clear()
    bits = split_classical_value(marker, encode)
    assert bits.dtype == np.int64 and bits.tolist() == encoded
    assert join_classical_value(bits, decode) is marker
    assert calls == ["encode", ("decode", False, list)]
    calls.clear()
    assert allocate_classical_value(3, decode) is marker
    assert calls == [("decode", False, list)]
