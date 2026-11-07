"""Unmodified shaped Partition callbacks and shared native codec composition."""

from functools import partial

import numpy as np
import pytest

from flagquantum.simulation.classical_partition import (
    PartitionValuePort,
    partition_classical_value,
    unpartition_classical_values,
)
from tests.unit.test_classical_comparison import compare_calls

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def configuration(kind, shape):
    from qualtran import QBit, QFxp, QInt, QUInt, Register
    from qualtran.bloqs.bookkeeping.partition import Partition

    from flagquantum.simulation.bit_encoding import (
        bit_array_from_bits,
        bit_array_to_bits,
        bit_from_bits,
    )
    from flagquantum.simulation.fixed_point_encoding import (
        fixed_point_array_from_bits,
        fixed_point_array_to_bits,
        fixed_point_from_bits,
    )
    from flagquantum.simulation.integer_encoding import (
        integer_array_from_bits,
        integer_array_to_bits,
        integer_from_bits,
        integer_to_bits,
    )

    width = 1 if kind == "bit" else 2
    if kind == "bit":
        dtype, decode, decode_array, encode_array = (
            QBit(),
            bit_from_bits,
            bit_array_from_bits,
            bit_array_to_bits,
        )
    elif kind == "fixed":
        dtype = QFxp(width, 1)
        decode = partial(fixed_point_from_bits, width=width, fractional_width=1)
        decode_array = partial(
            fixed_point_array_from_bits, width=width, fractional_width=1
        )
        encode_array = partial(
            fixed_point_array_to_bits, width=width, fractional_width=1
        )
    else:
        signed = kind == "int"
        dtype = QInt(width) if signed else QUInt(width)
        decode = partial(integer_from_bits, width=width, signed=signed)
        decode_array = partial(integer_array_from_bits, width=width, signed=signed)
        encode_array = partial(integer_array_to_bits, width=width, signed=signed)
    native = (
        PartitionValuePort("cells", width, shape, decode, decode_array, encode_array),
        PartitionValuePort(
            "flag", 1, (), bit_from_bits, bit_array_from_bits, bit_array_to_bits
        ),
    )
    n = int(np.prod(shape + (width,))) + 1
    source = Partition(
        n, (Register("cells", dtype, shape=shape), Register("flag", QBit()))
    )
    return (
        source,
        native,
        partial(integer_to_bits, width=n),
        partial(integer_from_bits, width=n),
    )


CONFIGS = [
    (kind, shape)
    for kind in ("bit", "uint", "int", "fixed")
    for shape in ((), (1,), (2,), (1, 2), (0,))
]


@pytest.mark.parametrize("kind,shape", CONFIGS)
@pytest.mark.parametrize("selector", [0, 1, 2, 3, 4])
def test_original_partition_word_slices_scalar_array_types_and_empty_axes(
    kind, shape, selector
):
    source, ports, encode, _ = configuration(kind, shape)
    value = [0, 1, (1 << source.n) - 1, 1 << source.n, np.int64(1)][selector]
    compare_calls(
        lambda x: tuple(source.on_classical_vals(x=x).values()),
        lambda x: tuple(partition_classical_value(x, encode, ports).values()),
        (value,),
    )


@pytest.mark.parametrize("kind,shape", CONFIGS)
@pytest.mark.parametrize(
    "cells", [0, 1, -1, [], [0, 1], [[0, 1]], np.array([0, 1]), np.array([[1], [0]])]
)
def test_original_unpartition_uses_value_ravel_not_metadata_shape(kind, shape, cells):
    source, ports, _, decode = configuration(kind, shape)
    values = {"cells": cells, "flag": np.uint8(1), "extra": 999}
    compare_calls(
        lambda v: tuple(source.adjoint().on_classical_vals(**v).values()),
        lambda v: (unpartition_classical_values(v, ports, decode),),
        (values,),
    )


@pytest.mark.parametrize("kind", ["bit", "uint", "int", "fixed"])
def test_all_small_words_round_trip_and_declared_port_order(kind):
    source, ports, encode, decode = configuration(kind, (2,))
    for value in range(1 << source.n):
        expected = source.on_classical_vals(x=value)
        actual = partition_classical_value(value, encode, ports)
        assert list(actual) == list(expected) == ["cells", "flag"]
        assert (
            unpartition_classical_values(actual, ports, decode)
            == source.adjoint().on_classical_vals(**expected)["x"]
            == value
        )
    with pytest.raises(KeyError):
        unpartition_classical_values({"flag": 0}, ports, decode)


def test_unpartition_reuses_shared_join_boundary_and_replacement(monkeypatch):
    import flagquantum.simulation.classical_partition as module

    _, ports, _, decode = configuration("uint", (2,))
    seen = []
    marker = object()

    def replacement(bits, decoder):
        seen.append((bits.tolist(), decoder))
        return marker

    monkeypatch.setattr(module, "join_classical_value", replacement)
    assert (
        unpartition_classical_values({"cells": [[1], [2]], "flag": 1}, ports, decode)
        is marker
    )
    assert seen == [([0, 1, 1, 0, 1], decode)]


def test_port_codec_calls_retain_slice_and_numpy_conversion_order():
    calls = []
    marker = object()

    def scalar_decode(bits):
        calls.append(("scalar", type(bits), list(bits)))
        return marker

    def array_decode(bits):
        calls.append(("array", bits.shape, bits.tolist()))
        return np.array([7])

    def array_encode(values):
        calls.append(("encode_array", values.shape))
        return np.array([[0, 1]])

    ports = (
        PartitionValuePort("scalar", 1, (), scalar_decode, array_decode, array_encode),
        PartitionValuePort("array", 2, (1,), scalar_decode, array_decode, array_encode),
    )
    output = partition_classical_value(5, lambda x: [1, 0, 1], ports)
    assert output["scalar"] is marker
    assert calls == [("scalar", list, [1]), ("array", (1, 2), [[0, 1]])]
