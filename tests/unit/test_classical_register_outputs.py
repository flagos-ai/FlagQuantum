"""Original _get_in_vals assembly, dtype thresholds, errors and exact values."""

from itertools import product
from types import MappingProxyType

import numpy as np
import pytest
import sympy

from flagquantum.simulation.classical_registers import (
    bind_integer_register_values,
    collect_integer_register_value,
)

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def original_collect(assignments, width, shape, signed):
    from qualtran import QInt, QUInt, Register, RightDangle, Soquet
    from qualtran.simulation.classical_sim import _get_in_vals

    reg = Register("word", (QInt if signed else QUInt)(width), shape=shape)
    source = {
        Soquet(RightDangle, reg, idx=idx): value for idx, value in assignments.items()
    }
    return _get_in_vals(RightDangle, reg, source)


def compare_collect(assignments, width, shape, signed):
    try:
        expected = original_collect(assignments, width, shape, signed)
    except Exception as error:
        with pytest.raises(type(error)):
            collect_integer_register_value(
                assignments, width, shape=shape, signed=signed
            )
        return
    actual = collect_integer_register_value(
        assignments, width, shape=shape, signed=signed
    )
    assert type(actual) is type(expected)
    if isinstance(expected, np.ndarray):
        assert actual.dtype == expected.dtype
        for a, e in zip(actual.flat, expected.flat, strict=True):
            if isinstance(e, float) and np.isnan(e):
                assert isinstance(a, float) and np.isnan(a)
            else:
                assert a == e
        assert actual.flags.c_contiguous == expected.flags.c_contiguous
    else:
        assert actual is expected


@pytest.mark.parametrize("signed", [False, True])
@pytest.mark.parametrize(
    "width", [0, 1, 8, 9, 16, 17, 32, 33, 64, 65, sympy.Integer(9)]
)
@pytest.mark.parametrize("shape", [(), (2, 2), (2, 0)])
def test_dtype_thresholds_scalar_identity_empty_and_row_major(width, shape, signed):
    indices = product(*(range(n) for n in shape)) if shape else [()]
    # Assignment order is deliberately opposite to output traversal order.
    assignments = dict(reversed([(idx, i) for i, idx in enumerate(indices)]))
    compare_collect(MappingProxyType(assignments), width, shape, signed)


@pytest.mark.parametrize("signed", [False, True])
@pytest.mark.parametrize("value", [256, -1, 1.5, np.nan, sympy.Symbol("x"), 2**100])
@pytest.mark.parametrize("width", [8, 65])
def test_source_conversion_and_out_of_range_collection_is_not_revalidated(
    value, width, signed
):
    compare_collect({(0,): value}, width, (1,), signed)


@pytest.mark.parametrize("shape", [(), (2, 2), (2, 0)])
@pytest.mark.parametrize(
    "width", [sympy.Symbol("n", integer=True, positive=True), sympy.Integer(65)]
)
def test_symbolic_width_selection_and_scalar_bypass(width, shape):
    indices = product(*(range(n) for n in shape)) if shape else [()]
    assignments = {idx: sympy.Rational(1, 3) for idx in indices}
    compare_collect(assignments, width, shape, False)


@pytest.mark.parametrize(
    "shape,assignments", [((), {}), ((2, 2), {(0, 0): 1}), ((0,), {})]
)
def test_missing_entry_or_empty_array_matches_source(shape, assignments):
    compare_collect(assignments, 3, shape, False)


def test_bind_transform_collect_matches_unchanged_source_with_wide_values():
    from qualtran import QUInt
    from qualtran.bloqs.arithmetic.bitwise import XorK

    for width, values in ((8, [[0, 2], [7, 255]]), (65, [[0, 2**64], [2**65 - 1, 3]])):
        initial = np.asarray(values, dtype=object)
        assignments = bind_integer_register_values(
            "input", {"input": initial}, width, shape=(2, 2)
        )
        xor = XorK(QUInt(width), 1)
        mapped = {
            idx: xor.on_classical_vals(x=value)["x"]
            for idx, value in assignments.items()
        }
        actual = collect_integer_register_value(mapped, width, shape=(2, 2))
        expected = original_collect(mapped, width, (2, 2), False)
        np.testing.assert_array_equal(actual, expected)
        assert actual.dtype == expected.dtype
        restored = {
            idx: xor.on_classical_vals(x=value)["x"] for idx, value in mapped.items()
        }
        np.testing.assert_array_equal(
            collect_integer_register_value(restored, width, shape=(2, 2)), initial
        )
