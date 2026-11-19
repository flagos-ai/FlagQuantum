# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
"""Remaining complete-file dtype scenarios use existing native owners."""

from collections.abc import Callable
from typing import Any

import pytest

from flagquantum.core.ir.program.register_types import opaque_register_type
from flagquantum.core.ir.program.registers import Register
from flagquantum.simulation.bounded_integer_encoding import (
    bounded_integer_to_bits,
)
from flagquantum.simulation.bounded_integer_values import validate_bounded_integer_value
from flagquantum.simulation.classical_domains import register_classical_domain
from flagquantum.simulation.classical_registers import bind_opaque_register_values
from flagquantum.simulation.finite_field_encoding import FiniteFieldEncoding
from flagquantum.simulation.finite_field_polynomial_encoding import (
    FiniteFieldPolynomialEncoding,
)
from flagquantum.simulation.value_encoding import decode_value_array, encode_value_array
from tests.unit.test_integer_encoding import compare_call as original_compare_call

np = pytest.importorskip("numpy")
sympy = pytest.importorskip("sympy")
pytest.importorskip("qualtran")
pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
compare_call: Callable[[Callable[[Any], Any], Callable[[Any], Any], Any], None] = (
    original_compare_call
)


def test_legacy_custom_scalar_callbacks_use_shared_native_vectorization() -> None:
    from qualtran._infra.data_types import _BitEncodingShim
    from qualtran._infra.data_types_test import LegacyBQUInt

    legacy = LegacyBQUInt(8, 230)
    original = _BitEncodingShim(legacy)
    for value in (-1, 0, 5, 229, 230, 1.5, np.int64(7)):
        compare_call(
            original.to_bits, lambda x: bounded_integer_to_bits(x, 8, 230), value
        )
        compare_call(
            original.assert_valid_val,
            lambda x: validate_bounded_integer_value(x, 230),
            value,
        )
    for bits in ([0] * 8, [1] * 8, [1], [1, 9], []):
        # The custom legacy decoder delegates directly to UInt, including out-of-bound results.
        from flagquantum.simulation.integer_encoding import integer_from_bits

        compare_call(original.from_bits, lambda x: integer_from_bits(x, 8), bits)
    for array in (
        np.array([0, 5, 229]),
        np.array([[0, 1], [2, 3]]),
        np.array([1.5]),
        np.empty((0,)),
    ):
        compare_call(
            original.to_bits_array,
            lambda x: encode_value_array(
                x, lambda v: bounded_integer_to_bits(v, 8, 230)
            ),
            array,
        )
    for bits in (
        np.array([[0] * 8, [1] * 8]),
        np.array([0] * 8),
        np.empty((0, 8), dtype=int),
    ):
        compare_call(
            original.from_bits_array,
            lambda x: decode_value_array(
                x, lambda row: integer_from_bits(row.tolist(), 8)
            ),
            bits,
        )
    assert original.bitsize == 8
    assert list(original.get_domain()) == list(range(230))
    # The old shim ignores the supplied array diagnostic and calls scalar validation.
    for array in (np.array([0, 5]), np.array([1.5]), np.array([230]), np.empty((0,))):

        def validate(values: Any) -> None:
            for value in values.reshape(-1):
                validate_bounded_integer_value(value, 230)

        compare_call(
            lambda x: original.assert_valid_val_array(x, debug_str="custom"),
            validate,
            array,
        )


@pytest.mark.parametrize("value", [None, "opaque", -123, 2**200, 0.5, [1, 2]])
def test_opaque_source_validation_is_independent_from_bit_codec(value: Any) -> None:
    from qualtran import QAny

    original = QAny(3)
    original.assert_valid_classical_val(value)
    original.assert_valid_classical_val_array(np.array([value], dtype=object))
    assert bind_opaque_register_values("x", {"x": value}) == {(): value}
    with pytest.raises(TypeError, match="Ambiguous domain"):
        original.get_classical_domain()
    with pytest.raises(TypeError, match="Ambiguous domain"):
        register_classical_domain(Register("x", opaque_register_type(3)))


@pytest.mark.parametrize("degree", [0, 1, 3, sympy.Symbol("d", integer=True)])
def test_polynomial_symbolic_declaration_uses_existing_metadata(degree: Any) -> None:
    from qualtran import QGF, CGFPoly, QGFPoly

    field = FiniteFieldEncoding(2, 2)
    native = FiniteFieldPolynomialEncoding(degree, field)
    for classical, original in (
        (False, QGFPoly(degree, QGF(2, 2))),
        (True, CGFPoly(degree, QGF(2, 2))),
    ):
        descriptor = native.register_type(classical=classical)
        assert descriptor.num_bits == original.num_bits
        assert descriptor.num_qubits == original.num_qubits
        assert descriptor.num_cbits == original.num_cbits
        assert descriptor.is_symbolic() == original.is_symbolic()
        assert (
            dict(descriptor.parameters)["order"] == original.iteration_length_or_zero()
        )
        assert dict(descriptor.parameters)["polynomial_degree"] == degree
