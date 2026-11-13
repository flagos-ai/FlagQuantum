"""Original numerical multiplication callbacks and all scalar/polynomial inputs."""

from __future__ import annotations

from importlib.metadata import version
from typing import Any

import numpy as np
import pytest
from sympy import Symbol

from flagquantum.simulation.finite_field_addition import binary_field_encoding
from flagquantum.simulation.finite_field_multiplication import (
    binary_linear_classical_value,
    binary_polynomial_product_classical_values,
    field_product_classical_values,
    field_scale_classical_value,
    field_scale_matrix,
    field_shift_classical_value,
)

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.fixture(scope="module", autouse=True)
def locked_reference() -> None:
    import kickmix

    assert version("qualtran") == "0.7.0"
    assert kickmix.__version__ == "0.1" and hasattr(kickmix, "Simulator")


@pytest.mark.parametrize("width", [1, 2, 3, 4, 5])
def test_product_original_value_callbacks_all_pairs(width: int) -> None:
    from qualtran import QGF
    from qualtran.bloqs.gf_arithmetic.gf2_multiplication import GF2Multiplication

    encoding = binary_field_encoding(width)
    original = GF2Multiplication(QGF(2, width))
    for x in encoding.gf_type.elements:
        for y in encoding.gf_type.elements:
            values = field_product_classical_values(encoding, x, y)
            assert values == original.call_classically(x=x, y=y)
            assert values[0] is x and values[1] is y
    with pytest.raises(AssertionError):
        field_product_classical_values(encoding, 1, encoding.gf_type(1))
    with pytest.raises(AssertionError):
        field_product_classical_values(encoding, encoding.gf_type(1), 1)
    with pytest.raises(AssertionError):
        field_product_classical_values(encoding, x, y, 0)


@pytest.mark.parametrize("width", [2, 3])
def test_plus_equal_original_value_callbacks_all_triples(width: int) -> None:
    from qualtran.bloqs.gf_arithmetic.gf2_multiplication import GF2Multiplication

    encoding = binary_field_encoding(width)
    original = GF2Multiplication(width, plus_equal_prod=True)
    for x in encoding.gf_type.elements:
        for y in encoding.gf_type.elements:
            for z in encoding.gf_type.elements:
                assert field_product_classical_values(
                    encoding, x, y, z, plus_equal_prod=True
                ) == original.call_classically(x=x, y=y, result=z)


@pytest.mark.parametrize("width", [1, 2, 3, 5, 8])
def test_scale_shift_and_reduction_matrices(width: int) -> None:
    from qualtran import QGF
    from qualtran.bloqs.gf_arithmetic.gf2_multiplication import (
        GF2MulK,
        GF2MulMBUC,
        GF2ShiftLeft,
        GF2ShiftRight,
    )

    encoding = binary_field_encoding(width)
    field = encoding.gf_type
    qgf = QGF(2, width)
    reduction_constant = int(field(2) ** width) if width > 1 else 1
    np.testing.assert_array_equal(
        field_scale_matrix(encoding, reduction_constant),
        GF2MulMBUC(qgf).reduction_matrix_q,
    )
    for constant in [0, 1, (1 << width) - 1]:
        original = GF2MulK(qgf, constant)
        np.testing.assert_array_equal(
            field_scale_matrix(encoding, constant), original.reduction_matrix_q
        )
        for value in field.elements:
            assert (
                field_scale_classical_value(encoding, value, constant)
                == original.call_classically(g=value)[0]
            )
    for inverse, factory in [(False, GF2ShiftRight), (True, GF2ShiftLeft)]:
        for count in [0, 1, 2, 3, 4]:
            original = factory(qgf, count)
            for value in field.elements:
                assert (
                    field_shift_classical_value(encoding, value, count, inverse=inverse)
                    == original.call_classically(f=value)[0]
                )
        with pytest.raises(TypeError, match="concrete"):
            field_shift_classical_value(
                encoding, field(1), Symbol("k"), inverse=inverse
            )
        with pytest.raises(AssertionError):
            field_shift_classical_value(encoding, 1, 1, inverse=inverse)


@pytest.mark.parametrize("width", [2, 3, 4, 5, 6, 8])
def test_linear_original_classical_method_and_inverse(width: int) -> None:
    from qualtran.bloqs.gf_arithmetic.gf2_multiplication import (
        GF2MulMBUC,
        SynthesizeLRCircuit,
    )

    matrix = GF2MulMBUC(width).reduction_matrix_q
    for value in range(1 << width):
        bits = np.array([(value >> i) & 1 for i in range(width)], dtype=np.int64)
        result = binary_linear_classical_value(matrix, bits)
        np.testing.assert_array_equal(
            result, SynthesizeLRCircuit(matrix).call_classically(q=bits)[0]
        )
        np.testing.assert_array_equal(
            binary_linear_classical_value(matrix, result, adjoint=True), bits
        )
        np.testing.assert_array_equal(
            binary_linear_classical_value(matrix, bits, adjoint=True),
            SynthesizeLRCircuit(matrix, is_adjoint=True).call_classically(q=bits)[0],
        )


@pytest.mark.parametrize("shift", [None, 1, 2, 3])
def test_binary_polynomial_original_full_value_callbacks(shift: int | None) -> None:
    from qualtran.bloqs.gf_arithmetic.gf2_multiplication import (
        BinaryPolynomialMultiplication,
        MultiplyPolyByOnePlusXk,
    )

    width = 2
    original: Any = (
        BinaryPolynomialMultiplication(width)
        if shift is None
        else MultiplyPolyByOnePlusXk(width, shift)
    )
    target_width = original.signature[-1].shape[0]
    for x in range(4):
        for y in range(4):
            for z in range(1 << target_width):
                f, g, h = [
                    np.array([(value >> i) & 1 for i in range(n)], dtype=np.int64)
                    for value, n in [(x, width), (y, width), (z, target_width)]
                ]
                left, right, result = binary_polynomial_product_classical_values(
                    f, g, h, shift=shift
                )
                assert left is f and right is g
                expected = original.on_classical_vals(f=f, g=g, h=h)
                assert result == expected["h"]
    with pytest.raises(TypeError, match="concrete"):
        binary_polynomial_product_classical_values(f, g, h, shift=Symbol("k"))


def test_source_value_callback_admission() -> None:
    encoding = binary_field_encoding(2)
    field = encoding.gf_type
    with pytest.raises(AssertionError):
        field_product_classical_values(
            encoding, field(1), field(2), plus_equal_prod=True
        )
    bits = np.array([1, 0], dtype=np.int64)
    bad: Any = [1, 0]
    for values in [(bad, bits, bits), (bits, bad, bits), (bits, bits, bad)]:
        with pytest.raises(AssertionError):
            binary_polynomial_product_classical_values(*values)
    with pytest.raises(AssertionError):
        binary_linear_classical_value(np.eye(2, dtype=np.int64), bad)
