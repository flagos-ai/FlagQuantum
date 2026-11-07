"""Original representation methods retain pow cast asymmetry and numeric types."""

import numpy as np
import pytest
import sympy

from flagquantum.simulation.montgomery_values import (
    integer_to_montgomery_value,
    montgomery_inverse_value,
    montgomery_product_values,
    montgomery_to_integer_value,
    montgomery_value_domain,
)
from tests.unit.test_classical_modular_addition import compare_value_calls

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
WIDTHS = [0, 3, -1, sympy.Integer(3), sympy.Symbol("n", integer=True), np.int64(3), 3.0]
MODULI = [
    None,
    1,
    5,
    6,
    -5,
    0,
    sympy.Integer(5),
    sympy.Symbol("p", integer=True),
    np.int64(5),
    5.0,
]


@pytest.mark.parametrize("width", WIDTHS)
@pytest.mark.parametrize("modulus", MODULI)
@pytest.mark.parametrize("method", ["encode", "decode", "product", "inverse"])
def test_original_numeric_methods_values_types_cast_order_and_failures(
    width, modulus, method
):
    from qualtran import QMontgomeryUInt

    dtype = QMontgomeryUInt(width, modulus)
    sources = {
        "encode": dtype.uint_to_montgomery,
        "decode": dtype.montgomery_to_uint,
        "product": dtype.montgomery_product,
        "inverse": dtype.montgomery_inverse,
    }
    natives = {
        "encode": integer_to_montgomery_value,
        "decode": montgomery_to_integer_value,
        "product": montgomery_product_values,
        "inverse": montgomery_inverse_value,
    }
    for left, right in [
        (0, 0),
        (1, 2),
        (-1, 9),
        (np.int8(100), np.int8(3)),
        (np.uint8(200), np.uint8(2)),
        (np.array([1, 2]), np.array([2, 3])),
        (sympy.Integer(2), sympy.Integer(3)),
    ]:
        if method == "product":

            def original(x, y):
                return (sources[method](x, y),)

            def native(x, y):
                return (natives[method](x, y, width, modulus),)

            compare_value_calls(original, native, (left, right))
        else:

            def original(x):
                return (sources[method](x),)

            def native(x):
                return (natives[method](x, width, modulus),)

            compare_value_calls(original, native, (left,))


@pytest.mark.parametrize("width", WIDTHS)
@pytest.mark.parametrize("modulus", MODULI)
def test_original_domain_range_metadata_without_enumerating_large_words(width, modulus):
    from qualtran import QMontgomeryUInt

    source = QMontgomeryUInt(width, modulus)._bit_encoding
    try:
        expected = source.get_domain()
    except Exception as error:
        with pytest.raises(type(error)):
            montgomery_value_domain(width, modulus)
    else:
        actual = montgomery_value_domain(width, modulus)
        assert type(actual) is type(expected)
        assert (actual.start, actual.stop, actual.step) == (
            expected.start,
            expected.stop,
            expected.step,
        )


@pytest.mark.parametrize("width,modulus", [(3, 3), (4, 5), (5, 9), (6, 13)])
def test_complete_small_residue_conversion_product_and_inverse_laws(width, modulus):
    import math

    for x in range(modulus):
        encoded = integer_to_montgomery_value(x, width, modulus)
        assert montgomery_to_integer_value(encoded, width, modulus) == x
        for y in range(modulus):
            encoded_y = integer_to_montgomery_value(y, width, modulus)
            product = montgomery_product_values(encoded, encoded_y, width, modulus)
            assert (
                montgomery_to_integer_value(product, width, modulus)
                == (x * y) % modulus
            )
        if math.gcd(x, modulus) == 1:
            inverse = montgomery_inverse_value(encoded, width, modulus)
            assert montgomery_product_values(
                encoded, inverse, width, modulus
            ) == integer_to_montgomery_value(1, width, modulus)


def test_inverse_width_cast_policy_and_root_fermat_discrepancy_remain_distinct():
    from flagquantum.simulation.classical_montgomery_multiplication import (
        montgomery_multiply_classical_values,
    )

    assert integer_to_montgomery_value(2, sympy.Integer(3), 5) == 1
    with pytest.raises(TypeError):
        montgomery_inverse_value(2, sympy.Integer(3), 5)
    assert montgomery_product_values(1, 1, 5, 9) == 2
    with pytest.raises(AssertionError):
        montgomery_multiply_classical_values(1, 1, 5, 1, 9)
