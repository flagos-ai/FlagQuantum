"""Pinned scalar register validation without dtype objects or coercion."""

from itertools import product

import numpy as np
import pytest

from flagquantum.simulation.integer_values import validate_integer_word_value

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize(
    "name,width", tuple(product(("QUInt", "QInt", "QMontgomeryUInt"), (1, 3)))
)
def test_scalar_integer_register_bounds_match_original(name, width):
    import qualtran

    source = getattr(qualtran, name)(width)
    for value in (
        -9,
        -1,
        0,
        1,
        7,
        8,
        np.int8(1),
        np.uint8(1),
        True,
        1.0,
        np.array([1]),
        np.bool_(True),
    ):
        try:
            source.assert_valid_classical_val(value)
        except ValueError:
            with pytest.raises(ValueError):
                validate_integer_word_value(value, width, signed=name == "QInt")
        else:
            validate_integer_word_value(value, width, signed=name == "QInt")
