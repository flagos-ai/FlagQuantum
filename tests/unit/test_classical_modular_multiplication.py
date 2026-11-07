"""Source doubling and constant multiplication preserve raw value policies."""

import numpy as np
import pytest
import sympy

from flagquantum.simulation.classical_modular_multiplication import (
    controlled_modular_multiply_constant_classical_values,
    modular_double_classical_value,
)
from tests.unit.test_classical_modular_addition import compare_value_calls

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
VALUES = [
    -1,
    0,
    1,
    2,
    3,
    9,
    np.uint8(200),
    np.array([1]),
    np.array([0, 1]),
    sympy.Integer(2),
]


@pytest.mark.parametrize("value", VALUES)
@pytest.mark.parametrize(
    "modulus", [3, sympy.Integer(3), sympy.Symbol("p", integer=True, positive=True)]
)
def test_source_doubling_add_order_types_exterior_identity_and_errors(value, modulus):
    from qualtran import QUInt
    from qualtran.bloqs.mod_arithmetic.mod_multiplication import ModDbl

    source = ModDbl(QUInt(8), modulus)

    def original(x):
        return (source.on_classical_vals(x=x)["x"],)

    def native(x):
        return (modular_double_classical_value(x, modulus),)

    compare_value_calls(original, native, (value,))


@pytest.mark.parametrize("value", VALUES)
@pytest.mark.parametrize(
    "constant,modulus",
    [
        (2, 3),
        (2, 5),
        (sympy.Integer(2), sympy.Integer(3)),
        (sympy.Symbol("k"), sympy.Symbol("p", integer=True, positive=True)),
    ],
)
@pytest.mark.parametrize("control", [0, 1, 2, True, np.uint8(0), np.uint8(1)])
def test_source_constant_multiply_truthiness_lower_bounds_and_types(
    value, constant, modulus, control
):
    from qualtran import QUInt
    from qualtran.bloqs.mod_arithmetic.mod_multiplication import CModMulK

    source = CModMulK(QUInt(8), constant, modulus)

    def original(c, x):
        return tuple(source.on_classical_vals(ctrl=c, x=x).values())

    def native(c, x):
        return controlled_modular_multiply_constant_classical_values(
            c, x, constant, modulus
        )

    compare_value_calls(original, native, (control, value))


def test_inactive_and_exterior_objects_are_not_normalized():
    value = np.array([99, -1])
    control, output = controlled_modular_multiply_constant_classical_values(
        np.uint8(0), value, 2, 0
    )
    assert isinstance(control, np.uint8) and output is value
    scalar = np.uint8(9)
    assert modular_double_classical_value(scalar, 3) is scalar
    assert controlled_modular_multiply_constant_classical_values(2, 1, 2, 3) == (2, 2)
